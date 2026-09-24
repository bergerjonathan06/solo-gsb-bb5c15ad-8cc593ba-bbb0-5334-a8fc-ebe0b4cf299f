"""复核共识记忆 (review-consensus memory).

Single and batch matches are scored exactly as before.  This layer only runs
*after* scoring, and only against results that landed in ``needs_review``.

For a pending query it looks at the history of **reviewed** ``match_logs``
records whose query normalises to the same key -- road (street) tokens,
locality tokens and house number, each sorted/order-independent -- and:

* ``>= MIN_EVIDENCE`` reviewed records exist for the key;
* every one of them confirms the **same** still-valid standard address
  (``confirmed``/``overridden`` with one shared ``selected_id``);
* there is no rejection and no record pointing elsewhere (or nowhere);

then the pending result is automatically re-pointed at that address: the
target is moved to the front of the candidate list and the status becomes
``auto_matched``.  If the consensus target is not among this run's
candidates, or the historical conclusions conflict, the original judgment is
kept and a human-readable reason is returned instead.

New reviews take effect immediately because evidence is read live on every
match request; historical records are never rewritten.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .normalization import Normalized, normalize

# A consensus needs at least this many independent reviewed confirmations.
MIN_EVIDENCE = 2


# --------------------------------------------------------------- normalised key


def norm_key(query: str) -> dict[str, Any]:
    """Canonical consensus key for a raw query string."""
    return key_from_normalized(normalize(query))


def key_from_normalized(n: Normalized) -> dict[str, Any]:
    """Same definition as the same-case review grouping key."""
    return {
        "street": sorted(n.street_tokens),
        "locality": sorted(n.locality_tokens),
        "house": n.house or "",
    }


def key_digest(key: dict[str, Any]) -> str:
    return json.dumps(key, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------- evidence


def load_evidence_map(conn: sqlite3.Connection) -> dict[str, list[sqlite3.Row]]:
    """Group every human-reviewed match log by its normalised consensus key.

    Read once per match request (or batch) so every query in a batch is judged
    against the same evidence snapshot.
    """
    rows = conn.execute(
        """
        SELECT id, query, review_action, selected_id
          FROM match_logs
         WHERE status = 'reviewed' AND review_action IS NOT NULL
         ORDER BY id
        """
    ).fetchall()
    buckets: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        digest = key_digest(norm_key(row["query"]))
        buckets.setdefault(digest, []).append(row)
    return buckets


# ------------------------------------------------------------------ deciding


def evaluate(evidence: list[sqlite3.Row], valid_address_ids: set[int]
             ) -> dict[str, Any]:
    """Decide the consensus for one normalised key from its reviewed evidence.

    Returns a payload describing either a hit or why no consensus applies.
    ``hit is None`` means "keep the original judgment"; ``hit`` set to an
    address id means "re-point to that address".
    """
    base = {
        "reviewed_count": len(evidence),
        "evidence_count": 0,
        "evidence_match_ids": [r["id"] for r in evidence],
        "hit": None,
        "reason": "",
    }
    if not evidence:
        return base

    rejected = [r for r in evidence if r["review_action"] == "rejected"]
    decided = [r for r in evidence if r["selected_id"] is not None]

    # Anything that is not a clean "confirmed this address" breaks unanimity:
    # an explicit rejection, a reviewed "no match" (selected_id IS NULL), or
    # two different target addresses.
    if rejected:
        base["reason"] = (
            f"历史复核记录中存在 {len(rejected)} 条拒绝（无匹配）结论，"
            "共识不成立，保持原判"
        )
        return base
    if len(decided) < len(evidence):
        base["reason"] = "历史复核结论中存在未指向有效标准地址的记录，共识不成立，保持原判"
        return base

    targets = {r["selected_id"] for r in decided}
    if len(targets) > 1:
        base["reason"] = "历史复核结论指向多个不同的标准地址，结论冲突，保持原判"
        return base

    target_id = next(iter(targets))
    base.update(
        evidence_count=len(decided),
        evidence_match_ids=[r["id"] for r in decided],
    )
    if len(decided) < MIN_EVIDENCE:
        base["reason"] = (
            f"同一归一化道路/地区词元与门牌号的已复核记录仅 {len(decided)} 条"
            f"（至少需要 {MIN_EVIDENCE} 条），共识不成立，保持原判"
        )
        return base
    if target_id not in valid_address_ids:
        base["reason"] = "历史共识指向的标准地址已不存在，共识不成立，保持原判"
        return base

    base["hit"] = target_id
    base["reason"] = (
        f"复核共识命中：{len(decided)} 条已复核记录一致确认同一标准地址"
        f"（# {target_id}），自动指向该地址"
    )
    return base


def consensus_for_key(evidence_map: dict[str, list[sqlite3.Row]],
                      key: dict[str, Any],
                      valid_address_ids: set[int]) -> dict[str, Any]:
    return evaluate(evidence_map.get(key_digest(key), []), valid_address_ids)


# ------------------------------------------------------------------- applying


def apply_consensus(result: Any, consensus: dict[str, Any]) -> dict[str, Any]:
    """Re-point a scored :class:`MatchResult` according to a consensus payload.

    Only ``needs_review`` results are eligible; auto-matched and no-match
    judgments are never overridden.  On a hit the target is moved to the front
    of the (already computed) candidate list.  When the consensus target is not
    among this run's candidates the original judgment is kept with a reason.
    Returns the payload that should be attached to the API response.
    """
    if result.status != "needs_review":
        return {}

    # No reviewed history at all for this normalised key -> nothing to say,
    # leave the plain scored judgment untouched.
    if not consensus.get("reviewed_count"):
        return {}

    target_id = consensus.get("hit")
    if target_id is None:
        return {"consensus": consensus}

    target = next(
        (c for c in result.candidates if c.address["id"] == target_id),
        None,
    )
    if target is None:
        blocked = dict(consensus)
        blocked["hit"] = None
        blocked["reason"] = (
            f"历史复核共识指向标准地址 #{target_id}，"
            "但该地址不在本次匹配候选中，保持原判"
        )
        return {"consensus": blocked}

    others = [c for c in result.candidates if c.address["id"] != target_id]
    result.candidates = [target] + others
    result.best = target
    result.runner_up = others[0] if others else None
    result.status = "auto_matched"
    result.reason = consensus["reason"]
    return {"consensus": consensus}
