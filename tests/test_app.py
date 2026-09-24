"""Tests for normalisation, scoring and the FastAPI end-to-end flow."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402
from app.matcher import (  # noqa: E402
    levenshtein_ratio,
    match_address,
    score_house,
)
from app.normalization import normalize  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------- normalisation


def test_unicode_and_case_unification():
    n = normalize("１２３ ＭＡＩＮ ＳＴＲＥＥＴ")
    # house number is split out of the comparable text
    assert n.house == "123"
    assert "123" not in n.text
    assert "main" in n.tokens
    # street -> st
    assert "st" in n.tokens


def test_abbreviations_and_diacritics():
    n = normalize("Müller Straße 45")
    assert "muller" in n.tokens
    assert n.house == "45"


def test_house_number_formats():
    assert normalize("12 Main St").house == "12"
    assert normalize("#12-B Main St").house == "12-b"
    assert normalize("No. 12b Main Road").house == "12b"
    assert normalize("北京东路88号").house == "88"
    # 5-digit trailing value is a postcode, not a house number
    assert normalize("Main St, Austin TX 73301").house is None
    # building refs don't grab the house number
    assert normalize("99 SW 26th Ave Unit 7B, Portland").house == "99"


def test_cjk_traditional_to_simplified_and_fullwidth():
    n = normalize("北京東路８８號 3楼")
    assert n.house == "88"
    assert "东" in n.tokens          # 東 -> 东
    assert "北京" in n.tokens        # bigram


# ------------------------------------------------------------------ scoring


def test_levenshtein_ratio():
    assert levenshtein_ratio("main", "main") == 1.0
    assert levenshtein_ratio("", "x") == 0.0
    assert levenshtein_ratio("main", "mn") < 0.7


def test_house_scoring():
    assert score_house("12", "12") == 1.0
    assert score_house("12b", "12") == 0.7
    assert score_house(None, None) == 0.5
    assert score_house("12", None) == 0.15
    assert score_house("100", "1000") < 0.5
    assert score_house("12", "99") == 0.0


# ------------------------------------------------------------- end-to-end API


def test_health_and_sample_loaded(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["address_count"] >= 15


def test_web_page_and_static(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "地址匹配" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_match_abbreviation_auto(client):
    r = client.post("/api/match", json={"query": "123 main st., New York, NY 10001"})
    data = r.json()
    assert data["status"] == "auto_matched"
    assert data["best_id"] == 1
    assert data["best_score"] >= 0.8


def test_match_house_conflict_needs_review(client):
    r = client.post("/api/match", json={"query": "1010 Lake Dr, Austin TX"})
    data = r.json()
    # street tokens match fully but the closest house number differs -> review
    best = data["candidates"][0]
    assert best["token_f1"] >= 0.99
    assert data["status"] == "needs_review"
    conflict_fields = {c["field"] for c in best["conflicts"]}
    assert "house_number" in conflict_fields


def test_match_with_unit_suffix(client):
    r = client.post("/api/match",
                    json={"query": "99 SW 26th Ave Unit 7B, Portland OR"})
    data = r.json()
    assert data["normalized"]["house"] == "99"
    assert data["best_id"] == 14
    assert data["status"] == "auto_matched"


def test_match_terrace_abbreviation(client):
    r = client.post("/api/match", json={"query": "742 Evergreen Tce"})
    data = r.json()
    assert data["best_id"] == 15
    assert data["status"] == "auto_matched"


def test_match_chinese(client):
    r = client.post("/api/match", json={"query": "北京東路８８號 3楼, 上海"})
    data = r.json()
    assert data["normalized"]["house"] == "88"
    assert data["best_id"] == 10
    assert data["status"] == "auto_matched"


def test_match_no_match(client):
    r = client.post("/api/match", json={"query": "1 Milky Way, Mars Base"})
    assert r.json()["status"] == "no_match"


def test_batch_and_review_flow(client):
    r = client.post("/api/match/batch", json={
        "queries": ["Hillcrest Road, Dallas", "123 main st ny"],
    })
    data = r.json()
    assert data["summary"]["total"] == 2
    # ambiguous query (50 vs 52 Hillcrest, no house number) needs review
    ambiguous = data["results"][0]
    assert ambiguous["status"] == "needs_review"
    log_id = ambiguous["match_id"]

    # review queue lists it
    r = client.get("/api/matches", params={"status": "needs_review"})
    ids = [item["id"] for item in r.json()["items"]]
    assert log_id in ids

    # confirm the system's preferred candidate
    r = client.post(f"/api/matches/{log_id}/review", json={"action": "confirmed"})
    assert r.status_code == 200
    r = client.get(f"/api/matches/{log_id}")
    assert r.json()["status"] == "reviewed"
    assert r.json()["review_action"] == "confirmed"


def test_override_review(client):
    r = client.post("/api/match", json={"query": "Hillcrest Rd, Dallas TX"})
    log_id = r.json()["match_id"]
    candidates = r.json()["candidates"]
    second = candidates[1]["address"]["id"]
    r = client.post(f"/api/matches/{log_id}/review",
                    json={"action": "overridden", "selected_id": second})
    assert r.status_code == 200
    detail = client.get(f"/api/matches/{log_id}").json()
    assert detail["review_action"] == "overridden"
    assert detail["selected_id"] == second


def test_csv_upload_and_dedup(client):
    csv_bytes = b"id,full_address,town,zip\nX1,9 Test Lane,Testville,99999\n"
    r = client.post("/api/addresses/upload",
                    files={"file": ("x.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    assert r.json()["inserted"] == 1
    r2 = client.post("/api/addresses/upload",
                     files={"file": ("x.csv", csv_bytes, "text/csv")})
    assert r2.json()["inserted"] == 0
    assert r2.json()["duplicates"] == 1

    r = client.post("/api/match", json={"query": "9 Test Ln, Testville 99999"})
    assert r.json()["best_id"] is not None


# ----------------------------------------------------- CSV preview / confirm


def _preview_csv(prefix: str) -> bytes:
    # line 2 valid, line 3 invalid (all blanks), line 4 valid,
    # line 5 an in-file repeat of line 2 (duplicate).
    return (
        "external_id,address,city,zip\n"
        f"{prefix}-001,1 Preview Way,Previewton,10001\n"
        ",,,\n"
        f"{prefix}-002,2 Preview Ave,Previewton,10002\n"
        f"{prefix}-001,1 Preview Way,Previewton,10001\n"
    ).encode()


def test_preview_then_confirm_writes_only_valid(client):
    csv_bytes = _preview_csv("PA")
    before = client.get("/api/health").json()["address_count"]
    r = client.post("/api/addresses/preview",
                    files={"file": ("p.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] == 2
    assert data["duplicates"] == 1
    assert data["invalid"] == 1
    assert data["invalid_lines"] == [3]
    assert data["total"] == 4
    token = data["preview_token"]

    # Preview must not have written anything yet.
    assert client.get("/api/health").json()["address_count"] == before

    # Confirm requires the file to be re-submitted alongside the token.
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("p.csv", csv_bytes, "text/csv")},
        data={"preview_token": token},
    )
    assert r.status_code == 200
    result = r.json()
    assert result["inserted"] == 2
    assert result["duplicates"] == 1
    assert result["invalid"] == 1
    assert result["invalid_lines"] == [3]
    assert client.get("/api/health").json()["address_count"] == before + 2

    # The voucher is single-use: confirming again fails.
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("p.csv", csv_bytes, "text/csv")},
        data={"preview_token": token},
    )
    assert r.status_code == 404


def test_preview_uses_header_aliases(client):
    csv_bytes = b"id,addr,town,postcode\nPAL,7 Alpha Road,Alphaville,20001\n"
    r = client.post("/api/addresses/preview",
                    files={"file": ("a.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] == 1
    assert data["invalid"] == 0
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("a.csv", csv_bytes, "text/csv")},
        data={"preview_token": data["preview_token"]},
    )
    assert r.status_code == 200
    assert r.json()["inserted"] == 1


def test_preview_confirm_rejects_changed_file(client):
    csv_bytes = _preview_csv("PB")
    r = client.post("/api/addresses/preview",
                    files={"file": ("p.csv", csv_bytes, "text/csv")})
    token = r.json()["preview_token"]
    before = client.get("/api/health").json()["address_count"]

    # A different file body carrying the same token must be refused.
    tampered = csv_bytes.replace(b"2 Preview Ave", b"99 Other Street")
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("p.csv", tampered, "text/csv")},
        data={"preview_token": token},
    )
    assert r.status_code == 409
    # Nothing was written.
    assert client.get("/api/health").json()["address_count"] == before

    # The voucher is not consumed by the failed attempt, but the original file
    # was never confirmed; re-preview fresh (unique ids) data and confirm it.
    fresh = (
        "external_id,address\nPB2-001,3 Fresh Court\nPB2-002,4 Fresh Plaza\n"
    ).encode()
    r = client.post("/api/addresses/preview",
                    files={"file": ("f.csv", fresh, "text/csv")})
    assert r.status_code == 200
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("f.csv", fresh, "text/csv")},
        data={"preview_token": r.json()["preview_token"]},
    )
    assert r.status_code == 200
    assert r.json()["inserted"] == 2


def test_preview_confirm_rejects_changed_table(client):
    csv_bytes = _preview_csv("PC")
    r = client.post("/api/addresses/preview",
                    files={"file": ("p.csv", csv_bytes, "text/csv")})
    token = r.json()["preview_token"]
    before = client.get("/api/health").json()["address_count"]

    # Another writer (the legacy direct-upload endpoint) mutates the table.
    other = b"id,address\nPCINT,8 Interloper Lane\n"
    assert client.post("/api/addresses/upload",
                       files={"file": ("o.csv", other, "text/csv")}).status_code == 200

    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("p.csv", csv_bytes, "text/csv")},
        data={"preview_token": token},
    )
    assert r.status_code == 409
    # Only the interloper was written; the stale preview wrote nothing.
    assert client.get("/api/health").json()["address_count"] == before + 1

    # Re-preview fresh (unique ids) data against the new snapshot, then confirm.
    fresh = (
        "external_id,address\nPC2-001,5 New Crescent\nPC2-002,6 New Gardens\n"
    ).encode()
    r = client.post("/api/addresses/preview",
                    files={"file": ("f.csv", fresh, "text/csv")})
    assert r.status_code == 200
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("f.csv", fresh, "text/csv")},
        data={"preview_token": r.json()["preview_token"]},
    )
    assert r.status_code == 200
    assert r.json()["inserted"] == 2


def test_preview_confirm_unknown_token(client):
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("p.csv", _preview_csv("PD"), "text/csv")},
        data={"preview_token": "does-not-exist"},
    )
    assert r.status_code == 404


def test_preview_rejects_non_csv(client):
    r = client.post("/api/addresses/preview",
                    files={"file": ("p.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_preview_invalid_lines_use_physical_numbers(client):
    # A blank physical line between data rows is skipped by the csv module and
    # not counted; the trailing invalid row still reports its true line number.
    content = (
        b"external_id,address\n"   # line 1 header
        b"PLN1,11 Line Lane\n"     # line 2 valid
        b"\n"                       # line 3 blank -> skipped
        b",\n"                      # line 4 invalid
    )
    r = client.post("/api/addresses/preview",
                    files={"file": ("l.csv", content, "text/csv")})
    data = r.json()
    assert data["valid"] == 1
    assert data["invalid"] == 1
    assert data["invalid_lines"] == [4]


# ------------------------------------------------- same-case merged review


def _match_pending(client, query: str) -> int:
    r = client.post("/api/match", json={"query": query})
    data = r.json()
    assert data["status"] == "needs_review", data
    return data["match_id"]


def _group_by_house(client, house: str) -> dict:
    r = client.get("/api/review-groups")
    assert r.status_code == 200
    matches = [g for g in r.json()["groups"] if g["house"] == house]
    assert len(matches) == 1, matches
    return matches[0]


def test_review_groups_merge_same_case_and_confirm(client):
    # Three differently-written queries that normalise to the same
    # road/locality tokens + house number and therefore share one group.
    ids = [
        _match_pending(client, "300 Lake Dr, Austin TX"),
        _match_pending(client, "No. 300 Lake Dr, Austin TX"),
        _match_pending(client, "300 LAKE DR, AUSTIN TX"),
    ]
    g = _group_by_house(client, "300")
    assert g["count"] == 3
    assert sorted(g["match_ids"]) == sorted(ids)
    # the group exposes the original queries of its members
    assert {m["query"] for m in g["queries"]} == {
        "300 Lake Dr, Austin TX",
        "No. 300 Lake Dr, Austin TX",
        "300 LAKE DR, AUSTIN TX",
    }
    assert g["street_tokens"] == ["dr", "lake"]
    assert g["locality_tokens"] == ["austin", "tx"]

    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": g["match_ids"],
        "action": "confirmed",
    })
    assert r.status_code == 200
    assert r.json()["reviewed"] == 3
    for mid in ids:
        d = client.get(f"/api/matches/{mid}").json()
        assert d["status"] == "reviewed"
        assert d["review_action"] == "confirmed"
        assert d["selected_id"] == d["candidate_id"]

    # the whole group left the pending queue
    houses = [g2["house"] for g2 in client.get("/api/review-groups").json()["groups"]]
    assert "300" not in houses


def test_review_group_override_and_stale_member_set(client):
    ids = [
        _match_pending(client, "301 Lake Dr, Austin TX"),
        _match_pending(client, "301 Lake Drive, Austin TX"),
    ]
    g = _group_by_house(client, "301")
    assert sorted(g["match_ids"]) == sorted(ids)
    second = g["candidates"][1]["address"]["id"]
    second_score = g["candidates"][1]["score"]

    # stale member set -> conflict, and nothing is written
    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": ids + [999999],
        "action": "overridden", "selected_id": second,
    })
    assert r.status_code == 409
    for mid in ids:
        assert client.get(f"/api/matches/{mid}").json()["status"] == "needs_review"

    # the override target must be one of the shared candidates
    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": ids,
        "action": "overridden", "selected_id": 1,
    })
    assert r.status_code == 400
    for mid in ids:
        assert client.get(f"/api/matches/{mid}").json()["status"] == "needs_review"

    # the real override applies the same decision + score to every member
    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": ids,
        "action": "overridden", "selected_id": second,
    })
    assert r.status_code == 200
    assert r.json()["reviewed"] == 2
    for mid in ids:
        d = client.get(f"/api/matches/{mid}").json()
        assert d["status"] == "reviewed"
        assert d["review_action"] == "overridden"
        assert d["selected_id"] == second
        assert d["score"] == second_score


def test_review_group_member_reviewed_individually_conflicts(client):
    ids = [
        _match_pending(client, "302 Lake Dr, Austin TX"),
        _match_pending(client, "302 LAKE DR, AUSTIN TX"),
    ]
    g = _group_by_house(client, "302")

    # one member gets reviewed through the original single-record endpoint
    r = client.post(f"/api/matches/{ids[0]}/review", json={"action": "rejected"})
    assert r.status_code == 200

    # the group review now conflicts and writes nothing at all
    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": ids, "action": "confirmed",
    })
    assert r.status_code == 409
    assert client.get(f"/api/matches/{ids[1]}").json()["status"] == "needs_review"


def test_review_group_reject(client):
    ids = [
        _match_pending(client, "303 Lake Dr, Austin TX"),
        _match_pending(client, "No. 303 Lake Dr, Austin TX"),
    ]
    g = _group_by_house(client, "303")
    r = client.post("/api/review-groups/review", json={
        "group_key": g["group_key"], "match_ids": ids, "action": "rejected",
    })
    assert r.status_code == 200
    assert r.json()["reviewed"] == 2
    for mid in ids:
        d = client.get(f"/api/matches/{mid}").json()
        assert d["status"] == "reviewed"
        assert d["review_action"] == "rejected"
        assert d["selected_id"] is None


def test_review_groups_split_when_candidates_differ(client):
    # Same normalised key, but the address table changes in between so the
    # candidate IDs/order/scores differ -> the records must NOT share a group.
    first_id = _match_pending(client, "304 Lake Dr, Austin TX")
    r = client.post("/api/addresses", json={"addresses": [
        {"address": "305 Lake Drive, Austin, TX 73301", "external_id": "GRP-305"},
    ]})
    assert r.json()["inserted"] == 1
    second_id = _match_pending(client, "304 LAKE DR, AUSTIN TX")

    groups = [g for g in client.get("/api/review-groups").json()["groups"]
              if g["house"] == "304"]
    assert len(groups) == 2
    assert all(g["count"] == 1 for g in groups)
    assert {g["match_ids"][0] for g in groups} == {first_id, second_id}
    assert groups[0]["group_key"] != groups[1]["group_key"]

    # a singleton group can still be reviewed as a group (member set of one)
    g1 = next(g for g in groups if g["match_ids"] == [first_id])
    r = client.post("/api/review-groups/review", json={
        "group_key": g1["group_key"], "match_ids": [first_id],
        "action": "confirmed",
    })
    assert r.status_code == 200
    assert r.json()["reviewed"] == 1
    assert client.get(f"/api/matches/{first_id}").json()["status"] == "reviewed"


# --------------------------------------------- review-consensus memory --


def _add_standard(client, external_id: str, address: str, *,
                  city: str = "Consensuston", state: str = "NJ",
                  postal_code: str = "70001") -> int:
    r = client.post("/api/addresses", json={"addresses": [{
        "external_id": external_id, "address": address,
        "city": city, "state": state, "postal_code": postal_code,
    }]})
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1, r.json()
    listing = client.get("/api/addresses?limit=500").json()["items"]
    return next(a["id"] for a in listing if a["external_id"] == external_id)


def _review(client, match_id: int, action: str, selected_id: int | None = None):
    body = {"action": action}
    if selected_id is not None:
        body["selected_id"] = selected_id
    r = client.post(f"/api/matches/{match_id}/review", json=body)
    assert r.status_code == 200, r.text


def _match_saved(client, query: str) -> dict:
    r = client.post("/api/match", json={"query": query})
    assert r.status_code == 200, r.text
    return r.json()


def test_consensus_auto_repoints_after_two_confirmations(client):
    # Two standards on the same street; queries with a *different* house number
    # always stay below the auto threshold and need review.
    beacon1 = _add_standard(client, "CN-B1", "1 Beacon Court")
    beacon2 = _add_standard(client, "CN-B2", "3 Beacon Court")

    q1 = "7 Beacon Court, Consensuston, NJ 70001"
    q2 = "7 Beacon Ct, Consensuston, NJ 70001"
    q3 = "#7 Beacon Ct., Consensuston NJ 70001"

    first = _match_saved(client, q1)
    second = _match_saved(client, q2)
    assert first["status"] == "needs_review"
    assert second["status"] == "needs_review"
    # no reviewed history yet -> no consensus payload at all
    assert first["consensus"] is None

    # Reviewers override both earlier cases to the SAME target (beacon1).
    _review(client, first["match_id"], "overridden", beacon1)
    _review(client, second["match_id"], "overridden", beacon1)

    # A new, differently-written query normalising to the same road/locality
    # tokens + house number is now auto-re-pointed by the consensus memory.
    third = _match_saved(client, q3)
    assert third["status"] == "auto_matched"
    assert third["best_id"] == beacon1
    cons = third["consensus"]
    assert cons["hit"] == beacon1
    assert cons["evidence_count"] == 2
    assert set(cons["evidence_match_ids"]) == {first["match_id"], second["match_id"]}
    # the consensus target is presented as the first candidate
    assert third["candidates"][0]["address"]["id"] == beacon1
    assert beacon1 != second["best_id"]  # scoring itself had ranked beacon2 first

    # New reviews take effect immediately, but old records were not rewritten.
    old = client.get(f"/api/matches/{first['match_id']}").json()
    assert old["status"] == "reviewed"
    assert old["selected_id"] == beacon1


def test_consensus_conflict_on_rejection_keeps_original(client):
    maple1 = _add_standard(client, "CN-M1", "30 Maple Drive",
                          city="Mapleton", postal_code="60126")
    maple2 = _add_standard(client, "CN-M2", "40 Maple Drive",
                          city="Mapleton", postal_code="60126")

    id_a = _match_saved(client, "35 Maple Drive, Mapleton, IL 60126")["match_id"]
    id_b = _match_saved(client, "35 Maple Dr, Mapleton, IL 60126")["match_id"]
    # one confirmation, one rejection -> history is not unanimous
    _review(client, id_a, "overridden", maple1)
    _review(client, id_b, "rejected")

    data = _match_saved(client, "35 MAPLE DRIVE, Mapleton, IL 60126")
    assert data["status"] == "needs_review"  # original judgment kept
    cons = data["consensus"]
    assert cons["hit"] is None
    assert "拒绝" in cons["reason"]
    assert cons["reviewed_count"] == 2
    # neither historical record was rewritten
    assert client.get(f"/api/matches/{id_b}").json()["review_action"] == "rejected"
    assert maple2 is not None


def test_consensus_conflict_on_different_targets(client):
    aspen1 = _add_standard(client, "CN-A1", "10 Aspen Way")
    aspen2 = _add_standard(client, "CN-A2", "20 Aspen Way")

    id_a = _match_saved(client, "15 Aspen Way, Consensuston, NJ 70001")["match_id"]
    id_b = _match_saved(client, "no. 15 Aspen Way, Consensuston, NJ 70001")["match_id"]
    _review(client, id_a, "overridden", aspen1)
    _review(client, id_b, "overridden", aspen2)

    data = _match_saved(client, "15 ASPEN WAY, CONSENSUSTON, NJ 70001")
    assert data["status"] == "needs_review"
    cons = data["consensus"]
    assert cons["hit"] is None
    assert "冲突" in cons["reason"]


def test_consensus_needs_two_evidences(client):
    cedar = _add_standard(client, "CN-SO1", "14 Cedar Lane",
                          city="Oaktown", postal_code="60126")

    only = _match_saved(client, "18 Cedar Lane, Oaktown, IL 60126")
    assert only["status"] == "needs_review"
    _review(client, only["match_id"], "overridden", cedar)

    # only one reviewed record -> below the two-evidence bar, stays pending
    data = _match_saved(client, "18 Cedar Ln, Oaktown, IL 60126")
    assert data["status"] == "needs_review"
    cons = data["consensus"]
    assert cons["hit"] is None
    assert cons["reviewed_count"] == 1
    assert "1 条" in cons["reason"]


def test_consensus_target_missing_from_candidates_keeps_original(client):
    # The consensus target carries extra street words ("Grand Old") so its
    # token F1 against the query is lower than exact competitors; once five
    # better candidates exist, the target drops out of the top-5 candidate
    # list even though its house number matches best.
    target = _add_standard(client, "CN-T", "12 Grand Old Birch Road",
                           city="Riverton", postal_code="61000")
    pending = _match_saved(client, "10 Birch Road, Riverton, IL 61000")
    assert pending["status"] == "needs_review"
    assert pending["best_id"] == target
    id_a = pending["match_id"]
    id_b = _match_saved(client, "No. 10 Birch Road, Riverton, IL 61000")["match_id"]
    assert id_a != id_b
    _review(client, id_a, "overridden", target)
    _review(client, id_b, "overridden", target)

    for i, house in enumerate(["11", "9", "8", "13", "12"]):
        _add_standard(client, f"CN-C{i}", f"{house} Birch Road",
                      city="Riverton", postal_code="61000")

    data = _match_saved(client, "10 Birch Rd, Riverton, IL 61000")
    # every competitor outscores the consensus target but stays below the
    # auto threshold (wrong house numbers) -> original judgment still pending
    assert data["status"] == "needs_review"
    candidate_ids = [c["address"]["id"] for c in data["candidates"]]
    assert target not in candidate_ids
    assert len(candidate_ids) == 5
    cons = data["consensus"]
    assert cons["hit"] is None
    assert cons["evidence_count"] == 2
    assert "不在本次" in cons["reason"]


def test_consensus_does_not_apply_to_no_match(client):
    # Unit-level guard: consensus never overrides a no_match / auto_matched
    # judgment, only pending ones.
    from app.consensus import apply_consensus

    class Dummy:
        status = "no_match"

    assert apply_consensus(Dummy(), {"hit": 1, "reason": "x"}) == {}


def test_consensus_in_batch_summary(client):
    harbor1 = _add_standard(client, "CN-H1", "2 Harbor Court",
                            city="Harborton", state="ME", postal_code="40010")
    harbor2 = _add_standard(client, "CN-H2", "4 Harbor Court",
                            city="Harborton", state="ME", postal_code="40010")
    id_a = _match_saved(client, "3 Harbor Court, Harborton, ME 40010")["match_id"]
    id_b = _match_saved(client, "3 Harbor Ct, Harborton, ME 40010")["match_id"]
    _review(client, id_a, "overridden", harbor1)
    _review(client, id_b, "overridden", harbor1)

    r = client.post("/api/match/batch", json={"queries": [
        "3 HARBOR COURT, HARBORTON, ME 40010",
        "1 Milky Way, Mars Base",
    ]})
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["consensus_matched"] == 1
    consensus_hit = next(
        i for i in data["results"] if (i.get("consensus") or {}).get("hit") is not None
    )
    assert consensus_hit["status"] == "auto_matched"
    assert consensus_hit["best_id"] == harbor1
    assert consensus_hit["consensus"]["evidence_count"] == 2
    assert harbor2 is not None
