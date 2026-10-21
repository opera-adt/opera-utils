"""Tests for the verified CMR page walk in opera_utils._cmr (no network).

CMR's ``granules.umm_json`` leaves out NISAR records it counts in ``CMR-Hits``
and serves another granule's record under some concept ids (October 2026);
``granules.json`` (atom) lists every granule under its own name.
"""

from __future__ import annotations

import pytest
import requests

from opera_utils import _cmr

UMM = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
NAMES = [f"NISAR_L2_PR_GUNW_{i:03d}" for i in range(7)]


class FakeResponse:
    def __init__(self, body=None, status_code=200, after=None):
        self._body = body or {}
        self.status_code = status_code
        self.headers = {"CMR-Search-After": after} if after else {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeCMR:
    """Pages of two; ``hidden`` never come back from UMM-JSON, ``mislabelled``
    maps a concept id to the one whose record UMM-JSON serves in its place."""

    def __init__(self, names, hidden=(), mislabelled=None, deleted=(), duplicate=None):
        self.ids = {f"G{i:04d}-ASF": n for i, n in enumerate(names)}
        if duplicate:  # a second concept id for an existing name
            self.ids[duplicate[0]] = duplicate[1]
        self.hidden, self.deleted = set(hidden), set(deleted)
        self.mislabelled = mislabelled or {}
        self.failures: list = []
        self.concept_gets: list[str] = []

    def _umm(self, cid):
        return {"GranuleUR": self.ids[cid]}

    def _items(self, cids):
        return [
            {"meta": {"concept-id": c}, "umm": self._umm(self.mislabelled.get(c, c))}
            for c in cids
            if c not in self.hidden
        ]

    def get(self, url, params=None, headers=None, timeout=0):  # noqa: ARG002
        assert timeout > 0
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return FakeResponse(status_code=failure)
        if "/concepts/" in url:
            cid = url.rsplit("/", 1)[1]
            self.concept_gets.append(cid)
            if cid in self.deleted:
                return FakeResponse(status_code=404)
            return FakeResponse(self._umm(cid))
        cids = list(self.ids)
        start = int((headers or {}).get("CMR-Search-After", 0))
        page = cids[start : start + 2]
        after = str(start + 2) if start + 2 < len(cids) else None
        if "umm_json" in url:
            return FakeResponse({"items": self._items(page)}, after=after)
        entries = [{"id": c, "title": self.ids[c]} for c in page]
        return FakeResponse({"feed": {"entry": entries}}, after=after)

    def post(self, url, data=None, timeout=0):  # noqa: ARG002
        assert timeout > 0
        cids = [c for c in data["concept_id[]"] if c in self.ids]
        return FakeResponse({"items": self._items(cids)})


@pytest.fixture
def fake_cmr(monkeypatch):
    monkeypatch.setattr(_cmr, "_RETRY_DELAYS", (0.0, 0.0, 0.0, 0.0))

    def install(names, **kwargs):
        fake = FakeCMR(names, **kwargs)
        monkeypatch.setattr(_cmr, "requests", fake)
        return fake

    return install


def _names(items):
    return sorted(i["umm"]["GranuleUR"] for i in items)


def test_unverified_walk_returns_umm_json_as_it_comes(fake_cmr):
    fake_cmr(NAMES, hidden={"G0001-ASF"}, mislabelled={"G0002-ASF": "G0005-ASF"})

    items = _cmr.fetch_cmr_pages(UMM, {})

    assert len(items) == 6
    assert _names(items).count(NAMES[5]) == 2


def test_verified_walk_recovers_left_out_and_mislabelled_records(fake_cmr):
    cmr = fake_cmr(NAMES, hidden={"G0001-ASF"}, mislabelled={"G0002-ASF": "G0005-ASF"})

    items = _cmr.fetch_cmr_pages(UMM, {}, verify=True)

    assert _names(items) == NAMES
    by_id = {i["meta"]["concept-id"]: i["umm"]["GranuleUR"] for i in items}
    assert by_id["G0002-ASF"] == NAMES[2]
    # UMM-JSON by id serves the same wrong record, so both need the concepts endpoint.
    assert sorted(cmr.concept_gets) == ["G0001-ASF", "G0002-ASF"]


def test_report_and_one_record_per_name(fake_cmr):
    fake_cmr(
        NAMES,
        hidden={"G0003-ASF", "G0006-ASF"},
        deleted={"G0006-ASF"},
        mislabelled={"G0002-ASF": "G0005-ASF"},
        duplicate=("G0100-ASF", NAMES[0]),
    )
    report = _cmr.CmrSearchReport()

    items = _cmr.fetch_cmr_pages(UMM, {}, verify=True, report=report)

    assert _names(items) == sorted(NAMES[:6])
    kept = {i["umm"]["GranuleUR"]: i["meta"]["concept-id"] for i in items}
    assert kept[NAMES[0]] == "G0100-ASF"  # the most recently ingested
    assert report.listed == 8
    assert report.left_out == ["G0003-ASF", "G0006-ASF"]
    assert report.mislabelled == {"G0002-ASF": NAMES[5]}
    assert report.unrecovered == ["G0006-ASF"]
    assert report.duplicate_names == {NAMES[0]: ["G0000-ASF", "G0100-ASF"]}


@pytest.mark.parametrize("failure", [requests.exceptions.ReadTimeout("slow"), 503])
def test_transient_failures_are_retried(fake_cmr, failure):
    cmr = fake_cmr(NAMES)
    cmr.failures = [failure, failure]

    assert _names(_cmr.fetch_cmr_pages(UMM, {}, verify=True)) == NAMES


def test_bad_requests_are_not_retried(fake_cmr):
    cmr = fake_cmr(NAMES)
    cmr.failures = [400]

    with pytest.raises(requests.HTTPError):
        _cmr.fetch_cmr_pages(UMM, {})
    assert cmr.failures == []
