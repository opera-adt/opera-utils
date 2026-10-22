"""Shared CMR (Common Metadata Repository) search utilities.

Provides helpers for querying Earthdata's CMR API, including pagination
and URL extraction from UMM metadata.

Since October 2026, CMR's ``granules.umm_json`` results are unreliable for
some collections (NISAR GSLC and GUNW among them): records counted in
``CMR-Hits`` are left out, and under some concept ids another granule's record
is served. ``fetch_cmr_pages(..., verify=True)`` checks a UMM-JSON search
against the atom-JSON ``granules.json`` listing, which is correct, and fetches
every missing or mislabelled record again by concept id.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

from opera_utils.constants import UrlType

__all__ = [
    "CmrSearchReport",
    "fetch_cmr_pages",
    "get_download_url",
]

logger = logging.getLogger("opera_utils")

#: Seconds to wait before each retry of a failed CMR request.
_RETRY_DELAYS: tuple[float, ...] = (2.0, 5.0, 15.0, 30.0)
#: Statuses CMR returns when it is overloaded rather than when a request is bad.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Concept ids per POST when re-asking UMM-JSON for records a search got wrong.
_RECOVER_BATCH = 100


@dataclass
class CmrSearchReport:
    """What CMR got wrong in one verified UMM-JSON search.

    Attributes
    ----------
    listed : int
        Granules the atom-JSON listing returned.
    left_out : list[str]
        Concept ids UMM-JSON did not return.
    mislabelled : dict[str, str]
        Concept id -> the other granule's name UMM-JSON served under it.
    duplicate_names : dict[str, list[str]]
        Granule name -> every concept id CMR lists under it, when more than
        one. The search keeps one record per name.
    unrecovered : list[str]
        Concept ids no endpoint returned (deleted mid-search).

    """

    listed: int = 0
    left_out: list[str] = field(default_factory=list)
    mislabelled: dict[str, str] = field(default_factory=dict)
    duplicate_names: dict[str, list[str]] = field(default_factory=dict)
    unrecovered: list[str] = field(default_factory=list)


def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send a CMR request, retrying timeouts, dropped connections and 429/5xx."""
    kwargs.setdefault("timeout", 60)
    send = requests.get if method == "get" else requests.post
    for delay in (*_RETRY_DELAYS, None):
        try:
            response = send(url, **kwargs)
        except (Timeout, RequestsConnectionError) as err:
            if delay is None:
                raise
            logger.warning("CMR request failed (%s); retrying in %.0f s", err, delay)
        else:
            if response.status_code not in _RETRY_STATUS or delay is None:
                return response
            logger.warning(
                "CMR returned %d; retrying in %.0f s", response.status_code, delay
            )
        time.sleep(delay)
    msg = "unreachable: the last attempt returns or raises"
    raise AssertionError(msg)


def _walk(
    search_url: str, params: dict[str, int | str | list[str]], key: str
) -> list[dict[str, Any]]:
    """Follow ``CMR-Search-After`` through every page; ``key`` picks the records."""
    headers: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    while True:
        response = _request("get", search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if key == "items":
            records.extend(data.get("items", []))
        else:
            records.extend(data.get("feed", {}).get("entry", []))
        if "CMR-Search-After" not in response.headers:
            return records
        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]


def fetch_cmr_pages(
    search_url: str,
    params: dict[str, int | str | list[str]],
    *,
    verify: bool = False,
    report: CmrSearchReport | None = None,
) -> list[dict[str, Any]]:
    """Paginate through all CMR search results and return the UMM items.

    Parameters
    ----------
    search_url : str
        The CMR search endpoint URL
        (e.g. "https://cmr.earthdata.nasa.gov/search/granules.umm_json").
    params : dict[str, int | str | list[str]]
        Query parameters for the CMR request (short_name, page_size, etc.).
    verify : bool
        Check a ``granules.umm_json`` search against the atom-JSON listing of
        the same query, and fetch every record UMM-JSON left out or served
        under the wrong concept id again. The result then holds one record per
        granule name. Costs one extra page walk, plus one request per bad
        record.
    report : CmrSearchReport, optional
        With ``verify``, filled with what CMR got wrong, for reporting it.

    Returns
    -------
    list[dict[str, Any]]
        List of raw UMM item dicts from all pages. Each dict has at least
        a ``"umm"`` key containing the granule UMM metadata.

    """
    items = _walk(search_url, params, "items")
    if not verify or "/granules.umm_json" not in search_url:
        return items
    names = _list_granule_names(search_url, params)
    if not names:
        # Nothing to check against: return UMM-JSON as it came.
        return items
    return _verified(search_url, items, names, report)


def _list_granule_names(
    search_url: str, params: dict[str, int | str | list[str]]
) -> dict[str, str]:
    """Concept id -> granule name for the same query, from atom-JSON."""
    atom_url = search_url.replace("/granules.umm_json", "/granules.json")
    return {
        entry["id"]: entry.get("title", "")
        for entry in _walk(atom_url, params, "entry")
        if "id" in entry
    }


def _concept_id(item: dict[str, Any]) -> str:
    return item.get("meta", {}).get("concept-id", "")


def _granule_ur(item: dict[str, Any]) -> str:
    return item.get("umm", {}).get("GranuleUR", "")


def _concept_number(concept_id: str) -> int:
    digits = concept_id[1:].split("-", 1)[0]
    return int(digits) if digits.isdigit() else -1


def _verified(
    search_url: str,
    items: list[dict[str, Any]],
    names: dict[str, str],
    report: CmrSearchReport | None,
) -> list[dict[str, Any]]:
    """Replace the UMM-JSON records that disagree with atom-JSON, one per name."""
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        by_id.setdefault(_concept_id(item), item)
    unidentified = by_id.pop("", None)
    wrong = [c for c in names if c in by_id and _granule_ur(by_id[c]) != names[c]]
    if report is not None:
        report.listed = len(names)
        report.left_out = [c for c in names if c not in by_id]
        report.mislabelled = {c: _granule_ur(by_id[c]) for c in wrong}
    for cid in wrong:
        del by_id[cid]
    missing = [c for c in names if c not in by_id]
    if missing:
        logger.warning(
            "CMR UMM-JSON left out or mislabelled %d of %d granules; "
            "fetching them by concept id",
            len(missing),
            len(names),
        )
        by_id.update(_recover(search_url, missing, names))
    if report is not None:
        report.unrecovered = [c for c in missing if c not in by_id]

    by_name: dict[str, list[str]] = defaultdict(list)
    for cid, item in by_id.items():
        by_name[_granule_ur(item)].append(cid)
    duplicates = {
        n: sorted(c, key=_concept_number) for n, c in by_name.items() if len(c) > 1
    }
    if duplicates:
        logger.warning(
            "CMR lists %d granule names under more than one concept id; "
            "keeping the most recent of each",
            len(duplicates),
        )
    if report is not None:
        report.duplicate_names = duplicates
    kept = [by_id[max(c, key=_concept_number)] for c in by_name.values()]
    return kept + ([unidentified] if unidentified is not None else [])


def _recover(
    search_url: str, concept_ids: list[str], names: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Fetch the right UMM-G record of each concept id.

    UMM-JSON is asked again by concept id first, keeping only records whose
    name matches atom-JSON; the rest come from the concepts endpoint, which
    serves the native UMM-G record.
    """
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(concept_ids), _RECOVER_BATCH):
        batch = concept_ids[start : start + _RECOVER_BATCH]
        response = _request(
            "post",
            search_url,
            data={"concept_id[]": batch, "page_size": 2 * len(batch)},
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json().get("items", []):
            cid = _concept_id(item)
            if cid in names and _granule_ur(item) == names[cid]:
                found.setdefault(cid, item)

    rest = [c for c in concept_ids if c not in found]
    concept_url = search_url.split("/search/", 1)[0] + "/search/concepts/{}"

    def native(cid: str) -> dict[str, Any] | None:
        response = _request("get", concept_url.format(cid))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return {"meta": {"concept-id": cid}, "umm": response.json()}

    with ThreadPoolExecutor(max_workers=4) as executor:
        for cid, item in zip(rest, executor.map(native, rest)):
            if item is not None:
                found[cid] = item
    return found


def get_download_url(
    umm_data: dict[str, Any],
    protocol: UrlType = UrlType.HTTPS,
    filename_suffix: str | None = None,
) -> str:
    """Extract a download URL from a product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's UMM metadata dictionary.
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https".
    filename_suffix : str | None, optional
        If given, prefer URLs that end with this suffix (e.g. ".h5").
        Falls back to any matching URL if no suffixed match is found.

    Returns
    -------
    str
        The download URL.

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol
        is invalid.

    """
    if protocol not in ["https", "s3"]:
        msg = f"Unknown protocol {protocol}; must be https or s3"
        raise ValueError(msg)

    proto_str = str(protocol)

    # First pass: try to match protocol + suffix
    if filename_suffix is not None:
        for url in umm_data.get("RelatedUrls", []):
            if (
                url["Type"].startswith("GET DATA")
                and url["URL"].startswith(proto_str)
                and url["URL"].endswith(filename_suffix)
            ):
                return url["URL"]

    # Second pass (or only pass if no suffix): match protocol only
    for url in umm_data.get("RelatedUrls", []):
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(proto_str):
            return url["URL"]

    granule_id = umm_data.get("GranuleUR", "unknown")
    msg = f"No download URL found for granule {granule_id}"
    raise ValueError(msg)
