"""Live check that apifare answers the request ``search_apifare`` actually sends.

**This is the test that closes the one open risk in the apifare provider.**
Nothing else in the suite can: ``test_apifare_unit.py`` proves what is *sent* and
what the parser does with a body it is *handed*, but apifare's route slug, its
request body and its response envelope are unconfirmed. apifare's public
documentation shows ``POST /v1/call/{slug}`` with parameters as a JSON body --
the route *shape* matches -- but names no ``dataforseo`` slug, no ``q``/``count``
fields and no ``result.results`` envelope. Section 14 of
``.system_design/TEST_SUITE.md`` carries that as an accepted risk until this test
has been run against a funded account.

**A skipped live test proves nothing**, and this one skips by default. It becomes
evidence only when someone with a token runs it:

    KINDLY_RUN_LIVE_TESTS=1 APIFARE_TOKEN=... pytest tests/test_apifare_live.py

**It spends real money** -- one metered call against a prepaid balance -- which is
the other reason it is not in the default run.

Gated on ``KINDLY_RUN_LIVE_TESTS`` and marked ``live``, per section 6.3, rather
than on the older ``RUN_LIVE_TESTS`` that ``test_serper_live.py`` still reads.
Section 6.3 standardises on this pair; a module written now should land on the
target gate rather than add a third caller to the one being retired.

The assertions are deliberately structural, not exact: a real SERP changes
between runs, so this checks the envelope and the field *types*
``search_apifare`` depends on, never the content of a particular result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kindly_web_search_mcp_server.search.apifare import SEARCH_ENDPOINT, search_apifare

pytestmark = pytest.mark.live


def _live_enabled() -> bool:
    """Report whether live tests are switched on for this run.

    Returns:
        ``True`` when ``KINDLY_RUN_LIVE_TESTS`` holds an affirmative value.
    """
    return os.environ.get("KINDLY_RUN_LIVE_TESTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@pytest.fixture
def live_token() -> str:
    """Return the apifare token, skipping or failing rather than passing quietly.

    A missing token when live tests are *off* is a skip. A missing token when
    they are *on* is a failure: section 6.3's rule is that a skipped live suite
    is a failed live suite, so an enabled job that silently skips would report
    green while proving nothing.

    Returns:
        The token to authenticate with.
    """
    if not _live_enabled():
        pytest.skip("Live tests disabled; set KINDLY_RUN_LIVE_TESTS=1 to enable")

    token = os.environ.get("APIFARE_TOKEN", "").strip()
    assert token, "KINDLY_RUN_LIVE_TESTS is set but APIFARE_TOKEN is missing"
    return token


async def test_the_documented_route_answers_the_request_we_send(
    live_token: str,
) -> None:
    """Confirm the slug, the request body and the response envelope all hold

    Sends the exact request ``search_apifare`` builds -- same URL, same header,
    same body -- and reads the response the way the provider's parser does. A
    404, a 400, or a body shaped some other way fails here, which is the whole
    point: that failure is the evidence the unit tests cannot supply.

    Args:
        live_token: The apifare token, from the gating fixture.
    """
    payload = {"q": "model context protocol", "count": 3}
    headers = {
        "Authorization": f"Bearer {live_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(SEARCH_ENDPOINT, headers=headers, json=payload)

    assert response.status_code != 402, (
        "the apifare balance is empty, so this run proves nothing about the "
        "response envelope -- top up and re-run"
    )
    assert response.status_code == 200, (
        f"apifare answered HTTP {response.status_code} for the documented route; "
        "the slug or the request body is wrong"
    )

    body = response.json()
    assert isinstance(body, dict), "response body is not a JSON object"
    assert isinstance(body.get("result"), dict), "response has no `result` object"

    items = body["result"].get("results")
    assert isinstance(items, list), "`result.results` is not a list"
    assert items, "a live query returned no results at all"

    first = items[0]
    assert isinstance(first, dict), "a result entry is not an object"
    assert isinstance(first.get("title"), str) and first["title"]
    assert isinstance(first.get("url"), str) and first["url"]
    # `description` is optional in the parser, so its absence is not a failure --
    # only a non-string value where one is present would be.
    assert isinstance(first.get("description", ""), str)


async def test_the_provider_returns_usable_results_end_to_end(
    live_token: str,
) -> None:
    """Drive ``search_apifare`` itself so the parser meets the real payload

    The request test above could pass while the parser still rejected the live
    body -- the two assertions are not the same claim.

    Args:
        live_token: The apifare token, from the gating fixture.
    """
    results = await search_apifare("model context protocol", num_results=3)

    assert results, "search_apifare parsed no results out of a live response"
    assert len(results) <= 3
    for result in results:
        assert result.title
        assert result.link.startswith("http")
        # Filled in later by the MCP tool, never by the provider.
        assert result.page_content == ""
