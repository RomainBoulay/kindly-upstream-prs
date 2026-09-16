"""apifare search provider.

Queries apifare's metered ``dataforseo`` route and maps its organic results onto
:class:`~kindly_web_search_mcp_server.models.WebSearchResult`. apifare resells
search against a prepaid balance, so it is configured by a bearer token rather
than a per-vendor API key. It is the last entry in
:data:`~kindly_web_search_mcp_server.search.PROVIDERS`, so it serves a query only
when ``APIFARE_TOKEN`` is the one provider variable configured.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..models import WebSearchResult

SEARCH_ENDPOINT = "https://apifare.com/v1/call/dataforseo"

#: Host whose top-up links may be quoted back to the MCP client, together with
#: its subdomains. An allowlist rather than a denylist: the value being checked
#: is written by the remote end, so anything not recognised must be discarded.
TOPUP_URL_HOST = "apifare.com"

#: Longest top-up URL quoted into an error message. A top-up link is a short
#: dashboard URL; a long one is a payload, not a link.
TOPUP_URL_MAX_LENGTH = 200


class ApifareError(RuntimeError):
    """Report an apifare response this provider cannot turn into results."""


class ApifareConfigError(ApifareError):
    """Report that apifare was called without a usable ``APIFARE_TOKEN``."""


class ApifarePaymentRequiredError(ApifareError):
    """Report an exhausted apifare balance, with the top-up link when it is safe.

    apifare answers an exhausted balance with HTTP 402 and a structured body
    carrying a ``topup_url``. That URL is the actionable half of the error for
    the agent's operator, so it is surfaced rather than collapsed into a bare
    status code -- but only after :func:`_safe_topup_url` accepts it.

    The validation is not optional politeness. This message is served to the MCP
    client, which for this server is an LLM agent, and ``topup_url`` is written
    by the remote end. It is the only string in any provider that travels from a
    provider's response body into an error message; everywhere else the message
    is built from a local label and an HTTP status, and
    :func:`~kindly_web_search_mcp_server.search.searxng.search_searxng` redacts
    even a derived one. Unvalidated, a compromised or hostile endpoint could put
    arbitrary text -- including instructions addressed to the agent -- in front
    of the model under this server's name.
    """


def _get_apifare_token() -> str:
    """Read the apifare bearer token from the environment.

    Returns:
        The token, with surrounding whitespace removed.

    Raises:
        ApifareConfigError: If ``APIFARE_TOKEN`` is unset, empty, or only
            whitespace.
    """
    token = os.environ.get("APIFARE_TOKEN", "").strip()
    if not token:
        raise ApifareConfigError(
            "APIFARE_TOKEN is not set. Configure it as an environment variable in your IDE/run configuration."
        )
    return token


def _safe_topup_url(candidate: object, token: str) -> str | None:
    """Return ``candidate`` if it is a top-up link safe to quote, else ``None``.

    Accepts only an absolute ``https`` URL, free of whitespace and control
    characters, at most :data:`TOPUP_URL_MAX_LENGTH` characters long, carrying no
    userinfo or port, whose host is :data:`TOPUP_URL_HOST` or a subdomain of it,
    and which does not contain ``token`` anywhere.

    Every rule fails closed, which is why the host test is an equality-or-suffix
    check rather than a substring one: ``apifare.com.evil.example`` starts with
    the expected host and ``evil.example/apifare.com`` contains it, and both must
    be rejected. Userinfo is rejected outright because
    ``https://apifare.com@evil.example/`` reads as the real host to a human and
    resolves to the attacker's. Percent-encoded and punycode hosts are left
    undecoded by :func:`~urllib.parse.urlsplit`, so a homograph or an escaped
    spelling of the host simply fails the comparison.

    **The token check is not redundant with the host check.** A URL on the
    genuine host can still carry the bearer token in its query string -- the
    endpoint chooses that string, and this server would then quote the credential
    straight into an LLM's context. A real top-up link has no reason to contain
    the token, so refusing to quote one that does costs nothing and closes the
    disclosure.

    Args:
        candidate: The ``topup_url`` value taken from the response body. Any
            type, since it comes from parsed JSON and need not be a string.
        token: The bearer token this request was sent with, which the returned
            URL must not contain.

    Returns:
        The URL, re-composed from its validated parts, when every rule passes;
        otherwise ``None``.
    """
    if not isinstance(candidate, str):
        return None

    if token in candidate:
        return None

    if not candidate or len(candidate) > TOPUP_URL_MAX_LENGTH:
        return None

    # Whitespace and control characters would let the remote end close this
    # sentence and start one of its own, which is exactly what quoting nothing
    # unrecognised is meant to prevent.
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in candidate):
        return None

    # `urlsplit` raises on a malformed IPv6 literal, and `hostname`/`port` raise
    # on a malformed port, so every read of the parsed parts is guarded.
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        has_userinfo = parts.username is not None or parts.password is not None
        has_port = parts.port is not None
    except ValueError:
        return None

    if parts.scheme != "https" or has_userinfo or has_port or host is None:
        return None

    # A trailing dot names the same host to a resolver, so it is stripped before
    # the comparison rather than allowed to sidestep it. It is *not* stripped from
    # the returned URL: rejecting is the only safe direction, and a later author
    # "fixing" the dot by rewriting the URL would be re-introducing a rewrite this
    # function exists to avoid.
    if host.rstrip(".") != TOPUP_URL_HOST and not host.rstrip(".").endswith(
        f".{TOPUP_URL_HOST}"
    ):
        return None

    # Returned as re-composed parts rather than as the input string, so what is
    # quoted is provably what was validated. `urlsplit` silently drops tab and
    # newline characters before parsing, which would otherwise let an input pass
    # the checks in one spelling and reach the message in another. The whitespace
    # guard above already rejects exactly those inputs, so `return candidate`
    # here is an **equivalent mutant** -- measured, no input distinguishes the
    # two -- and a mutation run will report it forever. It is kept because it
    # makes the property structural instead of a consequence of two checks
    # staying in agreement; do not add a case trying to kill it.
    return urlunsplit(parts)


async def search_apifare(
    query: str,
    *,
    num_results: int,
    http_client: httpx.AsyncClient | None = None,
) -> list[WebSearchResult]:
    """Query apifare's metered search and return parsed organic results.

    apifare endpoint:

    - ``POST https://apifare.com/v1/call/dataforseo``
    - Header: ``Authorization: Bearer <APIFARE_TOKEN>``
    - JSON body: ``{"q": "<query>", "count": <num_results>}``

    The call is debited from the account's prepaid balance at the listed
    per-call price. ``count`` is forwarded as given -- not because apifare
    documents no maximum, as Serply does, but because nothing is known about its
    bound at all; the returned list is capped at ``num_results`` here instead.

    Args:
        query: The search query. A blank query returns no results without a
            request.
        num_results: Maximum number of results to return. A value below 1
            returns no results without a request.
        http_client: Client to send the request with. A short-lived client with a
            30-second timeout is created when omitted.

    Returns:
        At most ``num_results`` results, in the order apifare ranked them, each
        with an empty ``page_content``.

    Raises:
        ApifareConfigError: If ``APIFARE_TOKEN`` is not usable.
        ApifarePaymentRequiredError: If the prepaid balance is exhausted. Raised
            in place of the HTTP 402 so the operator gets the top-up link rather
            than a status code, subject to :func:`_safe_topup_url`.
        ApifareError: If the response is not a JSON object, carries no
            ``result`` object, carries a ``result.results`` that is not a list,
            or holds results none of which could be parsed. An *absent*
            ``results`` key is not an error -- it returns no results.
        httpx.HTTPError: If the request fails or apifare answers with any other
            error status. The router converts it so the request URL is not
            quoted.
    """
    if not query.strip():
        return []

    if num_results < 1:
        return []

    token = _get_apifare_token()
    payload = {"q": query, "count": int(num_results)}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def _do_request(client: httpx.AsyncClient) -> dict[str, Any]:
        """Send the search request and decode its JSON object.

        Args:
            client: The client to send the request with.

        Returns:
            The decoded response body.

        Raises:
            ApifarePaymentRequiredError: If apifare reports an exhausted balance.
            ApifareError: If the body is not valid JSON or not a JSON object.
            httpx.HTTPError: If the request fails or apifare answers with another
                error status.
        """
        resp = await client.post(SEARCH_ENDPOINT, headers=headers, json=payload)
        # Handled before `raise_for_status` because an exhausted balance is a
        # condition the operator can clear, not a transport fault: the router
        # would otherwise flatten it to "HTTP 402" and drop the link.
        if resp.status_code == 402:
            try:
                body = resp.json()
            except ValueError:
                body = {}
            topup = (
                _safe_topup_url(body.get("topup_url"), token)
                if isinstance(body, dict)
                else None
            )
            # The status is named the way `_without_request_url` names one, so a
            # 402 from a proxy or a CDN that never reached apifare does not read
            # as a definitive statement about the account's balance.
            detail = (
                "The apifare search provider failed: HTTP 402, which apifare "
                "uses for an exhausted prepaid balance."
            )
            if topup is None:
                raise ApifarePaymentRequiredError(
                    f"{detail} Top up the account at {TOPUP_URL_HOST} and retry."
                )
            raise ApifarePaymentRequiredError(f"{detail} Top up and retry: {topup}")
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:
            raise ApifareError("apifare response was not valid JSON.") from exc
        if not isinstance(data, dict):
            raise ApifareError("apifare response was not a JSON object.")
        return data

    if http_client is None:
        async with httpx.AsyncClient(timeout=30) as client:
            data = await _do_request(client)
    else:
        data = await _do_request(http_client)

    # A reshaped container raises rather than returning `[]`, so a schema change
    # cannot masquerade as a query with no hits. apifare's wire format is the one
    # part of this provider no live run has confirmed, which is exactly why this
    # branch must be loud. See `.system_design/TEST_SUITE.md` section 14.
    result = data.get("result")
    if not isinstance(result, dict):
        raise ApifareError("apifare response missing `result` object.")

    # An *absent* `results` key returns `[]` rather than raising. The repository's
    # provider rule is explicit that "empty results are a valid answer, not an
    # error", and for an API whose envelope nobody has confirmed, omitting the key
    # is the likeliest spelling of "nothing matched". A key that is present and
    # holds the wrong type is a different claim -- that is a reshaped envelope,
    # and it raises.
    if "results" not in result:
        return []

    raw = result.get("results")
    if not isinstance(raw, list):
        raise ApifareError("apifare response `result.results` is not a list.")

    results: list[WebSearchResult] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        link = item.get("url")
        if not isinstance(title, str) or not isinstance(link, str):
            continue

        # `description` is the SERP snippet. A result without one is still a
        # usable link, and requiring it would let a snippet-less response trip
        # the "none could be parsed" guard below as if the schema had changed.
        description = item.get("description")
        snippet = description if isinstance(description, str) else ""

        # `page_content` is populated later by the MCP tool (best-effort).
        results.append(WebSearchResult(title=title, link=link, snippet=snippet, page_content=""))
        if len(results) >= num_results:
            break

    # Discarding every result means the response did not match the shape expected
    # here. Returning an empty list would be indistinguishable from "no matches"
    # and would hide the mismatch, so surface it instead.
    if raw and not results:
        raise ApifareError(
            f"apifare returned {len(raw)} result(s) but none could be parsed; "
            "each needs a string `title` and `url`. The response schema may have changed."
        )

    return results
