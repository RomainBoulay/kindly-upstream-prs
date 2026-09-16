"""Cover the pooled-target branch of the nodriver worker's reuse path.

When ``KINDLY_NODRIVER_REUSE_BROWSER=1`` the worker does not launch a browser;
it connects to one the pool already runs and navigates a tab inside it.
`_ensure_reuse_page`, nested in `_fetch_html`, is the whole of that decision: it
reuses the pooled browser's first page target if there is one, and otherwise asks
Chromium to create one. This module owns the *create* half.

**That half had never run to completion.** It sent its
``Target.createTarget`` through ``browser.connection.send(...)``, and
`nodriver.Browser` has no live ``connection``: ``Browser.__init__`` assigns
``self.connection: Connection = None`` and nothing in nodriver 0.50.3 ever
reassigns it -- measured, the only ``.connection =`` in the installed package is
that line. Every call raised ``AttributeError: 'NoneType' object has no
attribute 'send'``, which the branch's own ``except Exception`` turned into
``RuntimeError("Failed to create pooled target; ...")``, which
`universal_html.py`'s `_pool_error_requires_restart` matches -- so the parent
terminated the slot, launched a fresh browser and succeeded on the retry. The
defect was invisible in the result and visible only as a browser launch per
request, which is the cost pooling exists to avoid (issue #96).

The branch is reachable on every request after the first, because
`_fetch_html`'s `_cleanup` closes the tab it navigated and that tab is the
pooled browser's only page target. A ``Target.createTarget`` with
``new_window=False`` then has no window to put a tab in and is refused with
``Failed to open new tab - no browser is open``, which the same restart matcher
also matches. So the branch carried two independent defects and this module pins
both separately: repairing one leaves the other's case red.

**Three of the facts above are Chromium's, not this code's, and no doubled
browser can assert them.** They were measured, not read: closing the last page
target leaves the process alive and its DevTools endpoint answering, so the
pool's health probe still passes and the slot is handed back out with zero tabs
*and zero windows*; ``newWindow=false`` is then refused with exactly that string;
``newWindow=true`` is accepted and leaves no window behind across repeated
create-and-close cycles. Measured on Chrome 153 under ``--headless=new`` on
Windows (2026-09-16), and reported on snap Chromium 152.0.7977.64 on Linux in
issue #96. A ``chromium``-lane case would hold them; that lane is registered in
``pyproject.toml`` and run nowhere today, so `TEST_SUITE.md` §9 carries the gap.

**One thing this module deliberately does not cover.** Every case passes
``referer=None``: in reuse mode the referer and the URL share one tab, so
`_cleanup` closes the same object twice and the second close is swallowed by its
blanket ``except Exception``. That is pre-existing and unrelated to target
creation.

An earlier draft of this docstring named a second gap -- that sending
``new_window`` unconditionally required a full Chrome/Chromium, because the
protocol reference marks the parameter unsupported by ``chrome-headless-shell``.
**Measured false, and removed rather than left as a caveat**: the shell *accepts*
``newWindow=true`` and hands back a tab, because "unsupported" there means the
windowing semantics are not honoured, not that the call errors. ``SYSTEM_DESIGN.md``
§1.3 carries the table.

**Every double is autospecced from the real class**, following
:mod:`tests.test_nodriver_worker_sandbox` and for the reason that module records:
a hand-written stand-in accepts any call at all, which is how a signature change
once disabled eight of its tests without one of them objecting.

Two fidelity details, both measured against nodriver 0.50.3, that a hand-written
double would get wrong and an autospec cannot supply on its own:

* ``Tab`` resolves ``type_`` and ``target_id`` through ``__getattr__`` onto its
  ``target``, so neither is on ``create_autospec(nodriver.Tab)`` and both are set
  explicitly here. Production filters targets with ``getattr(target, "type_",
  None)``, which would silently *drop* a double missing them -- the test would
  then pass through the wrong branch.
* ``Browser.connection`` is an instance attribute, so it is absent from the
  autospec too. It is set to ``None`` here, which is what production reads off
  the real object, so a reader of ``browser.connection`` fails with the
  production error rather than a mock artefact. One case swaps that for a
  property that raises on access, and says why.

Nothing here starts a browser, binds a socket or spawns a process:
:func:`nodriver.start` is doubled, and it is the only thing the reuse path would
otherwise use to reach the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import NonCallableMagicMock, create_autospec, patch

import nodriver
import pytest

from kindly_web_search_mcp_server.scrape import nodriver_worker

#: Every environment variable the *pooled* branch of `_fetch_html` reads, traced
#: from that branch: ``KINDLY_NODRIVER_SANDBOX`` via `_resolve_sandbox_enabled`,
#: the four browser paths via `_resolve_browser_executable_path`, and the
#: DevTools budget via `_resolve_devtools_ready_timeout_seconds`.
#:
#: Deliberately **not** imported from :mod:`tests.test_nodriver_worker_sandbox`,
#: whose list is a documented superset chosen for the *unpooled* branch it
#: covers. Sharing it would make this module's controlled inputs change whenever
#: that one's reasoning did. The repository already states this rule for doubles
#: (`make_pinned_detector`); it holds for input pins for the same reason.
#:
#: Four of the seven are unreachable while a case passes an explicit executable,
#: and they are cleared anyway: a case that leaves them alone is a case whose
#: result depends on what the developer exported, and it would start depending on
#: it the day the resolver consults the environment first.
READ_ENVIRONMENT_VARIABLES = (
    "KINDLY_NODRIVER_SANDBOX",
    "KINDLY_NODRIVER_DEVTOOLS_READY_TIMEOUT_SECONDS",
    "KINDLY_BROWSER_EXECUTABLE_PATH",
    "BROWSER_EXECUTABLE_PATH",
    "CHROME_BIN",
    "CHROME_PATH",
)

#: Handed to `_fetch_html` directly so the executable resolver short-circuits on
#: its first branch and no ``PATH`` probe happens. The pooled branch never
#: launches this file, so it does not have to exist.
BROWSER_PATH = "/usr/bin/chromium-for-tests"

#: Where the pooled browser's DevTools endpoint is, as the parent would pass it.
#: `_fetch_html` refuses ``reuse_browser`` without both.
POOL_HOST = "127.0.0.1"
POOL_PORT = 39222

#: The target id Chromium is pretended to return from ``Target.createTarget``.
#: Distinctive so the "created target never appeared" case can assert that the
#: failure message names the id it waited for.
CREATED_TARGET_ID = "created-target-8fa1"

#: The target id of a tab a case pre-seeds into the pooled browser, for the
#: reuse-what-is-there half of the branch.
EXISTING_TARGET_ID = "existing-target-01c3"

#: A ceiling on any single `_fetch_html` call. `_ensure_reuse_page` retries
#: target discovery three times with a doubling sleep, so a regression that stops
#: terminating that loop must fail this module rather than hang the suite. Five
#: seconds is an order of magnitude above the ~0.4 s the slowest case here takes.
PER_TEST_TIMEOUT_SECONDS = 5.0


class CreateTargetCommand:
    """A stand-in for the CDP command object ``create_target`` returns.

    The real `nodriver.cdp.target.create_target` is a *generator* function: it
    returns a generator that a live ``send`` drives to produce the wire request
    and consume the reply. A doubled ``send`` never drives it, so the generator
    would carry no observable information at all -- which is exactly why the
    command is doubled and this identity object put in its place. Asserting that
    ``send`` received *this object* is what ties the two calls together, and the
    doubled ``create_target`` records the keyword arguments separately.
    """

    def __repr__(self) -> str:
        """Return a form that reads as itself in an assertion message."""
        return "<CreateTargetCommand>"


def make_page_target(target_id: str, html: str) -> Any:
    """Build a pooled page-target double.

    Args:
        target_id: The id the worker matches when it looks for a created target.
        html: The document the tab returns from ``get_content``.

    Returns:
        An autospec `nodriver.Tab` carrying what the worker reads off it.
    """
    page = create_autospec(nodriver.Tab, instance=True)
    # Resolved through `Tab.__getattr__` on the real object, so absent from the
    # autospec; production reads both with `getattr`, which would quietly skip a
    # target that lacked them.
    page.type_ = "page"
    page.target_id = target_id
    # `_navigate_tab` reads the frame id out of the navigate reply and stores it.
    page.send.return_value = ("frame-1", "loader-1")
    page.get_content.return_value = html
    return page


class PooledBrowser:
    """A connected pooled-browser double and the targets it hands out.

    Models the one behaviour the branch depends on: ``update_targets`` refreshes
    ``targets`` from the browser, so a target created between two refreshes is
    absent from the first and present in the second. A static list cannot
    represent that, and the branch's discovery retry is written for it.

    Attributes:
        browser: The autospec `nodriver.Browser` handed to the worker.
        create_target: The doubled `nodriver.cdp.target.create_target`, whose
            recorded keyword arguments are what a case asserts on.
        created: Every page target creation handed out, in order.
        sent: Every object the worker put through ``browser.send``, in order.
    """

    def __init__(
        self,
        *,
        existing: list[Any],
        html: str,
        discoverable: bool,
        trap_connection: bool,
    ) -> None:
        """Wire the double to serve one `_fetch_html` call.

        Args:
            existing: Page targets the pooled browser already has.
            html: The document a created tab returns from ``get_content``.
            discoverable: Whether a created target appears in ``targets`` on the
                next refresh. ``False`` models the target Chromium accepted but
                never surfaced, which the branch must report rather than hang on.
            trap_connection: Make *reading* ``browser.connection`` raise, instead
                of yielding the ``None`` the real object carries.
        """
        self._existing = existing
        self._html = html
        self._discoverable = discoverable
        self.created: list[Any] = []
        self.sent: list[Any] = []
        self.create_target = create_autospec(
            nodriver.cdp.target.create_target,
            return_value=CreateTargetCommand(),
        )
        self.browser = create_autospec(nodriver.Browser, instance=True)
        self._install_connection(trap_connection)
        self.browser.targets = list(existing)
        self.browser.send.side_effect = self._send
        self.browser.update_targets.side_effect = self._update_targets

    def _install_connection(self, trap: bool) -> None:
        """Give the double a ``connection`` attribute, absent from the autospec.

        ``Browser.connection`` is assigned in ``__init__``, and autospec copies a
        class's methods and descriptors rather than the attributes ``__init__``
        sets, so the double has none. Without one, a reader fails on a mock
        artefact rather than on what production would hit.

        The default is faithful: ``None``, which is what the real object carries
        and what makes the shipped defect fail with its own
        ``AttributeError: 'NoneType' object has no attribute 'send'``. The trap is
        strictly stronger and deliberately *un*-faithful -- a read of ``None``
        succeeds and only *using* it fails, so a case asserting that the attribute
        is never read at all cannot be written against the faithful form.
        **Patching the type reaches this double only**, which is worth stating
        because the code reads like the opposite: ``create_autospec``
        synthesises a fresh subclass *also named* ``NonCallableMagicMock`` per
        mock, so ``type(self.browser)`` is private to this instance despite a
        ``__name__`` that suggests the shared class. Nothing is left behind and
        no teardown is owed.

        Asserted rather than only measured, by
        :func:`test_the_connection_trap_does_not_leak_to_other_doubles`: two
        review rounds read this line as session-wide mock pollution, so the
        isolation is now a case that would fail if it ever stopped holding.

        Args:
            trap: Raise on read instead of answering ``None``.
        """
        if not trap:
            self.browser.connection = None
            return

        def _refuse(_self: object) -> None:
            raise AssertionError(
                "the pooled branch read `browser.connection`; nodriver leaves it "
                "None, so the browser session is reached with `browser.send`"
            )

        type(self.browser).connection = property(_refuse)

    async def _send(self, command: object) -> str:
        """Serve the browser-session ``send`` the branch is expected to use.

        Records rather than asserts: this runs inside production's ``try``, which
        converts any exception into ``Failed to create pooled target`` and would
        bury a wrong-command diagnosis in ``__cause__``. The recording is asserted
        after the harness exits, where the failure reads plainly.

        Args:
            command: The CDP command object the worker sent.

        Returns:
            The id of the target this call created.
        """
        self.sent.append(command)
        self.created.append(make_page_target(CREATED_TARGET_ID, self._html))
        return CREATED_TARGET_ID

    async def _update_targets(self) -> None:
        """Refresh ``targets`` the way a real ``update_targets`` would."""
        discovered = self.created if self._discoverable else []
        self.browser.targets = [*self._existing, *discovered]

    def create_target_keywords(self) -> dict[str, Any]:
        """Return the keyword arguments the branch asked ``create_target`` for.

        Returns:
            The sole call's keyword arguments.

        Raises:
            AssertionError: If the branch made no call, or more than one. Either
                means the case is asserting about a call that did not happen the
                way it reads.
        """
        calls = self.create_target.call_args_list
        assert len(calls) == 1, f"expected one create_target call, got {len(calls)}"
        return dict(calls[0].kwargs)


@contextlib.contextmanager
def pooled_harness(
    *,
    existing: list[Any] | None = None,
    html: str = "<html><body>pooled</body></html>",
    discoverable: bool = True,
    send_error: Exception | None = None,
    trap_connection: bool = False,
) -> Iterator[PooledBrowser]:
    """Install the doubles one pooled `_fetch_html` call needs.

    Doubles exactly two collaborators: :func:`nodriver.start`, the only call the
    pooled branch makes that would reach the machine, and
    `nodriver.cdp.target.create_target`, so the keyword under assertion is
    recordable. Everything else in `_fetch_html` -- the argument validation, the
    branch itself, the discovery retry, navigation and `_cleanup` -- runs for
    real, which is the point.

    ``os.geteuid`` is pinned because `_resolve_sandbox_enabled` reads it and a
    container running as root would otherwise answer differently from a
    developer's machine. ``shutil.which`` is pinned to "nothing installed" for the
    reason the sandbox module's harness records: a case that leaves the real
    lookup in place is a case whose result depends on whether the developer has
    Chromium. ``_DIAG_ENABLED`` is a module global assigned only by the worker's
    own entry point, so clearing ``KINDLY_DIAGNOSTICS`` does not reach it and it
    is patched directly.

    Args:
        existing: Page targets the pooled browser already has; ``None`` means a
            browser whose last tab has been closed, which is the state after
            `_cleanup` and the state that reaches the create branch.
        html: The document the navigated tab returns.
        discoverable: Whether a created target appears on the next refresh.
        send_error: Raised by ``browser.send`` instead of creating a target, for
            the case that asserts how a refused creation is reported.
        trap_connection: Make reading ``browser.connection`` raise rather than
            answer ``None``. See :meth:`PooledBrowser._install_connection`.

    Yields:
        The :class:`PooledBrowser` that was installed.
    """
    pooled = PooledBrowser(
        existing=list(existing or []),
        html=html,
        discoverable=discoverable,
        trap_connection=trap_connection,
    )
    if send_error is not None:
        pooled.browser.send.side_effect = send_error

    # A cleared slate: every variable the pooled branch reads is removed, so a
    # case declares the whole of its own input.
    saved = {name: os.environ.get(name) for name in READ_ENVIRONMENT_VARIABLES}
    for name in READ_ENVIRONMENT_VARIABLES:
        os.environ.pop(name, None)
    try:
        with (
            patch.object(
                nodriver, "start", autospec=True, return_value=pooled.browser
            ),
            patch.object(
                nodriver.cdp.target, "create_target", pooled.create_target
            ),
            patch.object(
                nodriver_worker.shutil, "which", autospec=True, return_value=None
            ),
            patch.object(
                nodriver_worker.os, "geteuid", return_value=1000, create=True
            ),
            patch.object(nodriver_worker, "_DIAG_ENABLED", False),
        ):
            yield pooled
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def fetch_pooled_html(**overrides: Any) -> str:
    """Call `_fetch_html` with the arguments a *pooled* fetch supplies.

    Written out rather than splatted from a dict, so the call to the function
    under test stays checkable; the sibling helper in
    :mod:`tests.test_nodriver_worker_sandbox` does the same and says why.

    Args:
        **overrides: Replacements for the pooled defaults below.

    Returns:
        The HTML `_fetch_html` produced.

    Raises:
        TimeoutError: If the call does not finish within
            :data:`PER_TEST_TIMEOUT_SECONDS`, which means the branch failed to
            terminate rather than failed an assertion.
    """
    kwargs: dict[str, Any] = {
        "url": "https://example.com/second",
        "referer": None,
        "user_agent": "kindly-test-agent",
        "wait_seconds": 0.0,
        "browser_executable_path": BROWSER_PATH,
        "reuse_browser": True,
        "remote_host": POOL_HOST,
        "remote_port": POOL_PORT,
        "user_data_dir": None,
        "overall_timeout_seconds": 30.0,
    }
    kwargs.update(overrides)
    return await asyncio.wait_for(
        nodriver_worker._fetch_html(
            kwargs["url"],
            referer=kwargs["referer"],
            user_agent=kwargs["user_agent"],
            wait_seconds=kwargs["wait_seconds"],
            browser_executable_path=kwargs["browser_executable_path"],
            reuse_browser=kwargs["reuse_browser"],
            remote_host=kwargs["remote_host"],
            remote_port=kwargs["remote_port"],
            user_data_dir=kwargs["user_data_dir"],
            overall_timeout_seconds=kwargs["overall_timeout_seconds"],
        ),
        timeout=PER_TEST_TIMEOUT_SECONDS,
    )


async def test_an_empty_pooled_browser_gets_its_target_through_the_browser_session() -> None:
    """Create the target with ``browser.send``, the call nodriver itself uses.

    A pooled browser whose last tab was closed has no page target, so the branch
    has to create one. It must send the command through the browser session --
    ``Browser`` *is* a ``Connection`` and ``Browser.get()`` sends its own
    ``create_target`` through ``self.send`` -- and not through
    ``browser.connection``, which is ``None`` on every real instance.

    Against the shipped code this fails with that ``AttributeError`` as the cause
    of ``Failed to create pooled target``: the issue reproduced.
    """
    with pooled_harness(html="<html><body>created</body></html>") as pooled:
        html = await fetch_pooled_html()

    assert html == "<html><body>created</body></html>"
    # Read off the recording rather than off `await_args.args[0]`, which would
    # raise `IndexError` instead of failing cleanly if the command were ever
    # passed by keyword.
    assert [type(command) for command in pooled.sent] == [CreateTargetCommand]


async def test_the_pooled_target_is_created_as_a_new_window() -> None:
    """Ask for a new window, because there is no window to put a tab in.

    The branch is reachable only with no page targets, and Chromium closes the
    window when its last tab closes. ``Target.createTarget`` with
    ``newWindow=false`` then has nowhere to put the tab and is refused with
    ``Failed to open new tab - no browser is open`` -- a string the parent's
    `_pool_error_requires_restart` matches, so the symptom is a pool restart
    rather than an error.

    Asserted on the keyword rather than on an encoded CDP payload, so the
    assertion names the decision it protects. That Chromium *accepts*
    ``newWindow=true`` in a windowless browser is the other half of this claim
    and cannot be asserted with the browser doubled; it is measured rather than
    tested, and the module docstring records the build it was measured on.
    """
    with pooled_harness() as pooled:
        await fetch_pooled_html()

    assert pooled.create_target_keywords().get("new_window") is True


async def test_the_browsers_own_connection_attribute_is_never_read() -> None:
    """Refuse to be *reached* through ``browser.connection`` at all.

    Pinned separately from the first case because the two can come apart: code
    that reads the attribute and then sends through the session -- ``conn =
    browser.connection`` followed by ``await browser.send(...)`` -- satisfies
    that case and is still building on an attribute nodriver leaves ``None``.
    Here the double raises on *access*, so only a branch that never touches it
    passes.

    This is the one case whose double is deliberately unfaithful; the faithful
    ``None`` the others carry cannot fail on a read.
    """
    with pooled_harness(trap_connection=True) as pooled:
        await fetch_pooled_html()

    pooled.browser.send.assert_awaited_once()


async def test_a_pooled_browser_that_already_has_a_tab_creates_nothing() -> None:
    """Reuse the tab that is there, on the session's first request.

    The half of the branch that already worked, and the one a repair could
    plausibly break by always creating a window: a pooled browser handed out with
    a live tab must be navigated, not grown a second one.

    "A live tab" includes one no `_cleanup` ever closed -- a worker the parent
    killed on timeout leaves its tab behind -- so this case is also what pins
    that the branch navigates whatever is there rather than starting clean.
    """
    existing = make_page_target(EXISTING_TARGET_ID, "<html><body>reused</body></html>")
    with pooled_harness(existing=[existing]) as pooled:
        html = await fetch_pooled_html()

    assert html == "<html><body>reused</body></html>"
    pooled.create_target.assert_not_called()
    pooled.browser.send.assert_not_awaited()


async def test_a_target_that_cannot_be_created_is_reported_as_a_pool_failure() -> None:
    """Report a refused creation as a pooled failure, saying so in the message.

    `universal_html.py` decides whether to terminate and re-acquire the slot by
    scanning the message chain, and ``"failed to create pooled target"`` is one
    of the substrings it looks for. The wording is **not** load-bearing for
    recovery, though: the worker is a subprocess, and `_run_worker_command` wraps
    any nonzero exit as ``"nodriver worker failed (exit=N): …"``, which the same
    matcher catches first. What this case protects is the *diagnosis* -- the
    sentence an operator reads in the restart diagnostics -- and the classifying
    of a refused creation as a pooled failure rather than a returned error.

    A **regression guard, not a reproduction**: it passed against the shipped
    code too, because the ``AttributeError`` the defect raised produced the same
    message.
    """
    with (
        pooled_harness(send_error=RuntimeError("target creation refused")),
        pytest.raises(RuntimeError, match="Failed to create pooled target"),
    ):
        await fetch_pooled_html()


def test_the_connection_trap_does_not_leak_to_other_doubles() -> None:
    """Keep the read-trap inside the one double that asked for it

    A harness check rather than a claim about production, and the only one in
    this module. `PooledBrowser._install_connection` assigns a property to
    ``type(self.browser)``, and two review rounds read that as installing onto
    the shared :class:`unittest.mock.NonCallableMagicMock` and leaking to every
    later mock in the pytest session -- which would make the *other* five cases
    pass or fail depending on the order they ran in, the worst failure a suite
    can have.

    It does not leak, because ``create_autospec`` builds a per-mock subclass that
    merely *reuses the name* ``NonCallableMagicMock``. That is genuinely
    surprising, and a comment saying so is only as good as the reader's trust in
    it, so this asserts it instead: the shared class stays clean, an independent
    double still takes an ordinary ``connection``, and the trapped one still
    raises.

    Synchronous, because nothing here runs the worker. One harness at a time,
    too: :func:`pooled_harness` patches :func:`nodriver.start` with an autospec,
    and a second one entered inside the first would try to autospec that patch,
    which `unittest.mock` refuses outright (``InvalidSpecError: Cannot spec a
    Mock object``) -- so a nested-harness version of this case fails on its own
    setup and proves nothing either way.
    """
    with pooled_harness(trap_connection=True) as trapped:
        # The shared class is what a leak would touch, so name it directly
        # rather than inferring from behaviour.
        assert "connection" not in vars(NonCallableMagicMock)

        with pytest.raises(AssertionError, match="read `browser.connection`"):
            _ = trapped.browser.connection

        # An independent double, built while the trap is installed: the case
        # against a leak, at the moment a leak would be live.
        sibling = create_autospec(nodriver.Browser, instance=True)
        sibling.connection = None
        assert sibling.connection is None

    # And one built after the harness exits, since nothing undoes the property.
    later = create_autospec(nodriver.Browser, instance=True)
    later.connection = None
    assert later.connection is None


async def test_a_created_target_that_never_appears_is_reported_with_its_id() -> None:
    """Give up on an accepted-but-invisible target, naming what was awaited.

    Creation returning an id is not discovery: the target has to turn up in
    ``update_targets`` before it can be navigated. The branch retries three times
    and then fails, and the id belongs in that message because it is the only
    thing that distinguishes this from a browser that created nothing.
    """
    with (
        pooled_harness(discoverable=False),
        pytest.raises(RuntimeError, match=CREATED_TARGET_ID),
    ):
        await fetch_pooled_html()
