"""Playwright-backed executor for Fara-7B's action vocabulary.

Each Fara action maps to a Playwright operation. The browser is locked to
`device_scale_factor=1` so screenshot pixels align 1:1 with the viewport —
model-emitted coordinates are in screenshot pixel space.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from playwright.sync_api import Page, sync_playwright

DEFAULT_VIEWPORT = (1280, 800)
DEFAULT_SETTLE_MS = 1500

# URL patterns that mark a likely Critical Point. These are checked against
# `page.url` at the moment of an action; substring match is intentional —
# false positives are preferred to false negatives at a checkout boundary.
CRITICAL_URL_PATTERNS = (
    "/checkout",
    "/payment",
    "/pay/",
    "/order/place",
    "/cart/confirm",
    "/signup",
    "/sign-up",
    "/register",
)


class BrowserExecutor:
    def __init__(
        self,
        headless: bool = False,
        viewport: tuple[int, int] = DEFAULT_VIEWPORT,
        settle_ms: int = DEFAULT_SETTLE_MS,
    ):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            device_scale_factor=1,
        )
        self.page: Page = self._context.new_page()
        self.viewport = viewport
        self.settle_ms = settle_ms

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._pw.stop()

    def screenshot(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=False)
        return path

    def settle(self) -> None:
        """Wait for the page to settle. `domcontentloaded` is fast and
        reliable; `networkidle` is best-effort and allowed to time out on
        sites with persistent connections (analytics, chat widgets)."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=self.settle_ms)
        except Exception:
            self.page.wait_for_timeout(self.settle_ms)

    def at_critical_url(self) -> bool:
        url = (self.page.url or "").lower()
        return any(p in url for p in CRITICAL_URL_PATTERNS)

    def execute(self, action_name: str, arguments: dict[str, Any]) -> str:
        """Execute a Fara action. Returns a short status string."""
        try:
            method = getattr(self, f"_act_{action_name}")
        except AttributeError:
            return f"unsupported action: {action_name}"

        try:
            result = method(**arguments)
        except TypeError as e:
            return f"bad arguments for {action_name}: {e}"
        except Exception as e:
            return f"error executing {action_name}: {e}"

        self.settle()
        return result or "ok"

    # ---- individual action handlers ------------------------------------

    def _act_visit_url(self, url: str, **_: Any) -> str:
        self.page.goto(url, wait_until="domcontentloaded")
        return f"navigated to {url}"

    def _act_web_search(self, query: str, **_: Any) -> str:
        self.page.goto(
            f"https://www.google.com/search?q={quote_plus(query)}",
            wait_until="domcontentloaded",
        )
        return f"searched: {query}"

    def _act_left_click(
        self,
        coordinate: list[int] | tuple[int, int] | None = None,
        x: int | None = None,
        y: int | None = None,
        **_: Any,
    ) -> str:
        x, y = self._resolve_xy(coordinate, x, y)
        if x is None or y is None:
            return "click missing coordinate"
        self.page.mouse.click(x, y)
        return f"clicked ({x}, {y})"

    def _act_mouse_move(
        self,
        coordinate: list[int] | tuple[int, int] | None = None,
        x: int | None = None,
        y: int | None = None,
        **_: Any,
    ) -> str:
        x, y = self._resolve_xy(coordinate, x, y)
        if x is None or y is None:
            return "move missing coordinate"
        self.page.mouse.move(x, y)
        return f"moved to ({x}, {y})"

    def _act_type(
        self,
        text: str,
        coordinate: list[int] | tuple[int, int] | None = None,
        input_target: str | None = None,
        press_enter: bool = False,
        delete_existing_text: bool = False,
        **_: Any,
    ) -> str:
        if coordinate is not None:
            x, y = self._resolve_xy(coordinate, None, None)
            if x is not None and y is not None:
                self.page.mouse.click(x, y)
        elif input_target is not None:
            target = self._find_input(input_target)
            if target is None:
                return f"could not locate input '{input_target}'"
            target.click()
        # else: type at currently focused element

        if delete_existing_text:
            self.page.keyboard.press("ControlOrMeta+A")
            self.page.keyboard.press("Delete")

        self.page.keyboard.type(text)
        if press_enter:
            self.page.keyboard.press("Enter")
        return f"typed: {text!r}" + (" + Enter" if press_enter else "")

    def _act_key(self, key: str, **_: Any) -> str:
        self.page.keyboard.press(key)
        return f"pressed {key}"

    def _act_scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        coordinate: list[int] | tuple[int, int] | None = None,
        **_: Any,
    ) -> str:
        dy = amount if direction == "down" else -amount
        if coordinate is not None:
            x, y = self._resolve_xy(coordinate, None, None)
            if x is not None and y is not None:
                self.page.mouse.move(x, y)
        self.page.mouse.wheel(0, dy)
        return f"scrolled {direction} {amount}"

    def _act_history_back(self, **_: Any) -> str:
        self.page.go_back()
        return "went back"

    def _act_wait(self, seconds: float = 1.0, **_: Any) -> str:
        time.sleep(seconds)
        return f"waited {seconds}s"

    def _act_pause_and_memorize_fact(self, fact: str = "", **_: Any) -> str:
        # The fact is surfaced in the status string and harvested by the agent
        # loop so it can be re-injected into the next turn's user text.
        return f"memorised: {fact}"

    def _act_terminate(self, **_: Any) -> str:
        return "terminate"

    # ---- helpers -------------------------------------------------------

    def _resolve_xy(
        self,
        coordinate: list[int] | tuple[int, int] | None,
        x: int | None,
        y: int | None,
    ) -> tuple[int | None, int | None]:
        if coordinate is not None and len(coordinate) >= 2:
            x, y = coordinate[0], coordinate[1]
        if x is None or y is None:
            return None, None
        # Clamp to viewport. Quantised models occasionally emit out-of-range
        # coords; clamping prevents the click from being silently dropped.
        w, h = self.viewport
        cx = max(0, min(int(x), w - 1))
        cy = max(0, min(int(y), h - 1))
        return cx, cy

    def _find_input(self, hint: str):
        """Locate an input matching `hint`. Hint-based locators are tried
        first (label, placeholder, accessible name); generic search-input
        selectors are the fallback."""
        if hint:
            candidates = [
                lambda: self.page.get_by_label(hint, exact=False).first,
                lambda: self.page.get_by_placeholder(hint).first,
                lambda: self.page.get_by_role("textbox", name=hint).first,
                lambda: self.page.get_by_role("searchbox", name=hint).first,
            ]
            for build in candidates:
                try:
                    loc = build()
                    if loc.is_visible(timeout=500):
                        return loc
                except Exception:
                    continue

        for sel in (
            "input[type='search']",
            "input[name='q']",
            "input[name='search']",
            "input[aria-label*='earch']",
            "textarea[name='q']",
        ):
            try:
                loc = self.page.locator(sel).first
                if loc.is_visible(timeout=500):
                    return loc
            except Exception:
                continue
        return None
