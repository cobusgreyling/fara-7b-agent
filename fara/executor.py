"""Playwright-backed executor for Fara-7B's action vocabulary.

Each Fara action maps to a Playwright operation. After executing,
the browser is given a short settle period and a fresh screenshot is
captured for the next turn.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

SETTLE_MS = 1500


COMMON_SEARCH_SELECTORS = [
    "input[type='search']",
    "input[name='q']",
    "input[name='search']",
    "input[aria-label*='earch']",
    "textarea[name='q']",
]


class BrowserExecutor:
    def __init__(self, headless: bool = False, viewport: tuple[int, int] = (1280, 800)):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]}
        )
        self.page: Page = self._context.new_page()

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._pw.stop()

    def screenshot(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=False)
        return path

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

        self.page.wait_for_timeout(SETTLE_MS)
        return result or "ok"

    # ---- individual action handlers ------------------------------------

    def _act_visit_url(self, url: str, **_: Any) -> str:
        self.page.goto(url, wait_until="domcontentloaded")
        return f"navigated to {url}"

    def _act_web_search(self, query: str, **_: Any) -> str:
        self.page.goto(
            f"https://www.google.com/search?q={query}", wait_until="domcontentloaded"
        )
        return f"searched: {query}"

    def _act_left_click(
        self,
        coordinate: list[int] | tuple[int, int] | None = None,
        x: int | None = None,
        y: int | None = None,
        **_: Any,
    ) -> str:
        if coordinate is not None:
            x, y = coordinate[0], coordinate[1]
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
        if coordinate is not None:
            x, y = coordinate[0], coordinate[1]
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
            self.page.mouse.click(coordinate[0], coordinate[1])
        elif input_target is not None:
            target = self._find_input(input_target)
            if target is None:
                return f"could not locate input '{input_target}'"
            target.click()
        # else: type at currently focused element

        if delete_existing_text:
            self.page.keyboard.press("Meta+A")
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
            self.page.mouse.move(coordinate[0], coordinate[1])
        self.page.mouse.wheel(0, dy)
        return f"scrolled {direction} {amount}"

    def _act_history_back(self, **_: Any) -> str:
        self.page.go_back()
        return "went back"

    def _act_wait(self, seconds: float = 1.0, **_: Any) -> str:
        time.sleep(seconds)
        return f"waited {seconds}s"

    def _act_pause_and_memorize_fact(self, fact: str = "", **_: Any) -> str:
        return f"memorised: {fact}"

    def _act_terminate(self, **_: Any) -> str:
        return "terminate"

    # ---- helpers -------------------------------------------------------

    def _find_input(self, hint: str):
        for sel in COMMON_SEARCH_SELECTORS:
            loc = self.page.locator(sel).first
            try:
                if loc.is_visible(timeout=500):
                    return loc
            except Exception:
                continue
        try:
            loc = self.page.get_by_placeholder(hint).first
            if loc.is_visible(timeout=500):
                return loc
        except Exception:
            pass
        return None
