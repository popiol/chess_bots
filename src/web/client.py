from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from src.web.config import BrowserConfig
from src.web.selectors import SiteSelectors


@dataclass
class SessionState:
    signed_in: bool = False
    guest_mode: bool = False


class ChessWebClient:
    def __init__(
        self,
        config: BrowserConfig,
        selectors: SiteSelectors,
        playwright=None,
        browser=None,
    ) -> None:
        self._config = config
        self._selectors = selectors
        self._playwright = playwright
        self._browser = browser
        self._context = None
        self._page = None
        self._owns_playwright = playwright is None
        self._owns_browser = browser is None
        self.state = SessionState()

    def start(self) -> None:
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._owns_playwright = True
        if self._browser is None:
            self._browser = self._playwright.chromium.launch(
                headless=self._config.headless, slow_mo=self._config.slow_mo_ms
            )
            self._owns_browser = True
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._page.set_default_navigation_timeout(self._config.navigation_timeout_ms)
        self._page.set_default_timeout(self._config.action_timeout_ms)
        self._page.goto(self._config.base_url)

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._browser and self._owns_browser:
            self._browser.close()
        if self._playwright and self._owns_playwright:
            self._playwright.stop()
        self._page = None

    def sign_up(self, email: str, username: str, password: str) -> None:
        self._click(self._selectors.auth.open_signup)
        self._fill(self._selectors.auth.email, email)
        self._fill(self._selectors.auth.username, username)
        self._fill(self._selectors.auth.password, password)
        self._fill(self._selectors.auth.confirm_password, password)
        self._click(self._selectors.auth.submit_signup)
        self._wait_visible(self._selectors.post_login_ready)
        self.state.signed_in = True
        self.state.guest_mode = False

    def sign_in(self, username: str, password: str) -> None:
        self._click(self._selectors.auth.open_signin)
        self._fill(self._selectors.auth.username, username)
        self._fill(self._selectors.auth.password, password)
        self._click(self._selectors.auth.submit_signin)
        self._wait_visible(self._selectors.post_login_ready)
        self.state.signed_in = True
        self.state.guest_mode = False

    def sign_out(self) -> None:
        self._click(self._selectors.auth.signout)
        self.state.signed_in = False

    def play_as_guest(self) -> None:
        self._click(self._selectors.game.play_as_guest)
        self.state.signed_in = False
        self.state.guest_mode = True

    def play_now(self) -> None:
        self._click(self._selectors.game.play_now)
        self._wait_visible(self._selectors.play_ready)

    def is_play_ready(self) -> bool:
        return self._is_visible(self._selectors.play_ready)

    def is_postgame_visible(self) -> bool:
        return self._is_visible(self._selectors.game_page.postgame_panel)

    def is_accept_draw_visible(self) -> bool:
        return self._is_visible(self._selectors.game_page.accept_draw)

    def select_time_control(self, index: int) -> None:
        self._click(self._selectors.time_control_option(index))

    def offer_draw(self) -> None:
        self._click(self._selectors.game_page.offer_draw)

    def resign(self) -> None:
        self._click(self._selectors.game_page.resign)

    def resign_confirm(self) -> None:
        self._click(self._selectors.game_page.resign_confirm)

    def accept_draw(self) -> None:
        self._click(self._selectors.game_page.accept_draw)

    def time_control_indices(self) -> list[int]:
        count = self.page.locator("[data-testid^='time-control-']").count()
        return list(range(count))

    def is_sign_in_available(self) -> bool:
        return self._is_visible(self._selectors.auth.open_signin)

    def is_sign_out_available(self) -> bool:
        return self._is_visible(self._selectors.auth.signout)

    def _click(self, selector: str) -> None:
        self._locator(selector).first.click()

    def _fill(self, selector: str, value: str) -> None:
        self._locator(selector).first.fill(value)

    def _wait_visible(self, selector: str, timeout_ms: int | None = None) -> None:
        try:
            self._locator(selector).first.wait_for(state="visible", timeout=timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Timed out waiting for {selector}") from exc

    def _is_visible(self, selector: str) -> bool:
        try:
            return self._locator(selector).first.is_visible()
        except PlaywrightTimeoutError:
            return False

    def _locator(self, selector: str):
        if selector.startswith("label="):
            label = selector.split("label=", 1)[1].strip()
            return self.page.get_by_label(label)
        return self.page.locator(selector)

    @property
    def page(self):
        if self._page is None:
            raise RuntimeError("Browser session not started. Call start() first.")
        return self._page
