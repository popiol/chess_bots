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

    def queue_play_now(self) -> None:
        self._click(self._selectors.game.play_now)

    def play_now(self) -> None:
        self._click(self._selectors.game.play_now)
        self._wait_visible(self._selectors.play_ready)

    def is_play_ready(self) -> bool:
        return self._is_visible(self._selectors.play_ready)

    def is_postgame_visible(self) -> bool:
        return self._is_visible(self._selectors.game_page.postgame_panel)

    def get_game_result(self) -> str | None:
        """Get the game result text from postgame panel.

        Returns:
            One of: "White wins", "Black wins", "Draw", or None if not available
        """
        try:
            panel = self._locator(self._selectors.game_page.postgame_panel)
            if not panel.first.is_visible():
                return None
            text = panel.first.text_content()
            if text:
                if "White wins" in text:
                    return "White wins"
                elif "Black wins" in text:
                    return "Black wins"
                elif "Draw" in text:
                    return "Draw"
            return None
        except Exception:
            return None

    def get_game_reason(self) -> str | None:
        """Get the game termination reason from postgame panel.

        Returns:
            Lowercase reason with underscores: "checkmate", "timeout", "resignation",
            "stalemate", "insufficient_material", "threefold_repetition",
            "fifty_move_rule", "agreement", or None if not available
        """
        try:
            panel = self._locator(self._selectors.game_page.postgame_panel)
            if not panel.first.is_visible():
                return None
            text = panel.first.text_content()
            if text:
                if "Checkmate" in text:
                    return "checkmate"
                elif "Timeout" in text:
                    return "timeout"
                elif "Resignation" in text:
                    return "resignation"
                elif "Stalemate" in text:
                    return "stalemate"
                elif "Insufficient material" in text:
                    return "insufficient_material"
                elif "Threefold repetition" in text:
                    return "threefold_repetition"
                elif "Fifty move rule" in text:
                    return "fifty_move_rule"
                elif "Agreement" in text:
                    return "agreement"
            return None
        except Exception:
            return None

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

    def make_move(self, from_square: str, to_square: str) -> None:
        """Make a chess move by clicking from_square then to_square.

        Args:
            from_square: Source square in algebraic notation (e.g. 'e2')
            to_square: Destination square in algebraic notation (e.g. 'e4')
        """
        self._click(self._selectors.square(from_square))
        self._click(self._selectors.square(to_square))

    def click_square(self, square: str) -> None:
        """Click a specific square on the chess board.

        Args:
            square: Square in algebraic notation (e.g. 'e4', 'a1')
        """
        self._click(self._selectors.square(square))

    def get_current_fen(self) -> str:
        """Get the current FEN (Forsyth-Edwards Notation) of the game position.

        Returns:
            The FEN string representing the current board state.
        """
        element = self._locator(self._selectors.game_page.game_fen).first
        return element.inner_text()

    def get_time_remaining(self) -> int | None:
        """Get the time remaining on the active player's clock in seconds.

        Returns:
            Time remaining in seconds or None if not available
        """
        try:
            clock_element = self._locator(self._selectors.game_page.active_clock).first
            if clock_element.is_visible():
                time_text = clock_element.text_content()
                if time_text and ":" in time_text:
                    parts = time_text.strip().split(":")
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        return minutes * 60 + seconds
            return None
        except Exception:
            return None

    def is_current_user_turn(self) -> bool:
        """Check if it's the current user's turn to move.

        In live games, the bottom PlayerInfo is the current user.
        This checks if the active player indicator is on the bottom panel.

        Returns:
            True if it's the current user's turn, False otherwise.
        """
        try:
            # Get all player info panels
            all_panels = self.page.locator("css=.player-info").all()
            if len(all_panels) < 2:
                return False

            # Bottom panel is the current user (last in DOM order)
            bottom_panel = all_panels[-1]

            # Check if the bottom panel has the active class
            class_attr = bottom_panel.get_attribute("class") or ""
            return "player-info--active" in class_attr
        except Exception:
            return False

    def get_player_color(self) -> str | None:
        """Determine if the current user is playing white or black.

        Checks the first square on the board. If it's a8, user plays white.
        If it's h1, user plays black.

        Returns:
            'white' if playing as white, 'black' if playing as black, None if cannot determine.
        """
        try:
            # Get the first square element in the DOM
            first_square = self.page.locator("css=[data-square]").first
            square_coord = first_square.get_attribute("data-square")

            if square_coord == "a8":
                return "white"
            elif square_coord == "h1":
                return "black"
            return None
        except Exception:
            return None

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
        self._locator(selector).first.wait_for(state="visible", timeout=timeout_ms)

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
