from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

import requests
import websocket

from src.web.chess_client import ChessClient

logger = logging.getLogger(__name__)


class ChessAPIClient(ChessClient):
    """API-backed chess client using the Main/Identity service REST APIs.

    Notes:
    - Uses requests.Session for cookies and auth header management.
    - For actions that require a live websocket (moves, resign, draw), the client
      will attempt to open a websocket if `websocket-client` is available; otherwise
      those actions will raise RuntimeError.
    """

    def __init__(self, base_url: str = "https://playbullet.gg") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._access_token: str | None = None
        self.signed_in: bool = False
        self.guest_mode: bool = False
        self._match_params: dict[str, Any] = {}
        self._matched_game: dict[str, Any] | None = None
        self._game_id: str | None = None
        self._ws = None
        self._ws_thread: threading.Thread | None = None
        self._ws_running: bool = False
        # Pinger thread to request authoritative state periodically
        self._ws_pinger_thread: threading.Thread | None = None
        self._last_get_state_time: float = 0.0
        self._get_state_interval: float = 10.0
        # Live state received from websocket messages (state_update / game_over)
        self._last_state: dict | None = None
        # Track latest draw offer ("white" or "black") if any
        self._draw_offered_by: str | None = None
        # Resign confirmation pending flag
        self._resign_pending: bool = False

    # --- lifecycle ---
    def start(self) -> None:
        # No persistent connection by default
        return None

    def close(self) -> None:
        # Stop reader thread and close websocket
        self._ws_running = False
        if self._ws is not None:
            try:
                try:
                    self._ws.close()
                except Exception:
                    logger.exception("Failed to close websocket")
            finally:
                self._ws = None
        if self._ws_thread is not None:
            # give the thread a moment to exit
            self._ws_thread.join(timeout=1.0)
            self._ws_thread = None
        if self._ws_pinger_thread is not None:
            # give the pinger a moment to exit
            self._ws_pinger_thread.join(timeout=1.0)
            self._ws_pinger_thread = None

    # --- auth ---
    def sign_up(self, email: str, username: str, password: str) -> None:
        url = f"{self.base_url}/api/identity/auth/register"
        payload = {"username": username, "email": email, "password": password}
        r = self._session.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logger.error("Failed to sign up: %s", r.text)
        r.raise_for_status()

    def sign_in(self, username: str, password: str) -> None:
        url = f"{self.base_url}/api/identity/auth/login"
        # OAuth2 password-style form
        r = self._session.post(
            url, data={"username": username, "password": password}, timeout=10
        )
        if r.status_code != 200:
            logger.error(
                "Failed to sign in: %s, data=%s",
                r.text,
                {"username": username, "password": password},
            )
        r.raise_for_status()
        body = r.json()
        token = body.get("access_token")
        if token:
            self._access_token = token
            self._session.headers.update({"Authorization": f"Bearer {token}"})
            self.signed_in = True
            self.guest_mode = False

    def sign_out(self) -> None:
        if not self._access_token:
            return
        url = f"{self.base_url}/api/identity/auth/logout"
        try:
            r = self._session.post(url, timeout=5)
            r.raise_for_status()
        except Exception:
            logger.exception("Failed to sign out")
        self._access_token = None
        self._session.headers.pop("Authorization", None)
        self.signed_in = False
        self.guest_mode = False

    # --- matchmaking / guest ---
    def play_as_guest(self) -> None:
        # Ensure a guest_id cookie exists
        if "guest_id" not in self._session.cookies:
            gid = str(uuid.uuid4())
            self._session.cookies.set(
                "guest_id",
                gid,
                domain=self.base_url.replace("https://", "").split(":")[0],
            )
        self._access_token = None
        self.signed_in = False
        self.guest_mode = True
        self.queue_play_now(rated=False)

    def queue_play_now(self, rated: bool = True) -> None:
        # Join matchmaking with defaults unless provided
        params = dict(
            time_control_initial=self._match_params.get("time_control_initial", 60),
            time_control_increment=self._match_params.get("time_control_increment", 0),
            mode="rated" if rated else "casual",
            auth_mode=("account" if self.signed_in else "guest"),
        )
        url = f"{self.base_url}/api/matchmaking/join"
        r = self._session.post(url, json=params, timeout=10)
        r.raise_for_status()
        return None

    def is_play_ready(self) -> bool:
        # Poll matchmaking/find to see if matched
        params = dict(
            time_control_initial=self._match_params.get("time_control_initial", 60),
            time_control_increment=self._match_params.get("time_control_increment", 0),
            mode=self._match_params.get("mode", "casual"),
            auth_mode=("account" if self.signed_in else "guest"),
        )
        url = f"{self.base_url}/api/matchmaking/find"
        r = self._session.post(url, json=params, timeout=10)
        r.raise_for_status()
        body = r.json()
        if body.get("matched"):
            self._matched_game = body
            self._game_id = body.get("game_id")
            return True
        return False

    # --- game / state ---
    def is_postgame_visible(self) -> bool:
        info = self._game_state()
        if not info:
            return False
        result = info.get("result")
        return result is not None and result != "*"

    def get_game_result(self) -> str | None:
        info = self._game_state()
        if not info:
            return None
        result = info.get("result")
        if result == "1-0":
            return "White wins"
        if result == "0-1":
            return "Black wins"
        if result == "1/2-1/2":
            return "Draw"
        return None

    def get_game_reason(self) -> str | None:
        info = self._game_state()
        if not info:
            return None
        return info.get("termination_reason")

    def is_accept_draw_visible(self) -> bool:
        # If a draw was offered by the opponent, the accept control is visible
        if not self._draw_offered_by:
            return False
        my_color = self.get_player_color()
        if not my_color:
            return False
        return self._draw_offered_by != my_color

    def select_time_control(self, index: int) -> None:
        index = int(index)
        time_controls = [(60, 0), (120, 1), (180, 0), (180, 2), (600, 0)]
        self._match_params["time_control_initial"] = time_controls[index][0]
        self._match_params["time_control_increment"] = time_controls[index][1]

    def offer_draw(self) -> None:
        self._send_ws({"type": "offer_draw", "data": {}})

    def resign(self) -> None:
        # mark resign as pending (UI should call resign_confirm to actually send)
        self._resign_pending = True

    def resign_confirm(self) -> None:
        # Send the resign message if pending (or send anyway if called directly)
        if not self._resign_pending:
            return
        try:
            self._send_ws({"type": "resign", "data": {}})
        finally:
            self._resign_pending = False

    def accept_draw(self) -> None:
        self._send_ws({"type": "accept_draw", "data": {}})

    def make_move(
        self, from_square: str, to_square: str, promotion: str | None = None
    ) -> None:
        payload = {
            "type": "move",
            "data": {
                "from_square": from_square,
                "to_square": to_square,
                "promotion": promotion,
            },
        }
        self._send_ws(payload)

    def get_current_fen(self) -> str:
        info = self._game_state()
        if not info:
            raise RuntimeError("No game state available")
        # prefer live fen if present
        return info.get("fen") or info.get("final_fen") or ""

    def get_last_move_valid(self) -> tuple[str | None, bool]:
        # Not applicable to API client; return (None, True)
        return (None, True)

    def get_time_remaining(self) -> int | None:
        info = self._game_state()
        if not info:
            return None
        clocks = info.get("clocks") or {}
        my_color = self.get_player_color()
        if my_color == "white":
            secs = clocks.get("white_seconds")
        else:
            secs = clocks.get("black_seconds")
        if secs is None:
            return None
        return int(secs)

    def is_current_user_turn(self) -> bool:
        info = self._game_state()
        if not info:
            return False
        return info.get("current_turn") == (self.get_player_color() or "")

    def get_player_color(self) -> str | None:
        if self._matched_game:
            my_color = self._matched_game.get("my_color")
            if my_color in ("white", "black"):
                return my_color
        return None

    def time_control_indices(self) -> list[int]:
        return list(range(5))

    def is_sign_in_available(self) -> bool:
        return not self.signed_in

    def is_sign_out_available(self) -> bool:
        return self.signed_in

    def is_post_login_ready(self) -> bool:
        return self.signed_in

    # --- helpers ---
    def _game_state(self) -> dict | None:
        if self._last_state:
            return self._last_state
        try:
            self._ensure_ws_connected()
        except Exception:
            logger.info(
                "WebSocket not available when fetching game state; falling back to REST"
            )
        if not self._game_id:
            return None
        url = f"{self.base_url}/api/games/{self._game_id}"
        try:
            r = self._session.get(url, timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            logger.exception("Failed to fetch game state")
            return None

    def _send_ws(self, message: dict) -> None:
        # Ensure websocket connected and reader running
        self._ensure_ws_connected()
        if not self._ws:
            raise RuntimeError("WebSocket unavailable for game actions")
        try:
            self._ws.send(json.dumps(message))
        except Exception:
            logger.exception("Failed to send websocket message")

    def _ensure_ws_connected(self) -> None:
        if self._ws is not None:
            return
        if not self._matched_game:
            raise RuntimeError("No matched game info available for websocket")
        ws_url = self._matched_game.get("websocket_url")
        token = self._matched_game.get("game_token")
        if not ws_url or not token:
            raise RuntimeError("Missing websocket_url or game_token in match info")
        # websocket_url from API may be a path - build absolute
        if ws_url.startswith("/"):
            scheme_host = self.base_url
            ws_url_full = (
                scheme_host.replace("http", "ws") + ws_url + f"?game_token={token}"
            )
        else:
            ws_url_full = ws_url
        self._ws = websocket.create_connection(ws_url_full)
        # start reader thread
        self._ws_running = True
        self._ws_thread = threading.Thread(target=self._ws_reader, daemon=True)
        self._ws_thread.start()
        # start pinger thread to request authoritative state periodically
        self._ws_pinger_thread = threading.Thread(target=self._ws_pinger, daemon=True)
        self._ws_pinger_thread.start()

    def _ws_reader(self) -> None:
        # Read messages from websocket and update live state
        while self._ws_running and self._ws is not None:
            try:
                raw = self._ws.recv()
                if not raw:
                    # remote closed connection
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.exception("Failed to parse websocket message")
                    continue
                mtype = msg.get("type")
                data = msg.get("data")
                if mtype in ("state_update", "game_over"):
                    # store authoritative live state
                    self._last_state = data or {}
                    # clear draw offers on game over
                    if mtype == "game_over":
                        self._draw_offered_by = None
                elif mtype == "draw_offered":
                    self._draw_offered_by = (data or {}).get("by")
                elif mtype == "error":
                    logger.warning("WebSocket error message: %s", data)
                    if isinstance(data, dict) and (
                        "result" in data
                        or "ended_at" in data
                        or ("game_id" in data and "fen" in data)
                    ):
                        self._last_state = data or {}
                        if data.get("result") is not None:
                            self._draw_offered_by = None
            except Exception:
                # connection likely closed or broken
                logger.exception(
                    "WebSocket reader exiting for game_id=%s", self._game_id
                )
                break
        # cleanup when reader exits
        self._ws_running = False
        try:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    logger.exception("Failed to close websocket on reader exit")
        finally:
            self._ws = None

    def _ws_pinger(self) -> None:
        # Periodically request authoritative state from server to ensure we
        # observe postgame transitions (some servers only push state in
        # response to an explicit `get_state` request).
        while self._ws_running:
            try:
                now = time.time()
                if now - self._last_get_state_time >= self._get_state_interval:
                    if self._ws is not None:
                        try:
                            logger.info(
                                "Sending get_state to game_id=%s", self._game_id
                            )
                            self._ws.send(json.dumps({"type": "get_state", "data": {}}))
                            self._last_get_state_time = now
                        except Exception:
                            logger.exception(
                                "Failed to send get_state via websocket pinger"
                            )
            except Exception:
                logger.exception("WebSocket pinger exiting due to error")
                break
