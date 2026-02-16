from __future__ import annotations

from abc import ABC, abstractmethod


class ChessClient(ABC):
    """Abstract base interface for chess clients (web, headless, test doubles).

    Implementations should provide the same public API used by agents.
    """

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def sign_up(self, email: str, username: str, password: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def sign_in(self, username: str, password: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def sign_out(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def play_as_guest(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def queue_play_now(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_play_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_postgame_visible(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_game_result(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def get_game_reason(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def is_accept_draw_visible(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def select_time_control(self, index: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def offer_draw(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def resign(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def resign_confirm(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def accept_draw(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_move(self, from_square: str, to_square: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_current_fen(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_last_move_valid(self) -> tuple[str | None, bool]:
        raise NotImplementedError

    @abstractmethod
    def get_time_remaining(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def is_current_user_turn(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_player_color(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def time_control_indices(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def is_sign_in_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_sign_out_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_post_login_ready(self) -> bool:
        raise NotImplementedError
