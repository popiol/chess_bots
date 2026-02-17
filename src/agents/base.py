from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.web.chess_client import ChessClient

logger = logging.getLogger(__name__)


class Agent(ABC):
    def __init__(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        chess_client: ChessClient,
    ) -> None:
        self._username = username
        self._password = password
        self._email = email
        self._classpath = classpath
        self._chess_client = chess_client
        self.session_done = False

    @property
    def classpath(self) -> str:
        return self._classpath

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password

    @property
    def email(self) -> str:
        return self._email

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot_state(self) -> dict:
        """Return a serializable snapshot of the agent's state."""

    @abstractmethod
    def load_state(self, state: dict) -> None:
        """Load the agent's state from a snapshot produced by snapshot_state."""
