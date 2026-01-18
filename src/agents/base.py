from __future__ import annotations

from abc import ABC, abstractmethod

from src.web.client import ChessWebClient

class Agent(ABC):
    def __init__(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        web_client: ChessWebClient,
    ) -> None:
        self._username = username
        self._password = password
        self._email = email
        self._classpath = classpath
        self._web_client = web_client
        self._registered = False
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

    def snapshot_state(self) -> dict:
        return {"registered": self._registered}

    def load_state(self, state: dict) -> None:
        self._registered = bool(state.get("registered", False))

    def ensure_registered(self) -> None:
        if self._registered:
            return
        self._web_client.sign_up(
            email=self._email,
            username=self._username,
            password=self._password,
        )
        self._registered = True

    def sign_in(self) -> None:
        self._web_client.sign_in(username=self._username, password=self._password)

    def sign_out(self) -> None:
        self._web_client.sign_out()
