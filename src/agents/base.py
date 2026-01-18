from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.web.client import ChessWebClient

logger = logging.getLogger(__name__)


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
            logger.info("Already registered", extra={"username": self._username})
            return
        logger.info(
            "Registering email=%s password=%s",
            self._email,
            self._password,
            extra={"username": self._username},
        )
        self._web_client.sign_up(
            email=self._email,
            username=self._username,
            password=self._password,
        )
        self._registered = True
        logger.info("Registered", extra={"username": self._username})

    def sign_in(self) -> None:
        if not self._web_client.is_sign_in_available():
            logger.info(
                "Sign-in button not available, assuming already signed in",
                extra={"username": self._username},
            )
            return
        logger.info("Signing in", extra={"username": self._username})
        self._web_client.sign_in(username=self._username, password=self._password)
        logger.info("Signed in", extra={"username": self._username})

    def sign_out(self) -> None:
        logger.info("Signing out", extra={"username": self._username})
        self._web_client.sign_out()
        logger.info("Signed out", extra={"username": self._username})
