from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Agent(ABC):
    def __init__(self, username: str, password: str, email: str, classpath: str) -> None:
        self._username = username
        self._password = password
        self._email = email
        self._classpath = classpath

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
    def snapshot_state(self) -> Optional[dict]:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: Optional[dict]) -> None:
        raise NotImplementedError
