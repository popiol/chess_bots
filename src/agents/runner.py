from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from src.agents.manager import AgentManager


@dataclass(frozen=True)
class RunnerConfig:
    create_probability: float = 1 / 3600
    start_probability: float = 1 / 300
    tick_sleep_seconds: float = 1.0
    max_active_sessions: int = 10


class AgentRunner:
    def __init__(
        self,
        manager: AgentManager,
        classpaths: Iterable[str],
        config: RunnerConfig | None = None,
    ) -> None:
        self._manager = manager
        self._classpaths = list(classpaths)
        if not self._classpaths:
            raise ValueError("At least one classpath is required.")
        self._config = config or RunnerConfig()

    def main_loop(self) -> None:
        while True:
            self._maybe_create_agent()
            self._maybe_start_session()
            self._run_active_sessions()
            time.sleep(self._config.tick_sleep_seconds)

    def _maybe_create_agent(self) -> None:
        if random.random() >= self._config.create_probability:
            return
        if (
            len(self._manager.list_active_sessions())
            >= self._config.max_active_sessions
        ):
            return
        username = self._random_username()
        password = self._random_password()
        email = f"{username}@playbullet.gg"
        classpath = random.choice(self._classpaths)
        self._manager.create_agent(
            username=username,
            password=password,
            email=email,
            classpath=classpath,
            state={},
        )

    def _maybe_start_session(self) -> None:
        if random.random() >= self._config.start_probability:
            return
        if (
            len(self._manager.list_active_sessions())
            >= self._config.max_active_sessions
        ):
            return
        usernames = self._manager.list_known_agents()
        if not usernames:
            return
        username = random.choice(usernames)
        self._manager.start_session(username)

    def _run_active_sessions(self) -> None:
        for username, agent in self._manager.list_active_sessions().items():
            agent.run()
            if agent.session_done:
                self._manager.end_session(username)

    @staticmethod
    def _random_username() -> str:
        return f"agent_{uuid4().hex[:10]}"

    @staticmethod
    def _random_password() -> str:
        return uuid4().hex
