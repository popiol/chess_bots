from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from src.agents.manager import AgentManager
from src.db.repository import AgentRepository
from src.web.config import BrowserConfig
from src.web.factory import WebClientFactory
from src.web.selectors import site_selectors

LOG_DIR = Path("logs")

logger = logging.getLogger(__name__)


class UsernameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "username"):
            record.username = "-"
        return True


@dataclass(frozen=True)
class RunnerConfig:
    create_probability: float = 1 / 3600
    start_probability: float = 1 / 60
    tick_sleep_seconds: float = 1.0
    max_active_sessions: int = 30


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

    def run_single_session(
        self, *, classpath: str | None = None, username: str | None = None
    ) -> str:
        if (classpath is None) == (username is None):
            raise ValueError("Provide exactly one of classpath or username.")

        if classpath is not None:
            username = self._create_random_agent(classpath=classpath)

        assert username is not None
        agent = self._manager.start_session(username)
        logger.info("Session started", extra={"username": username})
        while True:
            agent.run()
            if agent.session_done:
                logger.info("Session done", extra={"username": username})
                self._manager.end_session(username)
                return username
            time.sleep(self._config.tick_sleep_seconds)

    def _maybe_create_agent(self) -> None:
        if random.random() >= self._config.create_probability:
            return
        if (
            len(self._manager.list_active_sessions())
            >= self._config.max_active_sessions
        ):
            return
        self._create_random_agent()

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
        active_usernames = set(self._manager.list_active_sessions().keys())
        candidates = [name for name in usernames if name not in active_usernames]
        if not candidates:
            return
        username = random.choice(candidates)
        logger.info("Starting session", extra={"username": username})
        self._manager.start_session(username)
        active_count = len(self._manager.list_active_sessions())
        logger.info("Active sessions: %d", active_count, extra={"username": username})

    def _run_active_sessions(self) -> None:
        for username, agent in self._manager.list_active_sessions().items():
            agent.run()
            if agent.session_done:
                logger.info("Ending session", extra={"username": username})
                self._manager.end_session(username)

    @staticmethod
    def _random_username() -> str:
        return f"agent_{uuid4().hex[:10]}"

    @staticmethod
    def _random_password() -> str:
        return uuid4().hex

    def _create_random_agent(self, *, classpath: str | None = None) -> str:
        username = self._random_username()
        password = self._random_password()
        email = f"{username}@playbullet.gg"
        classpath = classpath or random.choice(self._classpaths)
        logger.info(
            "Creating agent classpath=%s",
            classpath,
            extra={"username": username},
        )
        self._manager.create_agent(
            username=username,
            password=password,
            email=email,
            classpath=classpath,
            state={},
        )
        return username


def main() -> None:
    classpath_map = {
        "CustomizableAgent": "src.agents.customizable_agent.CustomizableAgent",
    }
    available_classnames = list(classpath_map.keys())
    base_url = "https://playbullet.gg"
    parser = argparse.ArgumentParser(description="Run the agent runner loop.")
    parser.add_argument(
        "--classname",
        choices=available_classnames,
        help="Run a single session for a newly created agent class.",
    )
    parser.add_argument(
        "--username",
        help="Run a single session for an existing agent username.",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "runner.log"
    handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(username)s %(message)s"))
    handler.addFilter(UsernameFilter())
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    repo = AgentRepository.from_env()
    web_factory = WebClientFactory(
        BrowserConfig(base_url=base_url),
        site_selectors(),
    )
    manager = AgentManager(repo, web_factory.create_client)
    runner = AgentRunner(
        manager,
        classpaths=list(classpath_map.values()),
        config=RunnerConfig(),
    )
    try:
        if args.classname or args.username:
            classpath = (
                classpath_map[args.classname] if args.classname is not None else None
            )
            runner.run_single_session(classpath=classpath, username=args.username)
        else:
            runner.main_loop()
    finally:
        web_factory.close()


if __name__ == "__main__":
    main()
