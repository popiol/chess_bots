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

import psutil
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

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
    create_interval_seconds: float = 60.0  # How often to try creating agents
    start_interval_seconds: float = 60.0  # How often to try starting sessions
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
        self._last_create_time = time.time()
        self._last_start_time = 0.0
        # Track consecutive exceptions per session; end session after threshold
        self._consecutive_failures: dict[str, int] = {}
        self._max_consecutive_failures = 3
        # Track consecutive start failures (Playwright timeouts)
        self._start_failures = 0
        self._max_start_failures = 3

    def main_loop(self) -> None:
        while True:
            self._maybe_create_agent()
            self._maybe_start_session()
            self._run_active_sessions()

    def run_single_session(
        self, *, classpath: str | None = None, username: str | None = None
    ) -> str:
        if (classpath is None) == (username is None):
            raise ValueError("Provide exactly one of classpath or username.")

        if classpath is not None:
            username = self._create_random_agent(classpath=classpath)

        assert username is not None
        agent = self._manager.start_session(username)
        while True:
            agent.run()
            if agent.session_done:
                logger.info("Session done", extra={"username": username})
                self._manager.end_session(username)
                return username

    def _maybe_create_agent(self) -> None:
        current_time = time.time()
        if current_time - self._last_create_time < self._config.create_interval_seconds:
            return
        if self._manager.active_session_count() >= self._config.max_active_sessions:
            return
        known_agents = self._manager.list_known_agents()
        if len(known_agents) >= self._config.max_active_sessions:
            return
        self._last_create_time = current_time
        self._create_random_agent()

    def _maybe_start_session(self) -> None:
        try:
            current_time = time.time()
            if (
                current_time - self._last_start_time
                < self._config.start_interval_seconds
            ):
                return
            if self._manager.active_session_count() >= self._config.max_active_sessions:
                return
            self._last_start_time = current_time
            usernames = self._manager.list_known_agents()
            if not usernames:
                return
            active_usernames = self._manager.active_session_usernames()
            candidates = [name for name in usernames if name not in active_usernames]
            if not candidates:
                return
            username = random.choice(candidates)
            # Ensure sufficient free memory before starting a new session
            min_bytes = 1000 * 1024 * 1024  # 1000 MB
            if psutil.virtual_memory().available < min_bytes:
                logger.warning(
                    "Insufficient free memory to start session: available=%d, need >=%d bytes",
                    psutil.virtual_memory().available,
                    min_bytes,
                    extra={"username": username},
                )
                return

            self._manager.start_session(username)
            logger.info("Session started", extra={"username": username})
            active_count = self._manager.active_session_count()
            logger.info(
                "Active sessions: %d", active_count, extra={"username": username}
            )
        except PlaywrightTimeoutError as e:
            self._start_failures += 1
            logger.warning(
                "Failed to start session due to timeout (count=%d): %s",
                self._start_failures,
                str(e),
                extra={"username": username},
            )
            if self._start_failures >= self._max_start_failures:
                logger.error(
                    "Repeated start timeouts (%d), escalating",
                    self._start_failures,
                    extra={"username": username},
                )
                raise
            return
        else:
            self._start_failures = 0

    def _run_active_sessions(self) -> None:
        for username, agent in self._manager.active_sessions_items():
            try:
                agent.run()
            except Exception:
                cnt = self._consecutive_failures.get(username, 0) + 1
                self._consecutive_failures[username] = cnt
                logger.exception(
                    "Agent.run() raised an exception (count=%d)",
                    cnt,
                    extra={"username": username},
                )
                if cnt >= self._max_consecutive_failures:
                    logger.error(
                        "Too many consecutive failures (%d), ending session",
                        cnt,
                        extra={"username": username},
                    )
                    agent.session_done = True

            if username in self._consecutive_failures:
                self._consecutive_failures.pop(username, None)

            if agent.session_done:
                logger.info("Ending session", extra={"username": username})
                self._manager.end_session(username)

    @staticmethod
    def _random_username() -> str:
        p = Path("data/usernames.csv")
        with p.open(encoding="utf-8") as f:
            names = f.read().splitlines()
        names = [n.strip() for n in names if n.strip()]
        base = random.choice(names)
        length = random.randint(0, 3)
        if length == 0:
            suffix = ""
        else:
            max_val = 10**length - 1
            num = random.randint(0, max_val)
            suffix = str(num).zfill(length)
        return f"{base}{suffix}"

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
        metadata = self._manager.create_agent(
            username=username,
            password=password,
            email=email,
            classpath=classpath,
            state={},
        )
        return metadata.username


def main() -> None:
    classpath_map = {
        "NeuralNetworkAgent": "src.agents.neural_network_agent.NeuralNetworkAgent",
        "StockfishAgent": "src.agents.stockfish_agent.StockfishAgent",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
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
    root_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
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
