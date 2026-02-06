from __future__ import annotations

import gc
import logging
import tracemalloc
from datetime import datetime, timezone
from importlib import import_module
from typing import Callable, Type

import psutil

from src.agents.base import Agent
from src.db.repository import AgentRepository
from src.db.schema import AgentMetadata
from src.web.client import ChessWebClient

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(
        self,
        repo: AgentRepository,
        web_client_factory: Callable[[], ChessWebClient],
    ) -> None:
        self._web_client_factory = web_client_factory
        self._active_sessions: dict[str, Agent] = {}
        self._session_clients: dict[str, ChessWebClient] = {}
        self._repo = repo

    def create_agent(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        state: dict,
    ) -> AgentMetadata:
        return self._repo.create_agent_metadata(
            username=username,
            password=password,
            email=email,
            classpath=classpath,
            state=state,
        )

    def start_session(self, username: str) -> Agent:
        if username in self._active_sessions:
            logger.info("Session already active", extra={"username": username})
            return self._active_sessions[username]

        logger.info("Starting session", extra={"username": username})

        agent_meta = self._repo.get_metadata_by_username(username)
        if not agent_meta:
            logger.warning("Unknown agent", extra={"username": username})
            raise RuntimeError(f"Unknown agent: {username}")

        agent_class = self._load_agent_class(agent_meta.classpath)
        web_client = self._web_client_factory()
        try:
            agent_instance = agent_class(
                agent_meta.username,
                agent_meta.password,
                agent_meta.email,
                agent_meta.classpath,
                web_client,
            )
            agent_instance.load_state(agent_meta.state)
            self._repo.update_session_times(
                agent_meta.username, datetime.now(timezone.utc), None
            )
        except Exception:
            web_client.close()
            del web_client
            raise

        self._active_sessions[username] = agent_instance
        self._session_clients[username] = web_client  # noqa: F821
        logger.info(
            "Session started class=%s",
            agent_class.__name__,
            extra={"username": username},
        )
        return agent_instance

    def end_session(self, username: str) -> None:
        agent = self._active_sessions.pop(username, None)
        if not agent:
            return
        self._repo.update_agent_state(agent.username, agent.snapshot_state())
        self._repo.update_session_times(
            agent.username, None, datetime.now(timezone.utc)
        )
        web_client = self._session_clients.pop(username)
        web_client.close()
        del web_client

        # Try to free Python-level resources and log memory usage (RSS and tracemalloc)
        try:
            del agent
            gc.collect()

            proc = psutil.Process()
            rss = proc.memory_info().rss
            if tracemalloc.is_tracing():
                traced_current, traced_peak = tracemalloc.get_traced_memory()
            else:
                traced_current = traced_peak = 0

            logger.info(
                "Memory after session end: RSS=%.1fMB, tracemalloc current=%.1fMB, peak=%.1fMB",
                rss / 1024.0 / 1024.0,
                traced_current / 1024.0 / 1024.0,
                traced_peak / 1024.0 / 1024.0,
                extra={"username": username},
            )

            # Log child processes (browsers, engines) which often hold large native memory
            for child in proc.children(recursive=True):
                try:
                    mem = child.memory_info().rss
                    cmd = " ".join(child.cmdline())
                    logger.info(
                        "Child pid=%d name=%s RSS=%.1fMB cmd=%s",
                        child.pid,
                        child.name(),
                        mem / 1024.0 / 1024.0,
                        cmd,
                        extra={"username": username},
                    )
                except psutil.NoSuchProcess:
                    continue
        except Exception:
            logger.exception(
                "Failed during end_session cleanup", extra={"username": username}
            )

        logger.info("Session ended", extra={"username": username})

    def active_session(self, username: str) -> Agent | None:
        return self._active_sessions.get(username)

    def active_session_count(self) -> int:
        return len(self._active_sessions)

    def active_session_usernames(self) -> set[str]:
        return set(self._active_sessions.keys())

    def active_sessions_items(self) -> list[tuple[str, Agent]]:
        return list(self._active_sessions.items())

    def list_known_agents(self) -> list[str]:
        return self._repo.list_agent_usernames()

    @staticmethod
    def _load_agent_class(classpath: str) -> Type[Agent]:
        module_path, class_name = classpath.rsplit(".", 1)
        module = import_module(module_path)
        agent_class = getattr(module, class_name)
        assert issubclass(agent_class, Agent)
        return agent_class
