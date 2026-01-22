from __future__ import annotations

import logging
from datetime import datetime, timezone
from importlib import import_module
from typing import Callable, Type

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
            return self._active_sessions[username]

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
            raise

        self._active_sessions[username] = agent_instance
        self._session_clients[username] = web_client
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
        web_client = self._session_clients.pop(username, None)
        if web_client is not None:
            web_client.close()
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
