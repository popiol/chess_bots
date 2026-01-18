from __future__ import annotations

from datetime import datetime, timezone
import logging
from importlib import import_module
from typing import Type

from src.agents.base import Agent
from src.db.repository import AgentRepository
from src.db.schema import AgentMetadata
from src.web.client import ChessWebClient

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, repo: AgentRepository, web_client: ChessWebClient) -> None:
        self._web_client = web_client
        self._active_sessions: dict[str, Agent] = {}
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
            logger.warning("Unknown agent username=%s", username)
            raise RuntimeError(f"Unknown agent: {username}")

        agent_class = self._load_agent_class(agent_meta.classpath)
        agent_instance = agent_class(
            agent_meta.username,
            agent_meta.password,
            agent_meta.email,
            agent_meta.classpath,
            self._web_client,
        )
        agent_instance.load_state(agent_meta.state)
        self._repo.update_session_times(
            agent_meta.username, datetime.now(timezone.utc), None
        )

        self._active_sessions[username] = agent_instance
        logger.info("Session started username=%s", username)
        return agent_instance

    def end_session(self, username: str) -> None:
        agent = self._active_sessions.pop(username, None)
        if not agent:
            return
        self._repo.update_agent_state(agent.username, agent.snapshot_state())
        self._repo.update_session_times(
            agent.username, None, datetime.now(timezone.utc)
        )
        logger.info("Session ended username=%s", username)

    def active_session(self, username: str) -> Agent | None:
        return self._active_sessions.get(username)

    def list_active_sessions(self) -> dict[str, Agent]:
        return dict(self._active_sessions)

    def list_known_agents(self) -> list[str]:
        return self._repo.list_agent_usernames()

    @staticmethod
    def _load_agent_class(classpath: str) -> Type[Agent]:
        module_path, class_name = classpath.rsplit(".", 1)
        module = import_module(module_path)
        agent_class = getattr(module, class_name)
        assert issubclass(agent_class, Agent)
        return agent_class
