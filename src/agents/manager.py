from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from typing import Type

from src.agents.base import Agent
from src.db.config import PostgresConfig
from src.db.connection import get_sessionmaker
from src.db.repository import (
    list_agent_usernames,
    update_agent_state,
    update_session_times,
)
from src.db.schema import AgentMetadata
from src.web.client import ChessWebClient


class AgentManager:
    def __init__(self, db_config: PostgresConfig, web_client: ChessWebClient) -> None:
        self._db_config = db_config
        self._web_client = web_client
        self._active_sessions: dict[str, Agent] = {}

    def create_agent(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        state: dict,
    ) -> AgentMetadata:
        SessionLocal = get_sessionmaker(self._db_config)
        with SessionLocal() as session:
            agent = AgentMetadata(
                username=username,
                password=password,
                email=email,
                classpath=classpath,
                state=state,
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)
            return agent

    def start_session(self, username: str) -> Agent:
        if username in self._active_sessions:
            return self._active_sessions[username]

        SessionLocal = get_sessionmaker(self._db_config)
        with SessionLocal() as session:
            agent_meta = (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )
            if not agent_meta:
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
            update_session_times(
                self._db_config, agent_meta.username, datetime.now(timezone.utc), None
            )

            self._active_sessions[username] = agent_instance
            return agent_instance

    def end_session(self, username: str) -> None:
        agent = self._active_sessions.pop(username, None)
        if not agent:
            return
        update_agent_state(self._db_config, agent.username, agent.snapshot_state())
        update_session_times(
            self._db_config, agent.username, None, datetime.now(timezone.utc)
        )

    def active_session(self, username: str) -> Agent | None:
        return self._active_sessions.get(username)

    def list_active_sessions(self) -> dict[str, Agent]:
        return dict(self._active_sessions)

    def list_known_agents(self) -> list[str]:
        return list_agent_usernames(self._db_config)

    @staticmethod
    def _load_agent_class(classpath: str) -> Type[Agent]:
        module_path, class_name = classpath.rsplit(".", 1)
        module = import_module(module_path)
        agent_class = getattr(module, class_name)
        assert issubclass(agent_class, Agent)
        return agent_class
