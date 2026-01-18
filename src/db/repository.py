from dataclasses import dataclass
from datetime import datetime

from src.db.config import PostgresConfig
from src.db.connection import get_sessionmaker
from src.db.schema import AgentMetadata
from src.utils.env import load_env


@dataclass
class AgentRecord:
    id: int | None
    username: str
    password: str
    email: str
    classpath: str
    state: dict
    last_session_start: datetime | None
    last_session_end: datetime | None


class AgentRepository:
    def __init__(self, config: PostgresConfig) -> None:
        self._config = config

    @staticmethod
    def from_env() -> "AgentRepository":
        load_env()
        return AgentRepository(PostgresConfig.from_env())

    def create_agent_metadata(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        state: dict,
    ) -> AgentMetadata:
        SessionLocal = get_sessionmaker(self._config)
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

    def create_agent(self, record: AgentRecord) -> int:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            agent = AgentMetadata(
                username=record.username,
                password=record.password,
                email=record.email,
                classpath=record.classpath,
                state=record.state,
                last_session_start=record.last_session_start,
                last_session_end=record.last_session_end,
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)
            return agent.id

    def get_metadata_by_username(self, username: str) -> AgentMetadata | None:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            return (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )

    def get_agent_by_username(self, username: str) -> AgentRecord | None:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            agent = (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )
            if not agent:
                return None
            return AgentRecord(
                id=agent.id,
                username=agent.username,
                password=agent.password,
                email=agent.email,
                classpath=agent.classpath,
                state=agent.state,
                last_session_start=agent.last_session_start,
                last_session_end=agent.last_session_end,
            )

    def list_agent_usernames(self) -> list[str]:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            rows = session.query(AgentMetadata.username).all()
            return [row[0] for row in rows]

    def update_agent_state(self, username: str, state: dict) -> None:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            agent = (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )
            if not agent:
                return
            agent.state = state
            session.commit()

    def update_session_times(
        self,
        username: str,
        last_session_start: datetime | None,
        last_session_end: datetime | None,
    ) -> None:
        SessionLocal = get_sessionmaker(self._config)
        with SessionLocal() as session:
            agent = (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )
            if not agent:
                return
            if last_session_start is not None:
                agent.last_session_start = last_session_start
            if last_session_end is not None:
                agent.last_session_end = last_session_end
            session.commit()
