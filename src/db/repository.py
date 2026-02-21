from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import sessionmaker

from src.db.config import PostgresConfig
from src.db.connection import get_engine
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
        self._engine = get_engine(config)
        self._sessionmaker = sessionmaker(
            bind=self._engine, autoflush=False, expire_on_commit=False, future=True
        )

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
        with self._sessionmaker() as session:
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
        with self._sessionmaker() as session:
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
        with self._sessionmaker() as session:
            return (
                session.query(AgentMetadata)
                .filter(AgentMetadata.username == username)
                .one_or_none()
            )

    def get_agent_by_username(self, username: str) -> AgentRecord | None:
        with self._sessionmaker() as session:
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

    def list_agent_usernames(self, classpaths: list[str] | None = None) -> list[str]:
        with self._sessionmaker() as session:
            query = session.query(AgentMetadata.username)
            if classpaths:
                query = query.filter(AgentMetadata.classpath.in_(classpaths))
            rows = query.all()
            return [row[0] for row in rows]

    def update_agent_state(self, username: str, state: dict) -> None:
        with self._sessionmaker() as session:
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
        with self._sessionmaker() as session:
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
