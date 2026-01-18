from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.db.config import PostgresConfig
from src.db.connection import get_sessionmaker
from src.db.schema import AgentMetadata


@dataclass
class AgentRecord:
    id: Optional[int]
    username: str
    password: str
    email: str
    classpath: str
    state: Optional[dict]
    last_session_start: Optional[datetime]
    last_session_end: Optional[datetime]


def create_agent(config: PostgresConfig, record: AgentRecord) -> int:
    SessionLocal = get_sessionmaker(config)
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


def get_agent_by_username(
    config: PostgresConfig, username: str
) -> Optional[AgentRecord]:
    SessionLocal = get_sessionmaker(config)
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


def list_agent_usernames(config: PostgresConfig) -> list[str]:
    SessionLocal = get_sessionmaker(config)
    with SessionLocal() as session:
        rows = session.query(AgentMetadata.username).all()
        return [row[0] for row in rows]


def update_agent_state(
    config: PostgresConfig,
    username: str,
    state: Optional[dict],
) -> None:
    SessionLocal = get_sessionmaker(config)
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
    config: PostgresConfig,
    username: str,
    last_session_start: Optional[datetime],
    last_session_end: Optional[datetime],
) -> None:
    SessionLocal = get_sessionmaker(config)
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
