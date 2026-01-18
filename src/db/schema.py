from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.db.config import PostgresConfig
from src.db.connection import get_engine


class Base(DeclarativeBase):
    pass


class AgentMetadata(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    password: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False)
    classpath: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    last_session_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_session_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


def ensure_schema(config: PostgresConfig) -> None:
    engine = get_engine(config)
    Base.metadata.create_all(engine)
