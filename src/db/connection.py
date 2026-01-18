from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import sessionmaker

from src.db.config import PostgresConfig


def database_url(config: PostgresConfig) -> URL:
    return URL.create(
        drivername="postgresql+psycopg2",
        username=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database,
        query={"sslmode": config.sslmode},
    )


def get_engine(config: PostgresConfig) -> Engine:
    return create_engine(database_url(config), pool_pre_ping=True, future=True)


def get_sessionmaker(config: PostgresConfig) -> sessionmaker:
    return sessionmaker(bind=get_engine(config), autoflush=False, expire_on_commit=False, future=True)
