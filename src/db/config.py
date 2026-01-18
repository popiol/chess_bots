from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    sslmode: str = "prefer"

    @staticmethod
    def from_env() -> "PostgresConfig":
        return PostgresConfig(
            host=os.environ["CHESS_BOTS_DB_HOST"],
            port=int(os.environ["CHESS_BOTS_DB_PORT"]),
            database=os.environ["CHESS_BOTS_DB_NAME"],
            user=os.environ["CHESS_BOTS_DB_USER"],
            password=os.environ["CHESS_BOTS_DB_PASSWORD"],
            sslmode=os.environ["CHESS_BOTS_DB_SSLMODE"],
        )
