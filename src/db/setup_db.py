from src.db.config import PostgresConfig
from src.db.schema import ensure_schema


def main() -> None:
    config = PostgresConfig.from_env()
    ensure_schema(config)


if __name__ == "__main__":
    main()
