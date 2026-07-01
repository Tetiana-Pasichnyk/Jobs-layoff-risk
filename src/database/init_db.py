from pathlib import Path

from src.database import db_utils, settings

SQL_PATH = Path(__file__).resolve().parent / "sql-ai-impact-jobs-layoff-risk.sql"


def init_database():
    db_utils.run_sql_file(
        SQL_PATH,
        port=settings.DB_PORT,
        password=settings.DB_PASSWORD,
    )
    print(f"SUCCESS: Database schema created from {SQL_PATH.name}")


if __name__ == "__main__":
    init_database()
