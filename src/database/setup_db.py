from src.database.data_clean import load_csv_to_db
from src.database.init_db import init_database


def setup_database():
    init_database()
    load_csv_to_db()


if __name__ == "__main__":
    setup_database()
