import os

from src.paths import CSV_PATH, DATA_DIR, OUTPUT_DIR

DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_PORT = int(os.getenv("DB_PORT", "8889"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
