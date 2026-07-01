import os

from dotenv import load_dotenv

from src.paths import CSV_PATH, DATA_DIR, OUTPUT_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_PORT = int(os.getenv("DB_PORT", "8889"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
