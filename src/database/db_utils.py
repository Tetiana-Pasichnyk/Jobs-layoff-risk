import mysql.connector
from mysql.connector.constants import ClientFlag

DEFAULT_HOST = "localhost"
DEFAULT_USER = "root"
DEFAULT_DATABASE = "AI_Impact_DB"


def get_server_connection(port=0, password="root", host=DEFAULT_HOST, user=DEFAULT_USER):
    db_config = {
        "host": host,
        "user": user,
        "port": port,
        "password": password,
        "client_flags": [ClientFlag.MULTI_STATEMENTS],
    }

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"\n[CONNECTION ERROR]: Failed to connect to MySQL: {err}")
        raise err


def get_db_connection(port=0, password="root", host=DEFAULT_HOST, user=DEFAULT_USER):
    db_config = {
        "host": host,
        "user": user,
        "port": port,
        "password": password,
        "database": DEFAULT_DATABASE,
    }

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"\n[CONNECTION ERROR]: Failed to connect to MySQL: {err}")
        raise err


def run_sql_file(sql_path, port=0, password="root", host=DEFAULT_HOST, user=DEFAULT_USER):
    sql = sql_path.read_text(encoding="utf-8")
    conn, cursor = get_server_connection(port=port, password=password, host=host, user=user)

    try:
        for result in cursor.execute(sql, multi=True):
            if result.with_rows:
                result.fetchall()
        conn.commit()
    finally:
        cursor.close()
        conn.close()