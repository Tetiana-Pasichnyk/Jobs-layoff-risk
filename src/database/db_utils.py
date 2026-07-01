import mysql.connector

def get_db_connection(port=0, password="root"):
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'port': port,
        'password': password, 
        'database': 'AI_Impact_DB'
    }
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as err:
        print(f"\n[CONNECTION ERROR]: Failed to connect to MySQL: {err}")
        raise err