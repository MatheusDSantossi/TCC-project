import os
import psycopg2
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv(encoding="UTF-8")

def get_db_connection():
    conn = psycopg2.connect(
            database= os.environ.get("DATABASE_DB"),
            user=os.environ.get("DATABASE_USER"),
            password=os.environ.get("DATABASE_PASSWORD"),
            host=os.environ.get("DATABASE_HOST"),
            port=os.environ.get("DATABASE_PORT")
        )
    conn.autocommit = True
    
    return conn

conn = get_db_connection()
