import os
import psycopg2
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()

db = os.getenv("DATABASE_DB")
print("DB: ", db)
user = os.getenv("DATABASE_USER")
print(user)
password = os.getenv("DATABASE_PASSWORD")
print(password)
host = os.getenv("DATABASE_HOST")
print(host)
port = os.getenv("DATABASE_PORT")
print(port)

def get_db_connection():
    conn = psycopg2.connect(
        database=db,
        user=user,
        password=password,
        host=host,
        port=port
    )
    conn.autocommit = True
    
    return conn

conn = get_db_connection()

