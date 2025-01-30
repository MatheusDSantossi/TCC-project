import os
from dotenv import load_dotenv
import psycopg2

# Loading environment variables from .env file
load_dotenv()

database_password = os.getenv('DATABASE_PASSWORD')

conn = psycopg2.connect(
    database= "project_pcc_v3",
    user="postgres",
    password=database_password,
    host="127.0.0.1",
    port="5432"
)

conn.autocommit = True
