import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

host = os.getenv('DB_HOST')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

print("Connecting to AWS RDS server...")

# Connect WITHOUT specifying the database name
connection = pymysql.connect(
    host=host,
    user=user,
    password=password,
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        print(f"Creating database `{db_name}`...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
        print(f"SUCCESS: Database `{db_name}` has been created!")
    connection.commit()
finally:
    connection.close()