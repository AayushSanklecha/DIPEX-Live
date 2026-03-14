import time
import random
import psycopg2
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta

print("Waiting for databases to initialize...")
time.sleep(5)  # give docker containers a few seconds to start up

# -----------------
# 1. PostgreSQL
# -----------------
print("Connecting to PostgreSQL...")
pg_conn = psycopg2.connect("postgresql://hackathon:password123@localhost:5432/dipex_demo")
pg_conn.autocommit = True
cur = pg_conn.cursor()

print("Populating PostgreSQL tables...")
cur.execute("DROP TABLE IF EXISTS users;")
cur.execute("""
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE,
    credit_score INT,
    country VARCHAR(50)
);
""")

users_data = []
countries = ["USA", "UK", "Canada", "Germany", "India", "Australia"]
for i in range(1000):
    users_data.append((
        f"User_{i}",
        f"user{i}@example.com",
        (datetime.now() - timedelta(days=random.randint(10, 1000))).date(),
        random.randint(400, 850) if random.random() > 0.05 else None,  # 5% missing
        random.choice(countries)
    ))

psycopg2.extras = __import__("psycopg2.extras").extras
psycopg2.extras.execute_values(
    cur,
    "INSERT INTO users (name, email, signup_date, credit_score, country) VALUES %s",
    users_data
)
pg_conn.close()
print("PostgreSQL 'users' table populated! (1000 rows)")


# -----------------
# 2. MongoDB
# -----------------
print("\nConnecting to MongoDB...")
mongo_client = MongoClient("mongodb://hackathon:password123@localhost:27017/")
db = mongo_client["dipex_demo"]

print("Populating MongoDB collections...")
devices_col = db["device_logs"]
devices_col.drop()

mongo_data = []
for i in range(1500):
    mongo_data.append({
        "log_id": f"LOG-{i}",
        "device_type": random.choice(["iOS", "Android", "Web", "Desktop"]),
        "ip_address": f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}",
        "session_duration_sec": random.uniform(10.0, 3600.0) if random.random() > 0.1 else None, # 10% nulls
        "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 10000))
    })

devices_col.insert_many(mongo_data)
mongo_client.close()
print("MongoDB 'device_logs' collection populated! (1500 documents)")

print("\nAll hackathon data is successfully loaded into Postgres and MongoDB!")
