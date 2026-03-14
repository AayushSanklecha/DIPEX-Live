"""
Populate hackathon data directly into the existing DIPEX Docker containers
- dipex-postgres  →  user: dipex   pass: dipex_secret   db: dipex   (port 5433 on host)
- dipex-mongo     →  user: dipex   pass: dipex_secret   db: dipex   (port 27018 on host)
"""
import random
import psycopg2
import psycopg2.extras
from pymongo import MongoClient
from datetime import datetime, timedelta

# ── PostgreSQL ──────────────────────────────────────────────────────────────
print("Connecting to dipex-postgres (port 5433)...")
pg = psycopg2.connect("postgresql://dipex:dipex_secret@localhost:5433/dipex")
pg.autocommit = True
cur = pg.cursor()

cur.execute("DROP TABLE IF EXISTS hackathon_users;")
cur.execute("""
CREATE TABLE hackathon_users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE,
    credit_score INT,
    income NUMERIC(12,2),
    country VARCHAR(50),
    churn_risk VARCHAR(10)
);
""")

countries = ["USA", "UK", "Canada", "Germany", "India", "Australia"]
risk = ["High", "Low", "Medium"]
rows = []
for i in range(1000):
    rows.append((
        f"User_{i}",
        f"user{i}@example.com",
        (datetime.now() - timedelta(days=random.randint(10, 1000))).date(),
        random.randint(400, 850) if random.random() > 0.05 else None,
        round(random.uniform(25000, 200000), 2) if random.random() > 0.04 else None,
        random.choice(countries),
        random.choice(risk),
    ))

psycopg2.extras.execute_values(
    cur,
    "INSERT INTO hackathon_users (name, email, signup_date, credit_score, income, country, churn_risk) VALUES %s",
    rows
)
print("hackathon_users (1000 rows) inserted ✓")

# Also create transactions table
cur.execute("DROP TABLE IF EXISTS hackathon_transactions;")
cur.execute("""
CREATE TABLE hackathon_transactions (
    tx_id SERIAL PRIMARY KEY,
    user_id INT,
    amount NUMERIC(12,2),
    currency VARCHAR(5),
    status VARCHAR(20),
    tx_date DATE
);
""")
txns = []
statuses = ["COMPLETED","COMPLETED","COMPLETED","FAILED","PENDING"]
for i in range(2000):
    txns.append((
        random.randint(1, 1000),
        round(random.uniform(5, 8000), 2) if random.random() > 0.03 else None,
        random.choice(["USD","EUR","INR","GBP"]),
        random.choice(statuses),
        (datetime.now() - timedelta(days=random.randint(0, 60))).date(),
    ))

psycopg2.extras.execute_values(
    cur,
    "INSERT INTO hackathon_transactions (user_id, amount, currency, status, tx_date) VALUES %s",
    txns
)
print("hackathon_transactions (2000 rows) inserted ✓")
pg.close()

# ── MongoDB ─────────────────────────────────────────────────────────────────
print("\nConnecting to dipex-mongo (port 27018)...")
mc = MongoClient("mongodb://dipex:dipex_secret@localhost:27018/")
db = mc["dipex"]

db["hackathon_device_logs"].drop()
logs = []
for i in range(2000):
    logs.append({
        "log_id": f"LOG-{i:05d}",
        "device_type": random.choice(["iOS","Android","Web","Desktop","SmartTV"]),
        "ip_address": f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
        "user_agent": random.choice(["Chrome/120","Safari/17","Firefox/121","Edge/120"]),
        "session_duration_sec": round(random.uniform(5.0, 7200.0), 2) if random.random() > 0.1 else None,
        "pages_visited": random.randint(1, 30),
        "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 20000)),
    })

db["hackathon_device_logs"].insert_many(logs)
print("hackathon_device_logs (2000 documents) inserted ✓")

mc.close()

print("\n✅ All hackathon data loaded into existing DIPEX containers!")
print("PostgreSQL host=localhost port=5433 db=dipex user=dipex pass=dipex_secret")
print("  Tables: hackathon_users, hackathon_transactions")
print("MongoDB   host=localhost port=27018 db=dipex user=dipex pass=dipex_secret")
print("  Collections: hackathon_device_logs")
