import sqlite3
import pandas as pd
import random
import os
from datetime import datetime, timedelta

# Path to the database file
db_path = r"C:\Users\sankl\Desktop\dipex_project\sample_company.db"

# Connect to SQLite database (this creates it if it doesn't exist)
conn = sqlite3.connect(db_path)

print(f"Creating database at: {db_path}...")

# 1. Generate Customers Data
print("Generating 'customers' table...")
customers_data = {
    "customer_id": range(1001, 1051),
    "name": [f"Customer_{i}" for i in range(1, 51)],
    "age": [random.randint(22, 65) for _ in range(50)],
    "segment": [random.choice(["Retail", "Corporate", "Small Business"]) for _ in range(50)],
    "signup_date": [(datetime.now() - timedelta(days=random.randint(10, 1000))).date() for _ in range(50)],
    "churn_risk": [random.choice(["High", "Low", "Medium"]) for _ in range(50)]
}
df_customers = pd.DataFrame(customers_data)
df_customers.to_sql("customers", conn, if_exists="replace", index=False)

# 2. Generate Transactions Data
print("Generating 'transactions' table...")
transactions_data = {
    "tx_id": range(50001, 50201),
    "customer_id": [random.choice(customers_data["customer_id"]) for _ in range(200)],
    "amount": [round(random.uniform(10.0, 5000.0), 2) for _ in range(200)],
    "currency": [random.choice(["USD", "EUR", "GBP"]) for _ in range(200)],
    "status": [random.choice(["COMPLETED", "COMPLETED", "COMPLETED", "FAILED", "PENDING"]) for _ in range(200)],
    "tx_date": [(datetime.now() - timedelta(days=random.randint(0, 30))).date() for _ in range(200)]
}
df_transactions = pd.DataFrame(transactions_data)
df_transactions.to_sql("transactions", conn, if_exists="replace", index=False)

# 3. Add some nulls to simulate real-world data issues
print("Simulating missing data...")
conn.execute("UPDATE customers SET age = NULL WHERE customer_id % 7 = 0")
conn.execute("UPDATE transactions SET amount = NULL WHERE tx_id % 13 = 0")
conn.commit()

print("\nSuccess! Your database is ready.")
print("-" * 40)
print(f"Database File: {db_path}")
print("Tables created:")
print(" - customers (50 rows)")
print(" - transactions (200 rows)")
print("-" * 40)

conn.close()
