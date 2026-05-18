import sqlite3
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_mock_data():
    db_path = "Mock_DIPEX_Database.db"
    
    print(f"Creating mock SQLite database at {os.path.abspath(db_path)}...")
    conn = sqlite3.connect(db_path)
    
    # Generate Mock Banking Data (Transactions)
    n_records = 5000
    np.random.seed(42)
    random.seed(42)
    
    # Dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(n_records)]
    
    df_banking = pd.DataFrame({
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_records)],
        'account_id': [f"ACC_{random.randint(1000, 1500)}" for _ in range(n_records)],
        'customer_name': [random.choice(["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]) + " " + random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones"]) for _ in range(n_records)],
        'transaction_date': [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        'amount': np.round(np.random.exponential(scale=500, size=n_records), 2),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_records, p=[0.7, 0.2, 0.1]),
        'merchant_category': np.random.choice(['Retail', 'Food', 'Travel', 'Entertainment', 'Online'], n_records),
        'is_flagged_fraud': np.random.choice([0, 1], n_records, p=[0.98, 0.02]),
        'customer_age': np.random.randint(18, 85, n_records),
        'email': [f"user{random.randint(1,9999)}@example.com" for _ in range(n_records)] # Intentional PII
    })
    
    # Introduce some nulls for testing DataRescue
    df_banking.loc[np.random.choice(df_banking.index, size=100, replace=False), 'amount'] = np.nan
    df_banking.loc[np.random.choice(df_banking.index, size=50, replace=False), 'customer_age'] = np.nan
    
    df_banking.to_sql('banking_transactions', conn, if_exists='replace', index=False)
    
    # Generate Mock Healthcare Data (Patients)
    df_healthcare = pd.DataFrame({
        'patient_id': [f"PAT_{i:04d}" for i in range(2000)],
        'admission_date': [d.strftime("%Y-%m-%d") for d in dates[:2000]],
        'diagnosis_code': np.random.choice(['ICD-10-A00', 'ICD-10-B15', 'ICD-10-C34', 'ICD-10-E11', 'ICD-10-I21'], 2000),
        'treatment_cost': np.round(np.random.normal(loc=5000, scale=1500, size=2000), 2),
        'patient_age': np.random.randint(1, 99, 2000),
        'readmitted_30_days': np.random.choice(['Yes', 'No'], 2000, p=[0.15, 0.85]),
        'ssn': [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" for _ in range(2000)] # Intentional PII
    })
    
    # Clean up negative costs
    df_healthcare['treatment_cost'] = df_healthcare['treatment_cost'].apply(lambda x: x if x > 0 else 100)
    
    df_healthcare.to_sql('patient_records', conn, if_exists='replace', index=False)
    
    conn.close()
    print("Mock Database generated successfully! You can find it at Mock_DIPEX_Database.db")
    print("-----------------------------------------------------------------------------")
    print("To test the Database ingestion in DIPEX:")
    print("1. Go to the Run Pipeline > Connect to Database tab")
    print("2. Set Database Type to 'SQLite'")
    print(f"3. Put this absolute path in Database Name: {os.path.abspath(db_path)}")
    print("4. Click Discover Tables and click 'banking_transactions'")

if __name__ == "__main__":
    generate_mock_data()
