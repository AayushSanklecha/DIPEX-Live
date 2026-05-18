import sys
import subprocess

def install_deps():
    print("Ensuring dependencies are installed...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "psycopg2-binary", "pymongo"])

try:
    import pandas as pd
    import numpy as np
    import psycopg2
    from pymongo import MongoClient
    from datetime import datetime, timedelta
    import random
except ImportError:
    install_deps()
    import pandas as pd
    import numpy as np
    import psycopg2
    from pymongo import MongoClient
    from datetime import datetime, timedelta
    import random

def generate_mock_data():
    n_records = 5000
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(n_records)]
    
    # ── PostgreSQl: Banking Transactions ──
    print("Generating PostgreSQL Data...")
    df_banking = pd.DataFrame({
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_records)],
        'account_id': [f"ACC_{random.randint(1000, 1500)}" for _ in range(n_records)],
        'customer_name': [random.choice(["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]) + " " + random.choice(["Smith", "Johnson", "Williams"]) for _ in range(n_records)],
        'transaction_date': [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        'amount': np.round(np.random.exponential(scale=500, size=n_records), 2),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_records, p=[0.7, 0.2, 0.1]),
        'is_flagged_fraud': np.random.choice([False, True], n_records, p=[0.98, 0.02]),
        'customer_age': np.random.randint(18, 85, n_records)
    })
    
    # ── MongoDB: Patient Records ──
    print("Generating MongoDB Data...")
    df_healthcare = pd.DataFrame({
        'patient_id': [f"PAT_{i:04d}" for i in range(2000)],
        'admission_date': [d.strftime("%Y-%m-%d") for d in dates[:2000]],
        'diagnosis_code': np.random.choice(['ICD-10-A00', 'ICD-10-B15', 'ICD-10-C34', 'ICD-10-E11'], 2000),
        'treatment_cost': np.round(np.random.normal(loc=5000, scale=1500, size=2000), 2),
        'patient_age': np.random.randint(1, 99, 2000),
        'readmitted_30_days': np.random.choice(['Yes', 'No'], 2000, p=[0.15, 0.85]),
        'ssn': [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" for _ in range(2000)]
    })
    
    # ── MongoDB: eCommerce Orders ──
    df_ecommerce = pd.DataFrame({
        'order_id': [f"ORD_{i:07d}" for i in range(n_records)],
        'customer_id': [f"CST_{random.randint(500, 9999)}" for _ in range(n_records)],
        'product_sku': [f"SKU-{random.randint(10, 99)}" for _ in range(n_records)],
        'quantity': np.random.randint(1, 10, n_records),
        'unit_price': np.round(np.random.uniform(10.0, 500.0), 2),
        'order_status': np.random.choice(['Shipped', 'Pending', 'Delivered', 'Cancelled'], n_records, p=[0.5, 0.2, 0.25, 0.05]),
        'shipping_address': [f"{random.randint(100, 9999)} {random.choice(['Maple', 'Oak', 'Pine', 'Cedar'])} St" for _ in range(n_records)]
    })
    
    # ── MongoDB: HR Employees ──
    df_hr = pd.DataFrame({
        'employee_id': [f"EMP_{i:04d}" for i in range(1500)],
        'first_name': [random.choice(["John", "Jane", "Alice", "Bob", "Clara"]) for _ in range(1500)],
        'last_name': [random.choice(["Doe", "Smith", "Johnson", "Brown", "Taylor"]) for _ in range(1500)],
        'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'], 1500),
        'salary': np.round(np.random.normal(loc=80000, scale=20000, size=1500), 2),
        'hire_date': [d.strftime("%Y-%m-%d") for d in dates[:1500]],
        'job_title': np.random.choice(['Manager', 'Associate', 'Analyst', 'Director', 'Specialist'], 1500)
    })
    
    # ── MongoDB: Telecom Churn ──
    df_telecom = pd.DataFrame({
        'customer_id': [f"TEL_{i:05d}" for i in range(3000)],
        'monthly_charges': np.round(np.random.uniform(20.0, 120.0), 2),
        'total_charges': np.round(np.random.uniform(100.0, 5000.0), 2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 3000, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['Fiber optic', 'DSL', 'No'], 3000),
        'tenure_months': np.random.randint(1, 72, 3000),
        'churn_label': np.random.choice(['Yes', 'No'], 3000, p=[0.25, 0.75])
    })
    
    # ── PostgreSQl: Ambiguous / Messy Ledger ──
    df_messy = pd.DataFrame({
        'sys_id': [f"uid_{random.randint(1000, 9999)}" for _ in range(n_records)],
        'val_flt': np.round(np.random.normal(scale=100, size=n_records), 2),
        'tm_stamp': [d.strftime("%m-%d-%Y %H:%M:%S") for d in dates],
        'score_x': np.random.randint(0, 100, n_records),
        'cat_b': np.random.choice(['A', 'B', 'C', 'D'], n_records),
        'usr_nm': [random.choice(["X10", "X20", "Y30", "Z40"]) for _ in range(n_records)]
    })
    
    return df_banking, df_healthcare, df_messy, df_ecommerce, df_hr, df_telecom

def populate_postgres(df_bank, df_messy):
    print("Connecting to PostgreSQL...")
    try:
        from sqlalchemy import create_engine
        # Connect to existing dipex-postgres exposed on port 5433
        engine = create_engine('postgresql://dipex:dipex_secret@localhost:5433/dipex')
        df_bank.to_sql('banking_transactions', engine, if_exists='replace', index=False)
        print("[SUCCESS] PostgreSQL populated successfully with 'banking_transactions' table!")
        
        df_messy.to_sql('ambiguous_ledger', engine, if_exists='replace', index=False)
        print("[SUCCESS] PostgreSQL populated successfully with 'ambiguous_ledger' table!")
    except Exception as e:
        print(f"[ERROR] Could not connect to Postgres: {e}")

def populate_mongo(df_health, df_ecommerce, df_hr, df_telecom):
    print("Connecting to MongoDB...")
    try:
        # Connect with Authentication to dipex-mongo exposed on port 27018
        client = MongoClient('mongodb://dipex:dipex_secret@localhost:27018/')
        db = client['dipex']
        
        # Helper to load collection
        def load_collection(name, df_data):
            collection = db[name]
            collection.drop()
            records = df_data.to_dict('records')
            collection.insert_many(records)
            print(f"[SUCCESS] MongoDB populated successfully with '{name}' collection!")
            
        load_collection('patient_records', df_health)
        load_collection('ecommerce_orders', df_ecommerce)
        load_collection('hr_employees', df_hr)
        load_collection('telecom_churn', df_telecom)
        
    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB: {e}")

if __name__ == "__main__":
    print("=== DIPEX Mock Database Generator ===")
    df_bank, df_health, df_messy, df_ecommerce, df_hr, df_telecom = generate_mock_data()
    
    # Needs SQLAlchemy to write pandas easily to Postgres
    try:
        import sqlalchemy
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SQLAlchemy"])
        
    populate_postgres(df_bank, df_messy)
    populate_mongo(df_health, df_ecommerce, df_hr, df_telecom)
    print("\nAll done! You can now ingest these directly into DIPEX.")
