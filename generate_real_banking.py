import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

# Set realistic random seed for consistent showcase
np.random.seed(42)

# Generate 5000 realistic records
n_rows = 5000

# Base IDs
transaction_id = [str(uuid.uuid4()) for _ in range(n_rows)]
account_id = [f"ACC_{np.random.randint(100000, 999999)}" for _ in range(n_rows)]

# Transaction Amount (Highly right-skewed, power-law distribution like real banks)
# Lognormal simulates lots of small purchases, with a few massive enterprise transfers
transaction_amount = np.random.lognormal(mean=4.5, sigma=1.8, size=n_rows)

# Implement Real-World Data Issues:
# 1. Inject EXACTLY 0.0 amounts (Common pipeline bug where errors are logged as $0)
zero_idx = np.random.choice(n_rows, size=int(n_rows * 0.03), replace=False)
transaction_amount[zero_idx] = 0.0
transaction_amount = np.round(transaction_amount, 2)

# Fee Amount (~1-3% of transaction amount, plus base flat fee)
fee_amount = np.round(transaction_amount * np.random.uniform(0.01, 0.03, size=n_rows) + np.random.uniform(0.5, 5, size=n_rows), 2)
# Introduce Missing Data (NaNs) to trigger the Cleaner's MICE/KNN imputation
fee_amount[np.random.choice(n_rows, size=int(n_rows * 0.08), replace=False)] = np.nan

# Loan Disbursement & Collateral tracking
loan_amount = np.zeros(n_rows)
collateral_value = np.zeros(n_rows)

# Only 20% of banking records are loan disbursements/adjustments
is_loan = np.random.random(n_rows) < 0.20 
loan_idx = np.where(is_loan)[0]
loan_amount[loan_idx] = np.random.uniform(10000, 850000, size=len(loan_idx))

# Default Healthy Collateral value (LTV Ratio ~ 0.50 to 0.85)
ltv_ratios = np.random.uniform(0.5, 0.85, size=len(loan_idx))

# Inject Malicious LTV Violations (LTV > 0.90) to trigger Banking Regulatory Rules
violators = np.random.choice(len(loan_idx), size=int(len(loan_idx) * 0.08), replace=False)
ltv_ratios[violators] = np.random.uniform(0.92, 1.30, size=len(violators))

collateral_value[loan_idx] = loan_amount[loan_idx] / ltv_ratios
loan_amount = np.round(loan_amount, 2)
collateral_value = np.round(collateral_value, 2)

# Global Currencies
currency = np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CAD'], p=[0.65, 0.15, 0.10, 0.05, 0.05], size=n_rows)

# Messy Timestamps
base_date = datetime.now() - timedelta(days=365)
transaction_date = [(base_date + timedelta(days=int(np.random.randint(0, 365)), minutes=int(np.random.randint(0, 1440)))).isoformat() + "Z" for _ in range(n_rows)]

# Introduce Private Customer Data (PII) to purposefully trigger Governance Redaction!
merchant_category = np.random.choice(["Retail", "Travel", "Cryptocurrency", "Supermarket", "Utility", "Entertainment"], size=n_rows)
# Some rows will leak actual PII emails
customer_email = [f"investor_{i}@gmail.com" if np.random.random() < 0.15 else "" for i in range(n_rows)]

# Target Variable: Is Fraud (Highly imbalanced, ~2% of transactions)
is_fraud = np.zeros(n_rows, dtype=int)
# Transactions over $15,000 to Crypto merchants have a massive 40% fraud probability
fraud_mask = (transaction_amount > 15000) & (merchant_category == "Cryptocurrency") & (np.random.random(n_rows) < 0.4)
is_fraud[fraud_mask] = 1
# Add random background background fraud
is_fraud[np.random.choice(n_rows, size=int(n_rows * 0.015))] = 1

# Compile realistic messy dataset
df = pd.DataFrame({
    'transaction_id': transaction_id,
    'account_id': account_id,
    'transaction_date': transaction_date,
    'merchant_category': merchant_category,
    'transaction_amount': transaction_amount,
    'fee_amount': fee_amount,
    'loan_amount': loan_amount,
    'collateral_value': collateral_value,
    'currency': currency,
    'customer_email_pii': customer_email,
    'is_fraud': is_fraud
})

df.to_csv("mock_banking_data.csv", index=False)
print("Real-world banking dataset with 5,000 rows generated successfully!")
