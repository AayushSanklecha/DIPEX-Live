import os
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_ecommerce_data(num_rows=50000, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_rows):
        user_id = f"U{random.randint(1000, 99990)}"
        age = int(random.gauss(35, 10))
        age = max(18, min(80, age))
        
        purchase_amt = round(random.lognormvariate(3.5, 0.8), 2)
        total_purchases = random.randint(1, 50)
        
        days_since_active = random.randint(0, 365)
        last_active = start_date + timedelta(days=random.randint(0, 365))
        
        # Target: Churn
        # Higher churn if not active recently, low purchases, young age
        churn_prob = 0.1
        if days_since_active > 30: churn_prob += 0.3
        if total_purchases < 5: churn_prob += 0.2
        if purchase_amt < 20: churn_prob += 0.1
        
        churned = 1 if random.random() < churn_prob else 0
        
        data.append({
            "user_id": user_id,
            "age": age,
            "last_active_date": last_active.strftime("%Y-%m-%d"),
            "days_since_active": days_since_active,
            "recent_purchase_amount": purchase_amt,
            "total_purchases": total_purchases,
            "lifetime_value": round(purchase_amt * total_purchases * random.uniform(0.8, 1.2), 2),
            "is_churned": churned,
            "browser": random.choice(["Chrome", "Safari", "Firefox", "Edge", "Other"]),
            "device": random.choice(["Mobile", "Desktop", "Tablet"])
        })
        
    df = pd.DataFrame(data)
    out_path = os.path.join(output_dir, "ecommerce_churn.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {num_rows} rows of e-commerce data: {out_path}")

if __name__ == "__main__":
    generate_ecommerce_data()
