"""
scripts/generate_sample_data.py
--------------------------------
Generates sample datasets for all DIPEX source types:
  - CSV, Excel, JSON, Parquet  (file-based)
  - PostgreSQL seed SQL
  - MongoDB seed (JSON documents)
  - Redis seed (key-value pairs)
  - Neo4j seed (Cypher statements)
  - DuckDB (in-process, auto-seeded from CSV)

Run:  python scripts/generate_sample_data.py
"""

import os, json, pathlib
import pandas as pd
import numpy as np

SAMPLE_DIR = pathlib.Path("samples")
SAMPLE_DIR.mkdir(exist_ok=True)

# ── 1. Build the master dataset (200-row retail sales) ──────────────────────
np.random.seed(42)
n = 200
regions   = ["North", "South", "East", "West"]
products  = ["Laptop", "Monitor", "Keyboard", "Mouse", "Headset", "Webcam", "Tablet"]

df = pd.DataFrame({
    "date":     pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    "region":   np.random.choice(regions, n),
    "product":  np.random.choice(products, n),
    "sales":    np.round(np.random.uniform(200, 5000, n), 2),
    "units":    np.random.randint(1, 50, n),
    "discount": np.round(np.random.uniform(0, 0.3, n), 2),
    "cost":     np.round(np.random.uniform(100, 2500, n), 2),
    "customer_id": [f"C{np.random.randint(1000, 9999)}" for _ in range(n)],
    "target":   np.where(np.random.uniform(0, 1, n) > 0.35, 1, 0),  # binary churn flag
})

# ── 2. CSV ──────────────────────────────────────────────────────────────────
csv_path = SAMPLE_DIR / "sample_sales.csv"
df.to_csv(csv_path, index=False)
print(f"✓ CSV      → {csv_path}")

# ── 3. Excel ────────────────────────────────────────────────────────────────
xlsx_path = SAMPLE_DIR / "sample_sales.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Sales", index=False)
    # add a second sheet with a simple summary
    summary = df.groupby("region")["sales"].agg(["sum", "mean", "count"]).reset_index()
    summary.columns = ["region", "total_sales", "avg_sales", "transactions"]
    summary.to_excel(writer, sheet_name="Regional Summary", index=False)
print(f"✓ Excel    → {xlsx_path}")

# ── 4. JSON ─────────────────────────────────────────────────────────────────
json_path = SAMPLE_DIR / "sample_sales.json"
json_path.write_text(df.to_json(orient="records", indent=2))
print(f"✓ JSON     → {json_path}")

# ── 5. Parquet ───────────────────────────────────────────────────────────────
parquet_path = SAMPLE_DIR / "sample_sales.parquet"
df.to_parquet(parquet_path, index=False)
print(f"✓ Parquet  → {parquet_path}")

# ── 6. PostgreSQL seed SQL ───────────────────────────────────────────────────
sql_path = SAMPLE_DIR / "seed_postgres.sql"
rows = []
for _, r in df.iterrows():
    rows.append(
        f"('{r.date}','{r.region}','{r.product}',{r.sales},{r.units},{r.discount},{r.cost},'{r.customer_id}',{r.target})"
    )
sql_content = f"""-- DIPEX Sample Data — PostgreSQL seed
-- Run: psql -U admin -d dipex -f samples/seed_postgres.sql

DROP TABLE IF EXISTS sales;
CREATE TABLE sales (
    date        VARCHAR(10),
    region      VARCHAR(20),
    product     VARCHAR(30),
    sales       NUMERIC(10,2),
    units       INT,
    discount    NUMERIC(5,2),
    cost        NUMERIC(10,2),
    customer_id VARCHAR(10),
    target      INT
);

INSERT INTO sales VALUES
{chr(44).join(rows)};

-- Verify
SELECT region, COUNT(*) as rows, ROUND(AVG(sales)::numeric,2) as avg_sales
FROM sales GROUP BY region ORDER BY region;
"""
sql_path.write_text(sql_content)
print(f"✓ Postgres → {sql_path}")

# ── 7. MongoDB seed (JSON docs) ──────────────────────────────────────────────
mongo_path = SAMPLE_DIR / "seed_mongodb.json"
docs = df.to_dict(orient="records")
mongo_path.write_text(json.dumps(docs, indent=2))

mongo_script = SAMPLE_DIR / "seed_mongodb.py"
mongo_script.write_text("""\"\"\"
Seed MongoDB with sample sales data.
Run:  python samples/seed_mongodb.py
Requires:  pip install pymongo  + MongoDB running (docker compose up -d mongodb)
\"\"\"
import json, os
from pathlib import Path

try:
    from pymongo import MongoClient
    uri  = os.getenv("MONGO_URI", "mongodb://admin:supersecret@localhost:27017/dipex?authSource=admin")
    cli  = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db   = cli["dipex"]
    col  = db["sales"]
    col.drop()
    docs = json.loads(Path("samples/seed_mongodb.json").read_text())
    result = col.insert_many(docs)
    print(f"✓ MongoDB  → inserted {len(result.inserted_ids)} documents into dipex.sales")
    cli.close()
except Exception as e:
    print(f"✕ MongoDB  seed failed: {e}")
""")
print(f"✓ MongoDB  → {mongo_path}  |  seed script: {mongo_script}")

# ── 8. Redis seed script ─────────────────────────────────────────────────────
redis_script = SAMPLE_DIR / "seed_redis.py"
redis_script.write_text("""\"\"\"
Seed Redis with sample sales data (stored as JSON strings per customer).
Run:  python samples/seed_redis.py
Requires:  pip install redis  + Redis running (docker compose up -d redis)
\"\"\"
import json, os
from pathlib import Path

try:
    import redis
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD", "dipexredis"),
        decode_responses=True,
    )
    r.ping()
    docs = json.loads(Path("samples/seed_mongodb.json").read_text())  # reuse same data
    pipe = r.pipeline()
    for i, doc in enumerate(docs):
        key = f"dipex:sale:{i}"
        pipe.set(key, json.dumps(doc))
    pipe.set("dipex:sales:count", len(docs))
    pipe.execute()
    print(f"✓ Redis    → wrote {len(docs)} keys (dipex:sale:0 … dipex:sale:{len(docs)-1})")
except Exception as e:
    print(f"✕ Redis    seed failed: {e}")
""")
print(f"✓ Redis    → {redis_script}")

# ── 9. Neo4j seed (Cypher) ─────────────────────────────────────────────────
neo4j_script = SAMPLE_DIR / "seed_neo4j.py"
neo4j_script.write_text("""\"\"\"
Seed Neo4j with a product-region-customer graph.
Run:  python samples/seed_neo4j.py
Requires:  pip install neo4j  + Neo4j running (docker compose up -d neo4j)
\"\"\"
import json, os
from pathlib import Path

try:
    from neo4j import GraphDatabase
    uri  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd  = os.getenv("NEO4J_PASS", "supersecret")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    docs   = json.loads(Path("samples/seed_mongodb.json").read_text())

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")   # clear
        for doc in docs:
            session.run(
                \"\"\"
                MERGE (r:Region {name: $region})
                MERGE (p:Product {name: $product})
                MERGE (c:Customer {id: $cust})
                CREATE (t:Transaction {date: $date, sales: $sales, units: $units, target: $target})
                MERGE (c)-[:BOUGHT]->(p)
                MERGE (t)-[:IN_REGION]->(r)
                MERGE (c)-[:MADE]->(t)
                \"\"\",
                region=doc["region"], product=doc["product"],
                cust=doc["customer_id"], date=doc["date"],
                sales=doc["sales"], units=doc["units"], target=doc["target"],
            )
    print(f"✓ Neo4j    → seeded {len(docs)} transactions, {len(set(d['customer_id'] for d in docs))} customers, {len(set(d['product'] for d in docs))} products")
    driver.close()
except Exception as e:
    print(f"✕ Neo4j    seed failed: {e}")
""")
print(f"✓ Neo4j    → {neo4j_script}")

# ── 10. DuckDB seed ────────────────────────────────────────────────────────
duckdb_script = SAMPLE_DIR / "seed_duckdb.py"
duckdb_script.write_text("""\"\"\"
Seed DuckDB with the sample sales table.
Run:  python samples/seed_duckdb.py
Requires:  pip install duckdb
\"\"\"
import os
from pathlib import Path

try:
    import duckdb, pandas as pd
    db_path = os.getenv("DUCKDB_PATH", "data/dipex.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    df  = pd.read_csv("samples/sample_sales.csv")
    con.execute("DROP TABLE IF EXISTS sales")
    con.execute("CREATE TABLE sales AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    print(f"✓ DuckDB   → {count} rows written to table 'sales' in {db_path}")
    # Quick sanity query
    result = con.execute("SELECT region, COUNT(*) as n, ROUND(AVG(sales),2) as avg_sales FROM sales GROUP BY region ORDER BY region").fetchall()
    print("  Region summary:")
    for row in result:
        print(f"    {row[0]:8s}  n={row[1]:3d}  avg_sales={row[2]}")
    con.close()
except Exception as e:
    print(f"✕ DuckDB   seed failed: {e}")
""")
print(f"✓ DuckDB   → {duckdb_script}")

print("\n✅ All sample data generated in  samples/")
print("   File-based sources are ready immediately.")
print("   Database sources need Docker running — then run seeds:")
print("     python samples/seed_mongodb.py")
print("     python samples/seed_redis.py")
print("     python samples/seed_neo4j.py")
print("     python samples/seed_duckdb.py")
print("     psql -U admin -d dipex -f samples/seed_postgres.sql")
