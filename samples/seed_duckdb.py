"""
Seed DuckDB with the real Titanic dataset.
Run:  python samples/seed_duckdb.py
Needs: pip install duckdb
"""
import os
from pathlib import Path

try:
    import duckdb, pandas as pd
    db_path = os.getenv("DUCKDB_PATH","data/dipex.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    df  = pd.read_csv("samples/titanic.csv")
    con.execute("DROP TABLE IF EXISTS titanic")
    con.execute("CREATE TABLE titanic AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM titanic").fetchone()[0]
    print(f"✓ DuckDB   → {count} rows in table 'titanic' at {db_path}")
    rows = con.execute("""
        SELECT sex, pclass, COUNT(*) as n,
               ROUND(AVG(CAST(survived AS DOUBLE))*100,1) as survival_pct
        FROM titanic GROUP BY sex, pclass ORDER BY pclass, sex
    """).fetchall()
    print(f"  Survival by sex & class:")
    for row in rows:
        print(f"    {row[0]:7s} Class-{row[1]}  n={row[2]:3d}  survival={row[3]}%")
    con.close()
except Exception as e:
    print(f"✕ DuckDB seed failed: {e}")
