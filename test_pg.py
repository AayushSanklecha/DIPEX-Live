import psycopg2
conn = psycopg2.connect(host="localhost", port=5433, dbname="dipex", user="dipex", password="dipex_secret")
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
print("Tables:", [r[0] for r in cur.fetchall()])
cur.execute("SELECT COUNT(*) FROM hackathon_users")
print("Users count:", cur.fetchone()[0])
conn.close()
