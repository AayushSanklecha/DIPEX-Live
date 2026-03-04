"""
Seed Neo4j with Titanic graph: Passenger → Class, Passenger → Port (Embarked).
Run:  python samples/seed_neo4j.py
Needs: pip install neo4j  +  docker compose up -d neo4j  (give it ~60s to start)
"""
import os
import pandas as pd

try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI","bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASS","supersecret"))
    )
    df = pd.read_csv("samples/titanic.csv").fillna("Unknown").head(150)
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        for _, r in df.iterrows():
            s.run("""
                MERGE (cls:Class {name: $pclass})
                MERGE (port:Port {name: $embarked})
                CREATE (p:Passenger {name:$name, sex:$sex, age:$age, fare:$fare, survived:$survived})
                MERGE (p)-[:TRAVELLED_IN]->(cls)
                MERGE (p)-[:EMBARKED_AT]->(port)
            """, pclass=f"Class {int(r.Pclass)}" if r.Pclass!="Unknown" else "Unknown",
                 embarked=str(r.Embarked), name=str(r.Name),
                 sex=str(r.Sex), age=float(r.Age) if r.Age!="Unknown" else -1,
                 fare=float(r.Fare) if r.Fare!="Unknown" else 0,
                 survived=int(r.Survived) if r.Survived!="Unknown" else -1)
    classes = len(df["Pclass"].unique())
    ports   = len(df["Embarked"].unique())
    print(f"✓ Neo4j    → {len(df)} Passenger nodes, {classes} Class nodes, {ports} Port nodes")
    print(f"  Graph: Passenger -[TRAVELLED_IN]-> Class, Passenger -[EMBARKED_AT]-> Port")
    driver.close()
except Exception as e:
    print(f"✕ Neo4j seed failed: {e}")
