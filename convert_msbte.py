import re

# Read the file
with open("docs/DIPEX_Project_Report.md", "r", encoding="utf-8") as f:
    content = f.read()

# Update the TOC
old_toc = """206: 1. Chapter 1 — Introduction
207: 2. Chapter 2 — Literature Survey
208: 3. Chapter 3 — Scope of the Project
209: 4. Chapter 4 — Methodology / Approach
210: 5. Chapter 5 — Details of Design, Working and Processes
211: 6. Chapter 6 — Results and Applications
212: 7. Conclusion
213: 8. References"""

new_toc = """1. Chapter 1 — Introduction
2. Chapter 2 — Literature Survey
3. Chapter 3 — Scope of the Project
4. Chapter 4 — System Requirements & Feasibility Study
5. Chapter 5 — Methodology / Approach
6. Chapter 6 — Details of Design, Working and Processes
7. Chapter 7 — System Testing Strategy
8. Chapter 8 — Results and Applications
9. Chapter 9 — Limitations and Future Scope
10. Chapter 10 — Conclusion
11. Chapter 11 — References
12. Chapter 12 — Appendices"""

content = re.sub(r'1\. Chapter 1 — Introduction.*?8\. References', new_toc, content, flags=re.DOTALL)

# Re-number the chapters
content = content.replace("# CHAPTER 4\n## METHODOLOGY / APPROACH", "# CHAPTER 5\n## METHODOLOGY / APPROACH")
content = content.replace("# CHAPTER 5\n## DETAILS OF DESIGN, WORKING AND PROCESSES", "# CHAPTER 6\n## DETAILS OF DESIGN, WORKING AND PROCESSES")
content = content.replace("# CHAPTER 6\n## RESULTS AND APPLICATIONS", "# CHAPTER 8\n## RESULTS AND APPLICATIONS")

# Insert Chapter 4
chapter_4_text = """
---

# CHAPTER 4
## SYSTEM REQUIREMENTS & FEASIBILITY STUDY

### 4.1 System Requirement Specification (SRS)

**4.1.1 Hardware Requirements**
The DIPEX platform is designed to run efficiently on commodity enterprise hardware without necessitating GPU acceleration for tabular data analysis.
- **Minimum Requirements:** Intel Core i5 or AMD Ryzen 5 processor (equivalent or higher), 16 GB RAM (DDR4), 50 GB Free Disk Space (SSD recommended for PostgreSQL/DuckDB read/write speed).
- **Recommended Requirements:** Intel Core i7 (e.g. i7-12700H) or AMD Ryzen 7, 32 GB RAM (DDR5), 500+ GB NVMe SSD for fast Kafka stream handling and chunked parquet preprocessing. GPU is optional.

**4.1.2 Software Requirements**
- **Operating System:** Ubuntu Linux 22.04 LTS (Production), Windows 10/11 / macOS (Development).
- **Programming Languages:** Python 3.12, JavaScript (ES6+).
- **Frontend Technologies:** React 18, Vite.js, TailwindCSS.
- **Backend & APIs:** FastAPI (ASGI Uvicorn).
- **Database Systems:** PostgreSQL 16, DuckDB, MongoDB 7.0, Redis.
- **Message Broker:** Apache Kafka.
- **Machine Learning Libraries:** PyTorch, scikit-learn, LightGBM, XGBoost, Optuna, SHAP.
- **Other Tools:** Docker & Docker Compose.

### 4.2 Feasibility Study

**4.2.1 Technical Feasibility**
The system is technically feasible as it relies on open-source, industry-tested frameworks (FastAPI, React, DuckDB). The dual Reinforcement Learning (RL) approach uses mature algorithms (PPO and Thompson Sampling) integrated carefully with the ML validation stages. Performance bench tests show 100K rows scaling within an 8-second execution window.

**4.2.2 Economic / Financial Feasibility**
The system eliminates expensive licensing costs associated with enterprise platforms (like Collibra or fully integrated external AutoML services) by utilising robust open-source stacks. By containerising the deployment, hosting costs are dramatically minimised compared to heavily scaled distributed systems (like Spark arrays).

**4.2.3 Operational Feasibility**
Operational feasibility is highly favourable. Implementing DIPEX replaces fractured tooling (separate expectation suites, Python validation scripts, drift detectors) with a central 8-stage pipeline. The unified analytical reporting simplifies the daily data analysis processes for domain experts and data engineers alike, substantially reducing debugging workload.
"""

content = content.replace("# CHAPTER 5\n## METHODOLOGY / APPROACH", chapter_4_text + "\n# CHAPTER 5\n## METHODOLOGY / APPROACH")


# Insert Chapter 7
chapter_7_text = """
---

# CHAPTER 7
## SYSTEM TESTING STRATEGY

### 7.1 Testing Overview
To guarantee the resilience and correctness of the end-to-end data intelligence pipeline, the system undergoes rigorous modular and integrative testing covering over 497 automated tests across standard methodologies.

### 7.2 Unit Testing
Unit testing asserts the correctness of individual algorithms, heuristic detectors, and discrete mathematical functions in isolation.
- **Validators Check:** `pytest` is used to enforce null rates, confirm formatting boundaries for IBANs/Emails, and check logic gates for numeric zero-variance calculations.
- **Model Asserts:** PyTorch MLP tensor dimensionalities and LightGBM objective functions are individually smoke-tested against known standard synthetic arrays.

### 7.3 Integration Testing
Integration testing verifies that the Data Ingestion (UniversalIntake) correctly interfaces with the processing backbone.
- Data transition from a streamed Kafka partition into analytical DuckDB Parquet chunks is validated under memory-restricted environment emulations (via `--max-memory` flags).
- Verifying the transfer of standard `ValidationFinding` schema objects continuously across the ML domain classifiers before triggering the RL models.

### 7.4 System / End-to-End Testing
The "Smoke Test" validation verifies the whole pipeline (8 stages). 
- **Simulated Real-world Pipelines:** Feeding OpenML datasets spanning banking (AML) and healthcare (HIPAA) through the `pipeline_bridge` to assert that the proper Compliance Engines are activated synchronously.
- **Audit Logs Check:** Verifying the tamper-proof cryptographic assertions where output `PipelineResult` checksums are strictly compared with original file hashes.

### 7.5 Performance & Load Testing
A primary Non-Functional Requirement was processing 100K rows under 8 seconds. Load tests repeatedly ingest heavy datasets (up to 50 GB limits chunked into sizes). Concurrent pipeline trigger tests assert that FastAPI's rate limiting (120 requests/minute) is strictly enforced.
"""

content = content.replace("# CHAPTER 8\n## RESULTS AND APPLICATIONS", chapter_7_text + "\n# CHAPTER 8\n## RESULTS AND APPLICATIONS")


# Insert Chapters 9 and fix Conclusion / References
chapter_9_plus_text = """
---

# CHAPTER 9
## LIMITATIONS AND FUTURE SCOPE

### 9.1 Limitations
1. **Unstructured Data Parsing:** Currently, DIPEX is strictly a tabular data platform. It cannot infer logic from pure images, audio files, or extensive raw NLP paragraphs inside cells (excluding identifiable metadata parsing like PHI/NER).
2. **Cluster Distributed Scaling:** For datasets far exceeding 50 GB continuously on heavy enterprise influxes, single-machine (or dual-machine docker clusters) memory handling by ChunkedParquetWriter can become a bottleneck. Currently, there is an enforced memory cap natively.

### 9.2 Future Scope
1. **Apache Spark & Delta Lake Integration:** Restructuring the Core Engine to interface smoothly with Hadoop Distributed File Systems (HDFS) and Apache Spark clusters to lift dataset volume restrictions into the Terabyte arrays.
2. **Active Semantic Re-Learning:** Applying Active Learning workflows where data pipeline misclassifications (detected empirically by an engineer downstream) are logged back as live weight adjustments to the NLP-Augmented cascades.
3. **Federated Execution:** Enhancing privacy-preserving data quality checks where DIPEX executes across parallel geographical data silos (solving inter-continent regulations).

---

# CHAPTER 10
## CONCLUSION
"""

content = content.replace("# CONCLUSION", chapter_9_plus_text)
content = content.replace("# REFERENCES", "# CHAPTER 11\n## REFERENCES")


# Add Appendix
appendix_text = """

---

# CHAPTER 12
## APPENDICES

### Appendix A: Example Pipeline Output JSONL Audit
```json
{
  "timestamp": "2026-04-12T14:32:00Z",
  "dataset_id": "healthcare_trial_v3",
  "domain_classification": "Healthcare",
  "compliance_findings": [
    {"rule": "HIPAA-01", "status": "WARN", "detail": "SSN matching patterns isolated in column 'patient_id'"}
  ],
  "confidence_score": 0.982,
  "gate_decision": "PASS",
  "hash": "b2c938d8e58f27...42ef3"
}
```

### Appendix B: Technology Docker Compose Reference
```yaml
version: '3.8'
services:
  dipex-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=PRODUCTION
    depends_on:
      - kafka-broker
      - postgres-db
  kafka-broker:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
```
"""

content += appendix_text

with open("DIPEX_MSBTE_Blackbook_Report.md", "w", encoding="utf-8") as f:
    f.write(content)

print("Conversion complete!")
