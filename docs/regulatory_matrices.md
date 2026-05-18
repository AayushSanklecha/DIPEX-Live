# ADAP Platform - Regulatory Compliance Matrices

The ADAP Platform includes a comprehensive deterministic validation engine designed to scan and validate records against up to 18 regulatory domains.

## 1. Supported Regulatory Domains

| Domain | Scope | Description |
|---|---|---|
| **Banking** | Anti-Money Laundering (AML), Basel III, Credit Risk | Scans for suspicious transaction velocity, Loan-to-Value thresholds, Capital Adequacy ratios, and Liquidity Coverage Ratios (LCR). |
| **Finance** | SEC/FCA Reporting, SOX | Verifies double-entry accounting balances, revenue recognition boundaries, and fair value hierarchy (Level 3 asset limits). |
| **Healthcare** | HIPAA, Clinical Trials | Enforces protected health information (PHI) anonymization, diagnosis code formatting, vital sign physiological bounds, and patient consent indicators. |
| **GDPR** | Data Protection (EU) | Confirms data residency markers and explicit consent flags for European data subjects before analysis. |
| **CCPA** | Data Privacy (California) | Checks for "Do Not Sell" flags and consumer rights deletion status before modeling. |
| **PCI-DSS** | Payment Card Industry | Detects plaintext PANs, CVV storage instances, and ensures cardholder data segregation. |
| **FATF** | Anti-Terrorist Financing | Identifies structuring (smurfing) transaction patterns, PEP (Politically Exposed Persons) screening anomalies, and sanctions flags. |
| **ESG** | Environmental, Social, Governance | Validates reported carbon emission ranges and ESG score bounds for green finance and sustainability reporting. |
| **MiFID II** | European Financial Markets | Verifies algorithmic trading tags, LEI/ISIN format validity, and Best Execution timestamps. |
| **DORA** | Digital Operational Resilience (EU) | Assesses disaster recovery markers, continuous backup verifications, and ICT risk profiles. |
| **Insurance** | Solvency II context | Validates Solvency Capital Requirement (SCR) adequacy, combined ratio bounds, and claim settlement timelines. |
| **Ecommerce** | Consumer Rights, Payments | Validates SCA (Strong Customer Authentication) exemptions, chargeback thresholds, and refund window rights. |

## 2. Rule Evaluation and Resolution

- **Multi-Domain Processing**: The engine can process datasets against multiple domains simultaneously (e.g., Banking + GDPR).
- **Conflict Resolution**: Cross-domain contradictions are resolved via the `RuleConflictResolver`. The engine defaults to `strictest_wins`.
- **Severity Levels**:
    - `CRITICAL`: Causes immediate pipeline blockage.
    - `ERROR`: Masks the violating column.
    - `WARNING`: Flags the issue in the executive report.
