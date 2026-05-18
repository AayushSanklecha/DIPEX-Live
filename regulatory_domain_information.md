# DIPEX Regulatory Domain Architecture & Compliance Engine

The DIPEX analytical pipeline incorporates an extremely robust, multi-layer **Regulatory Compliance Engine** (located in `validation/regulatory/`). This engine mathematically enforces legal bounds, financial constraints, and privacy mandates dynamically based on the dataset's business domain (e.g., Banking, Healthcare, Finance).

This document details the architectural execution flow, the dynamic conflict resolution logic, and the mechanics of the deep domain rules.

---

## ⚙️ Architectural Execution Flow

The engine operates as a "Hard Gate" (Gate 1) in the primary pipeline. Any violations flagged as `CRITICAL` or `ERROR` will inherently block the dataset from proceeding to Machine Learning stages unless a remediation protocol is executed.

### Auto-Detection & Multi-Domain Execution
Instead of rigidly assigning a single domain to a dataset, the `RegulatoryEngine` natively supports **Multi-Domain processing**.
1. **Detection:** The engine utilizes the `from_config_auto()` entry point to semantically parse the data schema using an NLP auto-detector. 
2. **Multi-Domain Mapping:** If a dataset matches multiple profiles (e.g., a hospital's payment ledger matching both `Healthcare` and `Banking`), the engine dynamically injects rules from **both** domains simultaneously.
3. **Fault Tolerance:** Every rule evaluates independently within isolated execution blocks. If a specialized rule (e.g., `BaselIIILeverageRule`) crashes due to malformed data types, the engine cleanly logs the trace and continues evaluating the remaining 50+ rules without silently collapsing.

---

## ⚔️ Rule Conflict Resolution (`conflict_resolver.py`)

Because datasets can be processed under multiple domains simultaneously, it is entirely possible for rules to organically contradict one another on the same column. 

*Example: A Finance rule might flag a column as `ERROR` because it's missing double-entry balancing, but a GDPR privacy rule might flag it as `CRITICAL` because it contains unmasked PII.*

To handle this, the `RuleConflictResolver` computes outcomes using three available state strategies:

1. **`strictest_wins` (Default & Safest):** 
   - Hierarchy: `CRITICAL` > `ERROR` > `WARNING` > `PASS`. 
   - If two rules contradict on a column, the pipeline enforces the most severe verdict and explicitly suppresses the lesser violation, logging a `[ConflictResolver]` trace.
2. **`domain_priority`:** 
   - The primary configured domain (e.g., "Healthcare") takes absolute precedence. Conflicting rules from secondary domains (like "Banking") are immediately downgraded from `CRITICAL/ERROR` into a non-blocking `WARNING` advisory.
3. **`advisory_only`:** 
   - No resolution is applied. Both rules' verdicts are retained, but the system appends a programmatic `RULE_CONFLICT` meta-violation to alert human data engineers.

---

## 🏛️ Domain 1: BANKING (`banking_rules.py`)

The Banking regulatory engine strictly enforces fiscal and Anti-Money Laundering (AML) controls on transactional datasets. 

- **Rule: `SuspiciousTransactionPatternRule` (Velocity Spike):** Monitors identical `transaction_id_column` structures across sliding time frames. Highly correlated to FATF Recommendation 20 for algorithmic AML spotting.
- **Rule: `BaselIIILeverageRule` / `LCRRule` / `NSFRRule`:** Native Basel III risk framework bounding. It mathematically audits `tier1_capital` versus `exposure_col` to guarantee minimum leverage bounds mathematically (> 0.03).
- **Rule: `bank_positive_amount`:** Prevents negative anomalies in specific numeric columns where absolutely no negative signs can exist mathematically (e.g., `loan_amount`, `fee_amount`).
- **Rule: `bank_aml_10k`:** Enforces raw AML thresholds, warning on isolated transactions > $10,000.00.
- **Rule: `bank_ltv` (Loan-To-Value):** Bounds structural ratios between `loan_amount` and `collateral_value` (rejecting any rows with LTV > 0.90) to prevent mathematically impossible mortgage issuances from hitting ML stages.

---

## 📈 Domain 2: FINANCE (`finance_rules.py`)

The Finance module evaluates ledger datasets to ensure compliance with SEC filings and Generally Accepted Accounting Principles (GAAP).

- **Rule: `fin_double_entry`:** Requires absolute zero-sum integrity. The engine groups identically tracked `transaction_id` rows and ensures the summation of `entry_amount` equals exactly `0.00` (within a configurable float tolerance of e.g., `0.01`).
- **Rule: `fin_capital_adequacy`:** A CRITICAL rule strictly enforcing a minimum Capital Adequacy Ratio (CAR, e.g., > 0.08) between `tier1_capital` and `risk_weighted_assets`. 
- **Rule: `fin_margin_call`:** Triggers automatic reporting alerts when a `margin_balance` drops below its dynamic `maintenance_margin`.
- **Rule: `fin_fair_value`:** Bounds the allowable ratio (e.g., 20%) of Level 3 illiquid assets compared to total fair value structures.

---

## ⚕️ Domain 3: HEALTHCARE (`healthcare_rules.py`)

The Healthcare engine enforces HIPAA compliance, PII redaction maps, and deep biological constraints.

- **Rule: `hc_phi_access` (PHI Presence):** Evaluates text strings via `Regex` and structure mapping to explicitly reject unmasked Protected Health Information (PHI) like `ssn`, `insurance_id`, or `phone`.
- **Rule: `ConsentValidationRule`:** Automatically traverses `phi_columns` and guarantees that an associated boolean `consent_column` explicitly tracks permission, rejecting unconsented rows.
- **Rule: `hc_age_bounds`:** Deep biological sanity bounding—ensures `patient_age` dimensions mathematically never fall below 0.0 or logically exceed 125.0 years.
- **Rule: `DiagnosisCodeFormatRule`:** Strictly enforces regex bounds to ensure diagnosis strings structurally match ICD-10 or ICD-11 algorithmic formatting.

---

## 🌐 Secondary / Specialized Regulatory Frameworks

The system dynamically pulls in highly specialized global legal rulesets whenever relevant cross-domain signatures are detected:

- **E-Commerce (`ecommerce_rules.py`):** 
  - Validates `ChargebackThresholdRule` metrics and `RefundRightWindowRule` logic to ensure vendor bounds aren't violated mathematically.
- **Data Privacy (`ccpa_rules.py` / GDPR):** 
  - Validates California Consumer Privacy Act bounds, enforcing `CaliforniaResidencyDisclosureRule` and checking global privacy opt-out boolean fields.
  - Implements massive cross-domain GDPR logic (`GDPRDataResidencyRule`) ensuring region columns lock into `EU` or `EEA` regions exclusively.
- **Cybersecurity (`cyber_rules.py`):** 
  - Validates `EncryptionAtRestRule` limits, auditing datasets to guarantee boolean encryption flags correlate cleanly against sensitive fields.
- **ESG Frameworks (`esg_rules.py`):** 
  - Runs corporate sustainability checks including `CarbonEmissionsValidityRule`.
- **International Commerce (`fatf_rules.py` / `mifid2_rules.py` / `pci_dss_rules.py`):**
  - **FATF:** Implements `StructuringDetectionRule` (smurfing identification) and `PEPScreeningRule` for global targets.
  - **MiFID 2:** Maps `AlgoTradingTagRule` and `LEIISINFormatRule` for explicit European financial routing.
  - **PCI-DSS:** CRITICAL redaction checks mapping the mathematical structures of PANs (Primary Account Numbers) and structurally declining unencrypted CVV storage strings (`CVVStorageRule`).
