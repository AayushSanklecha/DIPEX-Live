"""
tests/test_governance_engine.py
---------------------------------
Verifies the new Data Governance module (PIIDetector and DataGovernor).
"""

import pandas as pd
import pytest

from validation.governance.pii_detector import PIIDetector
from validation.governance.governor import DataGovernor, GovernanceError

@pytest.fixture
def messy_pii_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "email": [
            "alice@example.com",
            "bob.smith@corp.co.uk",
            "no-email-here",
            "diana@secure.net"
        ],
        "ssn_or_notes": [
            "Good customer",
            "SSN is 123-45-6789 do not lose",
            "Another SSN: 987-65-4321",
            "No notes"
        ],
        "phone_contact": [
            "(555) 123-4567",
            "nothing here",
            "+1-800-555-0199",
            "555.999.8888"
        ],
        "cc_data": [
            "1234-5678-9012-3456",
            "My card is 4111222233334444",
            "No card",
            "N/A"
        ]
    })


def test_pii_detector_detect(messy_pii_df):
    detector = PIIDetector()
    report = detector.detect(messy_pii_df)
    
    assert "email" in report
    assert report["email"]["Email"] == 3
    
    assert "ssn_or_notes" in report
    assert report["ssn_or_notes"]["SSN"] == 2
    
    assert "phone_contact" in report
    assert report["phone_contact"]["Phone"] == 3
    
    assert "cc_data" in report
    assert report["cc_data"]["CreditCard"] == 2


def test_pii_detector_redact(messy_pii_df):
    detector = PIIDetector()
    cleansed, report = detector.redact(messy_pii_df)
    
    # 1. Emails should be gone
    assert "alice@example.com" not in cleansed["email"].values
    assert "[REDACTED_EMAIL]" in cleansed["email"].values
    assert cleansed["email"].iloc[2] == "no-email-here" # shouldn't touch valid non-pii
    
    # 2. SSNs should be masked inline
    assert cleansed["ssn_or_notes"].iloc[1] == "SSN is [REDACTED_SSN] do not lose"
    assert cleansed["ssn_or_notes"].iloc[2] == "Another SSN: [REDACTED_SSN]"
    
    # 3. Phone and CC should be masked
    assert cleansed["phone_contact"].iloc[0] == "[REDACTED_PHONE]"
    assert cleansed["cc_data"].iloc[0] == "[REDACTED_CC]"
    assert cleansed["cc_data"].iloc[1] == "My card is [REDACTED_CC]"


def test_governor_redact_policy(messy_pii_df):
    config = {
        "validation": {
            "governance": {
                "pii_detection": True,
                "policy": "redact"
            }
        }
    }
    governor = DataGovernor(config)
    cleansed, report = governor.enforce(messy_pii_df, "test_dataset")
    
    assert report["status"] == "redacted"
    assert report["total_redactions"] == 11  # 3 email + 2 ssn + 4 phone/cc
    
    assert cleansed["email"].iloc[0] == "[REDACTED_EMAIL]"


def test_governor_reject_policy(messy_pii_df):
    config = {
        "validation": {
            "governance": {
                "pii_detection": True,
                "policy": "reject"
            }
        }
    }
    governor = DataGovernor(config)
    
    with pytest.raises(GovernanceError) as exc_info:
        governor.enforce(messy_pii_df, "test_dataset")
        
    assert "REJECT" in str(exc_info.value)
    assert "11 instances of PII" in str(exc_info.value)


def test_governor_flag_policy(messy_pii_df):
    config = {
        "validation": {
            "governance": {
                "pii_detection": True,
                "policy": "flag"
            }
        }
    }
    governor = DataGovernor(config)
    cleansed, report = governor.enforce(messy_pii_df, "test_dataset")
    
    assert report["status"] == "flagged"
    # DataFrame shouldn't be touched
    assert cleansed["email"].iloc[0] == "alice@example.com"


def test_governor_inactive(messy_pii_df):
    config = {
        "validation": {
            "governance": {
                "pii_detection": False,
                "policy": "redact"
            }
        }
    }
    governor = DataGovernor(config)
    cleansed, report = governor.enforce(messy_pii_df, "test_dataset")
    
    assert report["status"] == "skipped"
    assert cleansed["email"].iloc[0] == "alice@example.com"
