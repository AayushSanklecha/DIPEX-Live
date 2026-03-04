"""
colab/train_pii_ner.py
------------------------
DIPEX — Google Colab Training Script: PII Named Entity Recognition Model
Run this on Google Colab (GPU recommended for faster spaCy training, CPU works).

Outputs
-------
  models/pii_ner_model/  — serialised spaCy NLP model directory

Instructions
------------
1. If you have labelled PII data (JSONL with entities), set PII_JSONL.
2. Otherwise this script uses synthetic PII + non-PII examples to fine-tune
   the blank spaCy model for NER.
3. Run all cells. Training takes ~5 minutes on Colab CPU.
4. Download the models/pii_ner_model/ folder → place in your project's models/.
   PIIDetector will auto-detect and load it.

Note: If you prefer the regex-only mode or en_core_web_sm, just don't place
the custom model in models/ — PIIDetector has a 3-tier fallback.
"""

# ─── Install ──────────────────────────────────────────────────────────────────
# !pip install spacy -q
# !python -m spacy download en_core_web_sm -q

import os
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

print("=== DIPEX PII NER Training ===")

# ─── Config ───────────────────────────────────────────────────────────────────

PII_JSONL  = None    # Path to custom JSONL training data or None for synthetic
N_ITER     = 30
OUTPUT_DIR = "models/pii_ner_model"

# ─── Step 1: Synthetic PII training examples ──────────────────────────────────

_PII_EXAMPLES_RAW = [
    # (text, [(start, end, label)])
    ("Call John Smith at 555-123-4567", [(5, 15, "PERSON"), (19, 31, "PHONE")]),
    ("Email alice@example.com for details", [(6, 23, "EMAIL")]),
    ("Send to Bob Johnson, CEO of Acme Corp", [(8, 19, "PERSON"), (28, 37, "ORG")]),
    ("DOB: 15/03/1990 for patient ID 87654321", [(5, 15, "DATE")]),
    ("Card: 4111 1111 1111 1111 exp 12/25", [(6, 25, "CREDIT_CARD")]),
    ("SSN 123-45-6789 required", [(4, 15, "SSN")]),
    ("IP address 192.168.1.1 detected", [(11, 22, "IP_ADDRESS")]),
    ("The CEO of Google is Sundar Pichai", [(22, 35, "PERSON"), (11, 17, "ORG")]),
    ("Patient born on 1985-07-22", [(16, 26, "DATE")]),
    ("Customer: Jane Doe, ZIP 94103", [(10, 18, "PERSON"), (24, 29, "ZIPCODE")]),
    ("Revenue for Q3 is $450,000", []),  # non-PII
    ("Total orders placed: 1,234", []),  # non-PII
    ("The average score was 87.5 points", []),  # non-PII
    ("Monthly trend shows 12% growth", []),  # non-PII
]

# Generate more examples by varying templates
_NAMES = ["Alice Brown", "Robert Chen", "María García", "David Lee", "Priya Sharma"]
_EMAILS = ["test@corp.com", "user@email.net", "info@business.org"]
_NON_PII = [
    "Total revenue: {}", "Average order value: {}", "Monthly growth: {}%",
    "Shipment count: {}", "Return rate: {}", "Satisfaction score: {}",
]

TRAIN_DATA = list(_PII_EXAMPLES_RAW)

# Add name variations
for name in _NAMES:
    TRAIN_DATA.append((f"Customer: {name}", [(10, 10 + len(name), "PERSON")]))
    TRAIN_DATA.append((f"User {name} logged in", [(5, 5 + len(name), "PERSON")]))

# Add email variations
for email in _EMAILS:
    TRAIN_DATA.append((f"Send confirmation to {email}", [(20, 20 + len(email), "EMAIL")]))

# Add non-PII negatives
import random as _random
for tmpl in _NON_PII:
    for val in [1234, 567, 89.5, 42, 10_000]:
        TRAIN_DATA.append((tmpl.format(val), []))

_random.shuffle(TRAIN_DATA)
print(f"Training examples: {len(TRAIN_DATA)}")

# ─── Step 2: Build spaCy NLP + NER ───────────────────────────────────────────

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner", last=True)

ALL_LABELS = {"PERSON", "PHONE", "EMAIL", "ORG", "DATE", "CREDIT_CARD", "SSN", "IP_ADDRESS", "ZIPCODE"}
for lbl in ALL_LABELS:
    ner.add_label(lbl)

# ─── Step 3: Convert to spaCy Example format ──────────────────────────────────

examples = []
for text, entities in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            ents.append(span)
    ann = {"entities": [(e.start_char, e.end_char, e.label_) for e in ents]}
    examples.append(Example.from_dict(doc, ann))

print(f"Valid training examples: {len(examples)}")

# ─── Step 4: Train ───────────────────────────────────────────────────────────

optimizer = nlp.begin_training()
losses_history = []

for i in range(N_ITER):
    _random.shuffle(examples)
    losses = {}
    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, sgd=optimizer, drop=0.35, losses=losses)
    losses_history.append(losses.get("ner", 0.0))
    if (i + 1) % 5 == 0:
        print(f"  Iteration {i+1:3d} — NER loss: {losses.get('ner', 0):.4f}")

print(f"\nFinal NER loss: {losses_history[-1]:.4f}")

# ─── Step 5: Quick sanity check ──────────────────────────────────────────────

test_texts = [
    "Call Michael Scott at michael@dundermifflin.com or 570-555-0100",
    "Revenue increased by 15% in Q4",
    "Patient ID 12345 born on 1990-05-10",
]
print("\nSanity check on test texts:")
for t in test_texts:
    doc = nlp(t)
    print(f"  '{t}' → {[(e.text, e.label_) for e in doc.ents]}")

# ─── Step 6: Save model ───────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
nlp.to_disk(OUTPUT_DIR)
print(f"\nModel saved to: {OUTPUT_DIR}/")
print(">> Download the entire pii_ner_model/ folder and place it in your project's models/ directory.")
print(">> PIIDetector will auto-detect and use it alongside regex matching.")
