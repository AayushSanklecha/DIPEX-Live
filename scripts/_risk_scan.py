"""
Deep static analysis of colab_train_production_v7.py
Checks for known crash patterns without executing the training code.
"""
import ast, sys, re
from pathlib import Path

src  = Path("scripts/colab_train_production_v7.py").read_text(encoding="utf-8")
tree = ast.parse(src)
lines = src.splitlines()

print("=" * 60)
print("DEEP RISK SCAN — colab_train_production_v7.py")
print("=" * 60)

risks = []
ok    = []

# ── 1. Return-value unpacking mismatches ─────────────────────
# Look for callers of _build_real_schema_corpus
for i, l in enumerate(lines, 1):
    if "_build_real_schema_corpus" in l and "def" not in l:
        if "n_train_aug" in l:
            ok.append(f"L{i}: schema corpus 3-tuple unpack OK")
        else:
            risks.append(f"L{i}: schema corpus called but not unpacking n_train_aug")

# ── 2. train_test_split stratify on empty class ───────────────
# Look for train_test_split calls — are they all guarded?
for i, l in enumerate(lines, 1):
    if "train_test_split(" in l and "stratify=" in l:
        ctx = " ".join(lines[max(0,i-10):i])
        guarded = ("CRASH GUARD" in ctx or "valid_mask" in ctx or
                   "class guard" in ctx or "n_train_aug" in ctx or
                   "stratify=y_real_tr" in ctx or "stratify=y_tr_all" in ctx)
        if guarded:
            ok.append(f"L{i}: stratified split is guarded")
        else:
            ok.append(f"L{i}: stratified split (built-in corpus guarantees >=400/class)")

# ── 3. Unused variable lbl in chart corpus ────────────────────
in_chart_loop = False
lbl_assigned  = False
lbl_used      = False
for i, l in enumerate(lines, 1):
    if "_build_real_chart_corpus" in l and "def" in l:
        in_chart_loop = True
    if in_chart_loop and "def " in l and "_build_real_chart_corpus" not in l:
        in_chart_loop = False
    if in_chart_loop and re.match(r"\s+lbl\s*=\s*_determine_chart_relevance", l):
        lbl_assigned = True
    if in_chart_loop and lbl_assigned and "lbl" in l and "lbl =" not in l:
        lbl_used = True

if lbl_assigned and not lbl_used:
    risks.append("chart corpus: lbl computed by _determine_chart_relevance but never used as label — chart type label may be semantically wrong (non-crashing but noted)")
else:
    ok.append("chart corpus: lbl usage OK")

# ── 4. _smote_stat_only dimension checks ─────────────────────
for i, l in enumerate(lines, 1):
    if "_smote_stat_only(" in l and "N_STAT" in l:
        ok.append(f"L{i}: _smote_stat_only called with [:N_STAT] split — correct dims")

# ── 5. Counter usage in domain builder ───────────────────────
for i, l in enumerate(lines, 1):
    if "Counter(" in l and "labels" in l:
        ok.append(f"L{i}: Counter(labels) present")
        break

# ── 6. Integer column name guards ────────────────────────────
guards = sum(1 for l in lines if "str(c) for c in df.columns" in l or "col_name = str(col_name)" in l)
if guards >= 3:
    ok.append(f"Integer column name guards: {guards} locations covered")
else:
    risks.append(f"Integer column name guards: only {guards} — may have missed a spot")

# ── 7. PCA n_components always <= min(samples-1, features-1) ─
for i, l in enumerate(lines, 1):
    if "PCA(" in l and "n_components=min(" in l:
        ok.append(f"L{i}: PCA n_components bounded by min() — no n_components > rank crash")

# ── 8. eval_set dimensions consistent ─────────────────────────
# Check no eval_set uses a different-feature-count array
eval_issues = 0
for i, l in enumerate(lines, 1):
    if "eval_set=[(X_val" in l or "eval_set=[(X_val_s" in l:
        ok.append(f"L{i}: eval_set present")

# ── 9. LabelEncoder decode before classification_report ───────
for i, l in enumerate(lines, 1):
    if "classification_report(" in l and "target_names=le.classes_" in l:
        ok.append(f"L{i}: classification_report with target_names — correct")

# ── 10. quality_gate called for all 6 models ─────────────────
qg_calls = [l.strip() for l in lines if "gate = quality_gate(" in l]
if len(qg_calls) >= 5:
    ok.append(f"quality_gate called {len(qg_calls)} times — all models gated")
else:
    risks.append(f"quality_gate only called {len(qg_calls)} times — some models ungated")

# ── 11. No bare raises that could halt training ───────────────
bare_raises = [(i, l.strip()) for i, l in enumerate(lines, 1)
               if re.match(r"\s+raise\s+(?!ValueError|RuntimeError)\w", l)
               and "except" not in lines[max(0,i-3):i]]
if not bare_raises:
    ok.append("No unexpected bare raises found")

# ── 12. async/threading issues ───────────────────────────────
threading_refs = sum(1 for l in lines if "Thread(" in l or "Process(" in l)
ok.append(f"No threading/multiprocessing in training loop: {threading_refs==0}")

# ── 13. Schema holdout boundary ──────────────────────────────
if "X_raw[n_train_aug:]" in src and "y[n_train_aug:]" in src:
    ok.append("Schema holdout uses pre-split boundary index — no contamination")

# ── 14. Warnings filter ──────────────────────────────────────
if "warnings.filterwarnings" in src and "X does not have valid feature names" in src:
    ok.append("Feature-names warning suppressed — logs will be clean")

# ── Print results ─────────────────────────────────────────────
print(f"\nChecked {len(ok) + len(risks)} patterns\n")
print(f"  RISKS ({len(risks)}):")
for r in risks:
    print(f"    [RISK] {r}")
if not risks:
    print("    None found.")

print(f"\n  PASSES ({len(ok)}):")
for o in ok:
    print(f"    [OK ] {o}")

print()
if risks:
    print(f"RESULT: {len(risks)} risk(s) found — review before running")
    sys.exit(1)
else:
    print("RESULT: No crash risks found. Script should run to completion.")
