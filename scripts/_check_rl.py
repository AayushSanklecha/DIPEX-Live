"""Final RL system syntax + integrity check."""
import ast, sys
from pathlib import Path

FILES = [
    "learning/rl_agent/agent.py",
    "learning/rl_agent/ppo_trainer.py",
    "learning/rl_agent/policy_network.py",
    "ingestion/pipeline_bridge.py",
    "scripts/train_models/train_rl_ppo_agent.py",
]

all_ok = True
for fpath in FILES:
    src = Path(fpath).read_text(encoding="utf-8")
    try:
        ast.parse(src)
        lines = src.splitlines()
        print(f"  [OK ] {fpath} ({len(lines):,} lines)")
    except SyntaxError as e:
        print(f"  [ERR] {fpath} — line {e.lineno}: {e.msg}")
        all_ok = False

print()

# Integrity checks
bridge  = Path("ingestion/pipeline_bridge.py").read_text(encoding="utf-8")
agent   = Path("learning/rl_agent/agent.py").read_text(encoding="utf-8")
trainer = Path("learning/rl_agent/ppo_trainer.py").read_text(encoding="utf-8")
policy  = Path("learning/rl_agent/policy_network.py").read_text(encoding="utf-8")
train_s = Path("scripts/train_models/train_rl_ppo_agent.py").read_text(encoding="utf-8")

CHECKS = [
    ("_stage_rl_update method exists",          "def _stage_rl_update" in bridge),
    ("recommend() called in Step 0",             "_ppo_agent.recommend" in bridge),
    ("_ppo_agent_instance cached on self",       "_ppo_agent_instance" in bridge),
    ("Stage 11 uses cached instance",            "getattr(self, \"_ppo_agent_instance\"" in bridge),
    ("record_outcome() called in Stage 11",      "agent.record_outcome(" in bridge),
    ("Buffer min 32 transitions",                "n_transitions >= 32" in agent),
    ("PPOTrainer stores actor_opt",              "self._actor_opt" in trainer),
    ("PPOTrainer stores critic_opt",             "self._critic_opt" in trainer),
    ("_sync_numpy_to_torch exists",              "def _sync_numpy_to_torch" in trainer),
    ("PolicyNetwork uses isolated RNG",          "_POLICY_RNG" in policy),
    ("np.random.choice removed from policy",     "np.random.choice" not in policy),
    ("healthcare_phi scenario implemented",      "healthcare_phi" in train_s),
    ("ecommerce_fraud scenario implemented",     "ecommerce_fraud" in train_s),
    ("time_series scenario implemented",         "time_series scenario" in train_s or "domain=\"finance\"" in train_s),
    ("N_EPISODES=1000",                          "N_EPISODES     = 1000" in train_s),
    ("EVAL_EPISODES=30",                         "EVAL_EPISODES  = 30" in train_s),
    ("Gate tightened to 0.09",                   "0.09" in train_s),
    ("user_approved_plan wired",                 "user_approved_plan=user_approved" in bridge),
    ("gate_decision_cache updated",              "_gate_decision_cache" in bridge),
    ("Non-fatal RL exception handling",          "except Exception as exc" in bridge),
]

print("INTEGRITY CHECKS:")
passed = 0
for name, ok in CHECKS:
    icon = "OK  " if ok else "FAIL"
    print(f"  [{icon}] {name}")
    if ok:
        passed += 1
    else:
        all_ok = False

print()
print(f"Passed: {passed}/{len(CHECKS)}")
print("ALL RL CHECKS PASSED" if all_ok else "ISSUES REMAIN")
sys.exit(0 if all_ok else 1)
