"""Audit RL integration in pipeline_bridge.py"""
lines = open('ingestion/pipeline_bridge.py', encoding='utf-8').readlines()
KEYS = ['rl_update', 'rl_plan', 'rl_recs', 'PPOAgent', 'recommend(', 'record_outcome',
        'ReinforcementUpdate', 'imputation', 'cv_strategy', 'model_complexity',
        'outlier', 'retry_budget', 'confidence_threshold']
for i, l in enumerate(lines, 1):
    if any(k in l for k in KEYS):
        print(f'{i:4d}: {l.rstrip()}')
