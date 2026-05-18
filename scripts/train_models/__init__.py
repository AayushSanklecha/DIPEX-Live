"""
scripts/train_models/__init__.py
---------------------------------
Model training scripts for ADAP. All scripts are designed to run in
Google Colab or locally.

Available scripts:
  - train_domain_classifier.py  : domain detection XGBoost + Optuna
  - train_rl_ppo_agent.py       : PPO RL agent on synthetic env
  - train_anomaly_detector.py   : IsolationForest + LOF ensemble

Quality gates enforced:
  - val/holdout gap < 3% (overfitting check)
  - val_score >= domain-specific minimum
  - Learning curve positively sloped (underfitting check)
"""
