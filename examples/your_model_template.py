"""
Template , plug YOUR model into SiMetrics Task 1 and compare to the leaderboard.

Run from the repo root:
    python examples/your_model_template.py

Replace the "TRAIN YOUR MODEL" block with anything (XGBoost, a neural net, a GNN, ...). The only
rule: train on the train split, produce a failure-probability score per TEST row, and score with the
OFFICIAL metric so your number is comparable to the shipped leaderboard.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repo root
from evaluate import task1_data, score_task1

# 1) Official featurized data (train seeds 1-6, val 7, test 8). Observables only; `rul` is a label.
d = task1_data()
print("features:", d["feature_names"])
print("train rows:", len(d["ytr"]), "| test rows:", len(d["yte"]))

# 2) ---- TRAIN YOUR MODEL (replace this block) ----
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1)
model.fit(d["Xtr"], d["ytr"])
test_scores = model.predict_proba(d["Xte"])[:, 1]          # P(fail within 90 days) per test row

# 3) Score with the official metric and compare.
print("\nYour model , Task 1:", score_task1(d["yte"], test_scores))
print("Leaderboard to beat   : MLP  AUC 0.843 / AP 0.356  (run `python evaluate.py` to reproduce)")

# ---- Want to use TOPOLOGY (e.g. a GNN)? ----
# The static graph is in d["top"]:
#   d["top"]["ids"], ["node_features"] (z, length, ampacity, vn, base_loading, downstream),
#   ["elec_edges"] (line-graph), ["coupling_edges"] (tie-transfer / contingency neighbours).
# Build your own spatio-temporal features from the raw run arrays in
# data/simetrics_benchmark/data/<case>_<policy>_seedNN.npz, but produce scores for the SAME test
# rows (or re-use build_xy's sampling) and score with score_task1 for a fair comparison.
