#!/usr/bin/env python
"""
StreamBox V3 -- Hyperparameter tuning, feature selection, ensemble optimization.
V2 baseline: primary=0.8698, rolling=0.8641 (CatBoost), 0.8635 (ensemble)

This script reuses V2 cached features and focuses on:
  1. Focused hyperparameter search (CatBoost, LightGBM, XGBoost)
  2. Feature selection (top-N by importance)
  3. Feature family ablation (behavioral vs customer-heavy)
  4. Ensemble weight optimization using OOF predictions
  5. Seed robustness check
"""
import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# ============================================================================
# Import V2 pipeline functions
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streambox_v2 import (
    load_data, build_features, make_temporal_splits, run_leakage_checks,
    _eff_snapshot, _safe_div, _cb_frame, _inner_split, _train_auc,
    HAVE, NUM_THREADS, RANDOM_SEED, ES_ROUNDS, DATA_DIR, CACHE_DIR,
    build_event_features_v2,
)
# Override V2's FORCE_REBUILD to use cached features
import streambox_v2
streambox_v2.FORCE_REBUILD = False

OUT_DIR = "outputs_v3"
T0 = time.time()
def log(msg=""): print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

ES_HOLDOUT = 0.15

# ============================================================================
# MODEL TRAINERS (parameterized)
# ============================================================================
def train_catboost(Xtr, ytr, Xva, cats, params, seed=RANDOM_SEED):
    from catboost import CatBoostClassifier
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=seed)
    Xt, Xv = _cb_frame(Xtr, cats), _cb_frame(Xva, cats)
    m = CatBoostClassifier(random_seed=seed, thread_count=NUM_THREADS, cat_features=cats, **params)
    m.fit(Xt.iloc[a], ytr[a], eval_set=(Xt.iloc[b], ytr[b]), use_best_model=True)
    pv = m.predict_proba(Xv)[:, 1]
    tr_auc = roc_auc_score(ytr[a[:min(20000, len(a))]], m.predict_proba(Xt.iloc[a[:min(20000, len(a))]])[:, 1])
    return m, pv, tr_auc, m.get_best_iteration()

def train_lightgbm(Xtr, ytr, Xva, cats, params, seed=RANDOM_SEED):
    import lightgbm as lgb
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=seed)
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=NUM_THREADS, **params)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    return m, m.predict_proba(Xva)[:, 1], roc_auc_score(ytr[a[:20000]], m.predict_proba(Xtr.iloc[a[:20000]])[:, 1]), m.best_iteration_

def train_xgboost(Xtr, ytr, Xva, cats, params, seed=RANDOM_SEED):
    import xgboost as xgb
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=seed)
    m = xgb.XGBClassifier(random_state=seed, n_jobs=NUM_THREADS, early_stopping_rounds=ES_ROUNDS, **params)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], verbose=False)
    return m, m.predict_proba(Xva)[:, 1], roc_auc_score(ytr[a[:20000]], m.predict_proba(Xtr.iloc[a[:20000]])[:, 1]), m.best_iteration

TRAINERS = {"catboost": train_catboost, "lightgbm": train_lightgbm, "xgboost": train_xgboost}

def eval_on_splits(model_name, full, cols, cats, splits, params, seed=RANDOM_SEED,
                   only_primary=False, verbose=True):
    """Train on each temporal fold, return dict with per-fold AUC and OOF predictions."""
    fn = TRAINERS[model_name]
    cats_used = [c for c in cats if c in cols]
    res = {"aucs": {}, "oof": [], "models": {}, "time": 0.0, "train_auc": [], "iters": []}
    for sp in splits:
        if only_primary and not sp["primary"]: continue
        t = time.time()
        Xtr = full.loc[sp["train_idx"], cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
        Xva = full.loc[sp["valid_idx"], cols]; yva = full.loc[sp["valid_idx"], "churn"].astype(int).values
        m, pv, tr_auc, bi = fn(Xtr, ytr, Xva, cats_used, params, seed)
        auc = roc_auc_score(yva, pv)
        res["aucs"][sp["valid_month"]] = auc; res["models"][sp["valid_month"]] = m
        res["train_auc"].append(tr_auc); res["iters"].append(bi)
        res["oof"].append(pd.DataFrame({"row_id": full.loc[sp["valid_idx"], "row_id"].values,
                                         "fold": sp["valid_month"], "y": yva, "pred": pv}))
        res["time"] += time.time() - t
        if verbose:
            print(f"      {model_name:9s} {sp['name']:22s} AUC={auc:.5f}  train={tr_auc:.4f} iters={bi} {time.time()-t:.0f}s")
    a = np.array(list(res["aucs"].values()))
    res["mean"], res["std"] = a.mean(), a.std()
    prim = [sp["valid_month"] for sp in splits if sp["primary"]][0]
    res["primary"] = res["aucs"].get(prim, np.nan)
    return res

def model_importance(name, model, cols):
    if name == "lightgbm": v = model.booster_.feature_importance("gain")
    elif name == "xgboost":
        s = model.get_booster().get_score(importance_type="gain")
        v = np.array([s.get(c, 0.0) for c in cols])
    else: v = model.get_feature_importance()
    v = np.asarray(v, dtype=float)
    return pd.DataFrame({"feature": cols, "importance": v}).sort_values("importance", ascending=False)

# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True); np.random.seed(RANDOM_SEED)
    print("=" * 70)
    print("STREAMBOX V3 -- Hyperparameter Tuning & Feature Selection")
    print("V2 baseline: CatBoost primary=0.8697, rolling=0.8641")
    print("             Ensemble primary=0.8698, rolling=0.8635")
    print(f"Libraries: {HAVE} | threads={NUM_THREADS}")
    print("=" * 70)

    # ---- Load data and build V2 features (cached) ----
    train, test, events, ss = load_data()
    full, feature_cols, cat_cols, groups = build_features(train, test, events)
    splits = make_temporal_splits(full)
    all_results = []

    # ================================================================
    # STEP 0: Verify V2 baseline
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 0: VERIFY V2 BASELINE\n" + "=" * 70)

    v2_cb_params = dict(iterations=4000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                        random_strength=0.5, bagging_temperature=0.3,
                        eval_metric="AUC", od_type="Iter", od_wait=200, verbose=0)
    log("V2 CatBoost baseline")
    r_cb = eval_on_splits("catboost", full, feature_cols, cat_cols, splits, v2_cb_params)
    all_results.append(("V2 CatBoost", "catboost", r_cb))
    print(f"   -> primary={r_cb['primary']:.5f}  rolling={r_cb['mean']:.5f}  std={r_cb['std']:.5f}")

    v2_lgb_params = dict(n_estimators=5000, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                         max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                         reg_alpha=0.1, reg_lambda=2.0, verbose=-1)
    log("V2 LightGBM baseline")
    r_lgb = eval_on_splits("lightgbm", full, feature_cols, cat_cols, splits, v2_lgb_params)
    all_results.append(("V2 LightGBM", "lightgbm", r_lgb))
    print(f"   -> primary={r_lgb['primary']:.5f}  rolling={r_lgb['mean']:.5f}  std={r_lgb['std']:.5f}")

    v2_xgb_params = dict(n_estimators=5000, learning_rate=0.02, max_depth=7, min_child_weight=10,
                         subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0,
                         tree_method="hist", enable_categorical=True, eval_metric="auc")
    log("V2 XGBoost baseline")
    r_xgb = eval_on_splits("xgboost", full, feature_cols, cat_cols, splits, v2_xgb_params)
    all_results.append(("V2 XGBoost", "xgboost", r_xgb))
    print(f"   -> primary={r_xgb['primary']:.5f}  rolling={r_xgb['mean']:.5f}  std={r_xgb['std']:.5f}")

    # ================================================================
    # STEP 1: CatBoost hyperparameter search
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 1: CATBOOST HYPERPARAMETER SEARCH\n" + "=" * 70)

    cb_candidates = [
        dict(depth=6, learning_rate=0.03, l2_leaf_reg=3.0, random_strength=0.5, bagging_temperature=0.3),
        dict(depth=7, learning_rate=0.02, l2_leaf_reg=5.0, random_strength=0.5, bagging_temperature=0.3),
        dict(depth=8, learning_rate=0.02, l2_leaf_reg=5.0, random_strength=0.5, bagging_temperature=0.5),
        dict(depth=8, learning_rate=0.03, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5),
        dict(depth=7, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=0.5, bagging_temperature=0.3),
        dict(depth=9, learning_rate=0.02, l2_leaf_reg=10.0, random_strength=1.0, bagging_temperature=0.5),
        dict(depth=6, learning_rate=0.05, l2_leaf_reg=5.0, random_strength=0, bagging_temperature=0),
        dict(depth=7, learning_rate=0.05, l2_leaf_reg=3.0, random_strength=0, bagging_temperature=0),
        dict(depth=8, learning_rate=0.03, l2_leaf_reg=10.0, random_strength=0.5, bagging_temperature=1.0),
        dict(depth=10, learning_rate=0.02, l2_leaf_reg=20.0, random_strength=1.0, bagging_temperature=1.0),
        dict(depth=7, learning_rate=0.03, l2_leaf_reg=5.0, random_strength=0, bagging_temperature=0.5),
        dict(depth=8, learning_rate=0.02, l2_leaf_reg=3.0, random_strength=0.5, bagging_temperature=0),
    ]

    cb_base = dict(iterations=5000, eval_metric="AUC", od_type="Iter", od_wait=200, verbose=0)
    cb_screen = []
    log("Screening CatBoost candidates on PRIMARY fold only...")
    for i, cand in enumerate(cb_candidates):
        p = {**cb_base, **cand}
        r = eval_on_splits("catboost", full, feature_cols, cat_cols, splits, p, only_primary=True, verbose=False)
        cb_screen.append((i, cand, r["primary"], r))
        print(f"   [{i:2d}] d={cand['depth']} lr={cand['learning_rate']} l2={cand['l2_leaf_reg']} "
              f"rs={cand['random_strength']} bt={cand['bagging_temperature']} -> April={r['primary']:.5f} {r['time']:.0f}s")

    cb_screen.sort(key=lambda x: -x[2])
    top_cb = cb_screen[:4]
    print(f"\n   Top 4 candidates -> full rolling evaluation:")

    best_cb_rolling = 0; best_cb_params = None; best_cb_result = None
    for rank, (idx, cand, prim_auc, _) in enumerate(top_cb):
        p = {**cb_base, **cand}
        r = eval_on_splits("catboost", full, feature_cols, cat_cols, splits, p, verbose=False)
        print(f"   [{idx}] primary={r['primary']:.5f} rolling={r['mean']:.5f} std={r['std']:.5f}")
        if r["mean"] > best_cb_rolling:
            best_cb_rolling = r["mean"]; best_cb_params = p; best_cb_result = r
    all_results.append(("V3 CatBoost tuned", "catboost", best_cb_result))
    log(f"Best CatBoost: rolling={best_cb_rolling:.5f} primary={best_cb_result['primary']:.5f}")

    # ================================================================
    # STEP 2: LightGBM hyperparameter search
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 2: LIGHTGBM HYPERPARAMETER SEARCH\n" + "=" * 70)

    lgb_candidates = [
        dict(num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0),
        dict(num_leaves=63, min_child_samples=50, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0),
        dict(num_leaves=63, min_child_samples=20, subsample=0.7, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=3.0),
        dict(num_leaves=127, min_child_samples=50, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0),
        dict(num_leaves=63, min_child_samples=30, subsample=0.85, colsample_bytree=0.65, reg_alpha=0.0, reg_lambda=2.0, min_split_gain=0.01),
        dict(num_leaves=47, min_child_samples=40, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=5.0),
        dict(num_leaves=63, min_child_samples=30, subsample=0.75, colsample_bytree=0.55, reg_alpha=0.5, reg_lambda=1.0),
        dict(num_leaves=95, min_child_samples=30, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=3.0),
    ]
    lgb_base = dict(n_estimators=5000, learning_rate=0.02, subsample_freq=1, verbose=-1)
    lgb_screen = []
    log("Screening LightGBM candidates on PRIMARY fold...")
    for i, cand in enumerate(lgb_candidates):
        p = {**lgb_base, **cand}
        r = eval_on_splits("lightgbm", full, feature_cols, cat_cols, splits, p, only_primary=True, verbose=False)
        lgb_screen.append((i, cand, r["primary"], r))
        print(f"   [{i:2d}] nl={cand['num_leaves']} mcs={cand['min_child_samples']} "
              f"ss={cand['subsample']} cbt={cand['colsample_bytree']} -> April={r['primary']:.5f} {r['time']:.0f}s")

    lgb_screen.sort(key=lambda x: -x[2])
    top_lgb = lgb_screen[:4]
    print(f"\n   Top 4 -> full rolling:")

    best_lgb_rolling = 0; best_lgb_params = None; best_lgb_result = None
    for rank, (idx, cand, prim_auc, _) in enumerate(top_lgb):
        p = {**lgb_base, **cand}
        r = eval_on_splits("lightgbm", full, feature_cols, cat_cols, splits, p, verbose=False)
        print(f"   [{idx}] primary={r['primary']:.5f} rolling={r['mean']:.5f} std={r['std']:.5f}")
        if r["mean"] > best_lgb_rolling:
            best_lgb_rolling = r["mean"]; best_lgb_params = p; best_lgb_result = r
    all_results.append(("V3 LightGBM tuned", "lightgbm", best_lgb_result))
    log(f"Best LightGBM: rolling={best_lgb_rolling:.5f} primary={best_lgb_result['primary']:.5f}")

    # ================================================================
    # STEP 3: XGBoost hyperparameter search
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 3: XGBOOST HYPERPARAMETER SEARCH\n" + "=" * 70)

    xgb_candidates = [
        dict(max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0),
        dict(max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0),
        dict(max_depth=8, min_child_weight=10, subsample=0.75, colsample_bytree=0.55, reg_alpha=0.5, reg_lambda=3.0),
        dict(max_depth=7, min_child_weight=15, subsample=0.85, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=5.0),
        dict(max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.65, reg_alpha=0.0, reg_lambda=2.0, gamma=0.1),
        dict(max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0),
        dict(max_depth=9, min_child_weight=15, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=5.0),
        dict(max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=3.0),
    ]
    xgb_base = dict(n_estimators=5000, learning_rate=0.02, tree_method="hist", enable_categorical=True, eval_metric="auc")
    xgb_screen = []
    log("Screening XGBoost candidates on PRIMARY fold...")
    for i, cand in enumerate(xgb_candidates):
        p = {**xgb_base, **cand}
        r = eval_on_splits("xgboost", full, feature_cols, cat_cols, splits, p, only_primary=True, verbose=False)
        xgb_screen.append((i, cand, r["primary"], r))
        print(f"   [{i:2d}] md={cand['max_depth']} mcw={cand['min_child_weight']} "
              f"ss={cand['subsample']} cbt={cand['colsample_bytree']} -> April={r['primary']:.5f} {r['time']:.0f}s")

    xgb_screen.sort(key=lambda x: -x[2])
    top_xgb = xgb_screen[:4]
    print(f"\n   Top 4 -> full rolling:")

    best_xgb_rolling = 0; best_xgb_params = None; best_xgb_result = None
    for rank, (idx, cand, prim_auc, _) in enumerate(top_xgb):
        p = {**xgb_base, **cand}
        r = eval_on_splits("xgboost", full, feature_cols, cat_cols, splits, p, verbose=False)
        print(f"   [{idx}] primary={r['primary']:.5f} rolling={r['mean']:.5f} std={r['std']:.5f}")
        if r["mean"] > best_xgb_rolling:
            best_xgb_rolling = r["mean"]; best_xgb_params = p; best_xgb_result = r
    all_results.append(("V3 XGBoost tuned", "xgboost", best_xgb_result))
    log(f"Best XGBoost: rolling={best_xgb_rolling:.5f} primary={best_xgb_result['primary']:.5f}")

    # ================================================================
    # STEP 4: Feature selection with best CatBoost
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 4: FEATURE SELECTION (CatBoost)\n" + "=" * 70)

    # Get feature importance from the PRIMARY fold's training portion
    # (not validation, to avoid leakage)
    prim_split = [s for s in splits if s["primary"]][0]
    Xtr_prim = full.loc[prim_split["train_idx"], feature_cols]
    ytr_prim = full.loc[prim_split["train_idx"], "churn"].astype(int).values
    Xva_prim = full.loc[prim_split["valid_idx"], feature_cols]
    cats_used = [c for c in cat_cols if c in feature_cols]

    # Train a single model for importance
    m_imp, _, _, _ = train_catboost(Xtr_prim, ytr_prim, Xva_prim, cats_used, best_cb_params)
    imp = model_importance("catboost", m_imp, feature_cols)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance_v3.csv"), index=False)

    # Test feature subsets
    for n_top in [150, 100, 75, 50]:
        top_feats = imp.head(n_top)["feature"].tolist()
        log(f"Feature selection: top {n_top}")
        r = eval_on_splits("catboost", full, top_feats, cat_cols, splits, best_cb_params, verbose=False)
        all_results.append((f"V3 CB top-{n_top}", "catboost", r))
        print(f"   top-{n_top}: primary={r['primary']:.5f} rolling={r['mean']:.5f} std={r['std']:.5f}")

    # ================================================================
    # STEP 5: Feature family ablation (behavioral vs customer-heavy)
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 5: BEHAVIORAL vs CUSTOMER-HEAVY FEATURES\n" + "=" * 70)

    # Customer/tenure-heavy: remove session/event behavioral features
    cust_feats = groups.get("customer", [])
    all_event_feats = (groups.get("counts", []) + groups.get("recency", []) + groups.get("windows", []) +
                       groups.get("gaps", []) + groups.get("decay", []) + groups.get("relative", []) +
                       groups.get("consistency", []) + groups.get("interactions", []))

    # Behavioral-heavy: prioritize event features, drop high-shift customer features
    high_shift_feats = {"referral_code_count_so_far", "referral_code", "tenure_days", "tenure_months",
                        "channel_x_plan", "tenure_in_cycles"}
    behavioral_feats = [c for c in feature_cols if c not in high_shift_feats]

    log("Behavioral-heavy (drop high-shift customer feats)")
    r_beh = eval_on_splits("catboost", full, behavioral_feats, cat_cols, splits, best_cb_params, verbose=False)
    all_results.append(("V3 behavioral-heavy", "catboost", r_beh))
    print(f"   behavioral-heavy: primary={r_beh['primary']:.5f} rolling={r_beh['mean']:.5f}")

    # Customer-only (no event features)
    cust_only = [c for c in feature_cols if c in cust_feats]
    log("Customer-only (no event features)")
    r_cust = eval_on_splits("catboost", full, cust_only, cat_cols, splits, best_cb_params, verbose=False)
    all_results.append(("V3 customer-only", "catboost", r_cust))
    print(f"   customer-only: primary={r_cust['primary']:.5f} rolling={r_cust['mean']:.5f}")

    # ================================================================
    # STEP 6: Ensemble optimization with OOF predictions
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 6: ENSEMBLE OPTIMIZATION\n" + "=" * 70)

    # Collect OOF predictions from the best tuned models (full 3-fold)
    log("Collecting OOF predictions from best tuned models...")
    oof_cb = pd.concat(best_cb_result["oof"], ignore_index=True)
    oof_lgb = pd.concat(best_lgb_result["oof"], ignore_index=True)
    oof_xgb = pd.concat(best_xgb_result["oof"], ignore_index=True)

    base = oof_cb[["row_id", "fold", "y"]].copy()
    base["cb"] = oof_cb["pred"].values
    base["lgb"] = oof_lgb["pred"].values
    base["xgb"] = oof_xgb["pred"].values

    # Rank versions
    for k in ["cb", "lgb", "xgb"]:
        base[k + "_rank"] = base.groupby("fold")[k].rank(pct=True)

    # Test many weight combinations
    weight_combos = []
    for w_cb in np.arange(0.3, 0.85, 0.05):
        for w_lgb in np.arange(0.05, 0.55, 0.05):
            w_xgb = 1.0 - w_cb - w_lgb
            if w_xgb < 0.02 or w_xgb > 0.5: continue
            weight_combos.append((round(w_cb, 2), round(w_lgb, 2), round(w_xgb, 2)))

    # Evaluate each weight combo on OOF
    best_ens_auc = 0; best_ens_weights = None; best_ens_type = None
    ens_results = []

    for w_cb, w_lgb, w_xgb in weight_combos:
        # Probability weighted
        base["ens_prob"] = w_cb * base["cb"] + w_lgb * base["lgb"] + w_xgb * base["xgb"]
        aucs_prob = {f: roc_auc_score(g["y"], g["ens_prob"]) for f, g in base.groupby("fold")}
        mean_prob = np.mean(list(aucs_prob.values()))

        # Rank weighted
        base["ens_rank"] = w_cb * base["cb_rank"] + w_lgb * base["lgb_rank"] + w_xgb * base["xgb_rank"]
        aucs_rank = {f: roc_auc_score(g["y"], g["ens_rank"]) for f, g in base.groupby("fold")}
        mean_rank = np.mean(list(aucs_rank.values()))

        ens_results.append((w_cb, w_lgb, w_xgb, mean_prob, mean_rank))

        for mean_val, etype in [(mean_prob, "prob"), (mean_rank, "rank")]:
            if mean_val > best_ens_auc:
                best_ens_auc = mean_val
                best_ens_weights = (w_cb, w_lgb, w_xgb)
                best_ens_type = etype

    # Sort and show top 10
    ens_df = pd.DataFrame(ens_results, columns=["w_cb", "w_lgb", "w_xgb", "rolling_prob", "rolling_rank"])
    ens_df["best_rolling"] = ens_df[["rolling_prob", "rolling_rank"]].max(axis=1)
    ens_df = ens_df.sort_values("best_rolling", ascending=False)
    print("\nTop 10 ensemble weight combinations:")
    print(ens_df.head(10).to_string(index=False))

    # Evaluate best ensemble on per-fold basis
    w_cb, w_lgb, w_xgb = best_ens_weights
    if best_ens_type == "prob":
        base["best_ens"] = w_cb * base["cb"] + w_lgb * base["lgb"] + w_xgb * base["xgb"]
    else:
        base["best_ens"] = w_cb * base["cb_rank"] + w_lgb * base["lgb_rank"] + w_xgb * base["xgb_rank"]
    ens_aucs = {f: roc_auc_score(g["y"], g["best_ens"]) for f, g in base.groupby("fold")}
    prim_month = [s["valid_month"] for s in splits if s["primary"]][0]
    ens_prim = ens_aucs[prim_month]
    ens_rolling = np.mean(list(ens_aucs.values()))
    ens_std = np.std(list(ens_aucs.values()))

    print(f"\nBest ensemble: type={best_ens_type} weights=CB:{w_cb} LGB:{w_lgb} XGB:{w_xgb}")
    print(f"   Per-fold: {ens_aucs}")
    print(f"   Primary={ens_prim:.5f}  Rolling={ens_rolling:.5f}  Std={ens_std:.5f}")

    # Also test the specific requested combos
    print("\nPre-specified ensemble combos:")
    for w_cb, w_lgb, w_xgb in [(0.70, 0.20, 0.10), (0.60, 0.25, 0.15), (0.80, 0.10, 0.10),
                                 (0.50, 0.30, 0.20), (1/3, 1/3, 1/3)]:
        for etype in ["prob", "rank"]:
            if etype == "prob":
                ens = w_cb * base["cb"] + w_lgb * base["lgb"] + w_xgb * base["xgb"]
            else:
                ens = w_cb * base["cb_rank"] + w_lgb * base["lgb_rank"] + w_xgb * base["xgb_rank"]
            aucs = {f: roc_auc_score(g["y"], ens.loc[g.index]) for f, g in base.groupby("fold")}
            rm = np.mean(list(aucs.values()))
            print(f"   {etype:4s} CB={w_cb:.2f} LGB={w_lgb:.2f} XGB={w_xgb:.2f} -> rolling={rm:.5f} primary={aucs[prim_month]:.5f}")

    # Store ensemble result
    ens_result_obj = {"aucs": ens_aucs, "mean": ens_rolling, "std": ens_std, "primary": ens_prim}
    all_results.append((f"V3 Ensemble ({best_ens_type} {w_cb}/{w_lgb}/{w_xgb})", "ensemble", ens_result_obj))

    # Also add equal-weight ensembles
    for etype in ["prob", "rank"]:
        if etype == "prob":
            ens = (base["cb"] + base["lgb"] + base["xgb"]) / 3
        else:
            ens = (base["cb_rank"] + base["lgb_rank"] + base["xgb_rank"]) / 3
        aucs = {f: roc_auc_score(g["y"], ens.loc[g.index]) for f, g in base.groupby("fold")}
        rm = np.mean(list(aucs.values())); rp = aucs[prim_month]; rs = np.std(list(aucs.values()))
        all_results.append((f"V3 Equal {etype}", "ensemble", {"aucs": aucs, "mean": rm, "std": rs, "primary": rp}))

    # ================================================================
    # STEP 7: Seed robustness for best CatBoost
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 7: SEED ROBUSTNESS CHECK\n" + "=" * 70)

    seeds = [42, 123, 2026]
    seed_results = []
    for seed in seeds:
        r = eval_on_splits("catboost", full, feature_cols, cat_cols, splits, best_cb_params, seed=seed, verbose=False)
        seed_results.append((seed, r["primary"], r["mean"], r["std"]))
        print(f"   seed={seed}: primary={r['primary']:.5f} rolling={r['mean']:.5f} std={r['std']:.5f}")

    seed_primaries = [x[1] for x in seed_results]
    seed_rollings = [x[2] for x in seed_results]
    print(f"   Primary range: {min(seed_primaries):.5f} - {max(seed_primaries):.5f} (spread={max(seed_primaries)-min(seed_primaries):.5f})")
    print(f"   Rolling range: {min(seed_rollings):.5f} - {max(seed_rollings):.5f} (spread={max(seed_rollings)-min(seed_rollings):.5f})")

    # ================================================================
    # STEP 8: Feature importance for best model
    # ================================================================
    print("\n" + "=" * 70 + "\nSTEP 8: TOP 30 FEATURES (Best CatBoost)\n" + "=" * 70)
    imp_top = imp.head(30).copy()
    imp_top["importance_pct"] = 100 * imp_top["importance"] / imp["importance"].sum()
    print(imp_top[["feature", "importance_pct"]].round(3).to_string(index=False))

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("STREAMBOX V3 FINAL RESULTS")
    print("=" * 60)

    print(f"\n{'Experiment':45s} {'Feb':>8s} {'Mar':>8s} {'Apr':>8s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 85)
    for name, model, r in all_results:
        aucs = r.get("aucs", {})
        feb = aucs.get("2026-02", np.nan)
        mar = aucs.get("2026-03", np.nan)
        apr = aucs.get("2026-04", np.nan)
        m = r.get("mean", np.nan); s = r.get("std", np.nan)
        print(f"{name:45s} {feb:8.4f} {mar:8.4f} {apr:8.4f} {m:8.4f} {s:8.4f}")

    # Find best by rolling mean
    best_name, best_model, best_r = max(all_results, key=lambda x: x[2].get("mean", 0))
    v2_primary = 0.8698; v2_rolling = 0.8641

    print(f"\nV2 baseline:")
    print(f"    Best primary AUC = {v2_primary}")
    print(f"    Best rolling mean = {v2_rolling}")
    print(f"\nV3 best (by rolling mean):")
    print(f"    Experiment = {best_name}")
    print(f"    Primary AUC = {best_r['primary']:.5f}")
    print(f"    Rolling mean = {best_r['mean']:.5f}")
    print(f"    Rolling std = {best_r.get('std', 0):.5f}")
    print(f"\nImprovement over V2:")
    print(f"    Primary = {best_r['primary'] - v2_primary:+.5f}")
    print(f"    Rolling = {best_r['mean'] - v2_rolling:+.5f}")

    print(f"\nBest CatBoost params: {best_cb_params}")
    print(f"Best LightGBM params: {best_lgb_params}")
    print(f"Best XGBoost params: {best_xgb_params}")
    print(f"Best ensemble: type={best_ens_type} weights=CB:{best_ens_weights[0]} LGB:{best_ens_weights[1]} XGB:{best_ens_weights[2]}")

    print(f"\nSeed stability (CatBoost):")
    print(f"    Primary spread = {max(seed_primaries)-min(seed_primaries):.5f}")
    print(f"    Rolling spread = {max(seed_rollings)-min(seed_rollings):.5f}")

    print(f"\nLeakage status: V2 leakage checks passed (brute-force + poison test)")
    print(f"Feature count: {len(feature_cols)}")

    # Save
    summary_rows = []
    for name, model, r in all_results:
        aucs = r.get("aucs", {})
        summary_rows.append({"experiment": name, "model": model,
                             "auc_feb": aucs.get("2026-02", np.nan), "auc_mar": aucs.get("2026-03", np.nan),
                             "auc_apr": aucs.get("2026-04", np.nan),
                             "rolling_mean": r.get("mean", np.nan), "rolling_std": r.get("std", np.nan)})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "v3_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    # Find best single model and best ensemble
    best_single_name, _, best_single_r = max(
        [(n, m, r) for n, m, r in all_results if "Ensemble" not in n and "Equal" not in n],
        key=lambda x: x[2].get("mean", 0))
    best_ens_name, _, best_ens_r = max(
        [(n, m, r) for n, m, r in all_results if "Ensemble" in n or "Equal" in n],
        key=lambda x: x[2].get("mean", 0))

    print(f"\nBest single model: {best_single_name}")
    print(f"    Rolling = {best_single_r['mean']:.5f}, Primary = {best_single_r['primary']:.5f}")
    print(f"\nBest ensemble: {best_ens_name}")
    print(f"    Rolling = {best_ens_r['mean']:.5f}, Primary = {best_ens_r['primary']:.5f}")

    if best_ens_r["mean"] > best_single_r["mean"]:
        print(f"\n-> RECOMMEND: {best_ens_name} for final Kaggle notebook")
    else:
        print(f"\n-> RECOMMEND: {best_single_name} for final Kaggle notebook")

    print(f"\nSaved to {OUT_DIR}/: v3_results.csv, feature_importance_v3.csv")
    log("done")

if __name__ == "__main__":
    main()
