#!/usr/bin/env python
"""
StreamBox churn -- LOCAL experiment pipeline (not the final Kaggle notebook).

Install (once):
    pip install pandas numpy scikit-learn catboost lightgbm xgboost
    pip install shap          # optional, only if USE_SHAP = True

Expected files in DATA_DIR:  train.csv  test.csv  events.csv  sample_submission.csv
Run:  python streambox_local_experiments.py

What it does
  1. loads + inspects data, runs leakage / drift diagnostics
  2. builds customer features and SNAPSHOT-AWARE event features (vectorised, cached)
  3. proves no future events enter the features (brute-force re-computation + "poison" test)
  4. temporal validation: rolling Jan->Feb, Jan+Feb->Mar, Jan+Feb+Mar->Apr (Apr = primary)
  5. experiment ladder (customer -> +counts -> +recency -> +windows), then CatBoost / LightGBM / XGBoost,
     rank-average ensemble, group ablation, feature importance (+ optional SHAP)
Every number printed is computed from your data. Nothing is hard-coded or fabricated.
"""
import os, sys, time, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG  (edit here)
# =============================================================================
DATA_DIR      = "data"
OUT_DIR       = "outputs"
CACHE_DIR     = "cache"
RANDOM_SEED   = 42
N_JOBS        = -1                 # -1 = all cores
FORCE_REBUILD = False              # True = ignore cached event features

WINDOWS       = [7, 14, 30, 60, 90, 120]       # days before snapshot
# Event inclusion rule. False (strict, default) = keep events with timestamp <= snapshot_date as a datetime.
# If snapshot_date is a pure date (00:00:00) this drops events during the snapshot day itself.
# True = treat the snapshot as END of that calendar day. Check the diagnostics printed in step 2 and decide.
SNAPSHOT_END_OF_DAY = False

GAP_DAYS      = 0                  # optional embargo: drop train rows within N days before validation start
ES_HOLDOUT    = 0.15               # inner random holdout of the TRAIN part used for early stopping
                                   # (validation month is never used for early stopping -> no optimistic bias)
ROLLING_FOR_BOOSTERS = True        # False = only primary split (Jan-Mar -> Apr) for CatBoost/LGBM/XGB (faster)
RUN_GROUP_ABLATION   = True
RUN_SHIFT_CHECK      = True
USE_SHAP             = False       # needs `pip install shap`; never required
WRITE_TEST_SUBMISSION = False      # fit on all train, predict test with best model (sanity only)
LADDER_MODEL  = "lightgbm"         # model used for experiments 1-4 and ablation (falls back if missing)

CATBOOST_PARAMS = dict(iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                       eval_metric="AUC", od_type="Iter", od_wait=150, verbose=0)
LIGHTGBM_PARAMS = dict(n_estimators=4000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
XGBOOST_PARAMS  = dict(n_estimators=4000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                       subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                       tree_method="hist", enable_categorical=True, eval_metric="auc")
LADDER_LGBM_PARAMS = dict(LIGHTGBM_PARAMS, learning_rate=0.06)   # faster for the ladder / ablation
ES_ROUNDS = 150

NUM_THREADS = (os.cpu_count() or 1) if N_JOBS in (-1, None) else N_JOBS
T0 = time.time()

def log(msg=""):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

def _have(mod):
    try: __import__(mod); return True
    except Exception: return False
HAVE = {m: _have(m) for m in ("catboost", "lightgbm", "xgboost", "shap")}

# =============================================================================
# 1. LOAD
# =============================================================================
def load_data():
    log("Loading data ...")
    p = lambda f: os.path.join(DATA_DIR, f)
    train = pd.read_csv(p("train.csv")); test = pd.read_csv(p("test.csv"))
    events = pd.read_csv(p("events.csv")); ss = pd.read_csv(p("sample_submission.csv"))
    for d in (train, test):
        d["snapshot_date"] = pd.to_datetime(d["snapshot_date"]); d["signup_date"] = pd.to_datetime(d["signup_date"])
    ts = pd.to_datetime(events["timestamp"], utc=False if events["timestamp"].astype(str).str.len().max() < 30 else True)
    if getattr(ts.dt, "tz", None) is not None: ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    events["timestamp"] = ts
    events["event_type"] = events["event_type"].astype("category")
    if train["churn"].dtype == object or str(train["churn"].dtype).startswith("str"):
        train["churn"] = train["churn"].astype(str).str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
    train["churn"] = train["churn"].astype(int)
    log(f"train {train.shape} | test {test.shape} | events {events.shape} | sample_submission {ss.shape}")
    return train, test, events, ss

# =============================================================================
# 2. INSPECT
# =============================================================================
def inspect_data(train, test, events, ss):
    print("\n" + "=" * 70 + "\nDATA INSPECTION\n" + "=" * 70)
    for name, d in (("train", train), ("test", test), ("events", events)):
        print(f"\n--- {name}: shape={d.shape}, memory={d.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(pd.DataFrame({"dtype": d.dtypes.astype(str), "missing": d.isna().sum(),
                            "missing%": (d.isna().mean() * 100).round(2), "nunique": d.nunique()}))
    print("\n--- target (train):", train["churn"].value_counts(normalize=True).round(4).to_dict())
    tm = train.groupby(train["snapshot_date"].dt.to_period("M"))["churn"].agg(["size", "mean"]).rename(columns={"size": "rows", "mean": "churn_rate"})
    print("\nchurn rate by snapshot month (train):\n", tm.round(4))
    print("\ntest rows by snapshot month:\n", test["snapshot_date"].dt.to_period("M").value_counts().sort_index())
    for name, d in (("train", train), ("test", test)):
        print(f"{name}: snapshot {d['snapshot_date'].min().date()} .. {d['snapshot_date'].max().date()} | "
              f"signup {d['signup_date'].min().date()} .. {d['signup_date'].max().date()}")
    print(f"events: {events['timestamp'].min()} .. {events['timestamp'].max()}")
    print("\n--- categorical cardinalities (train / test / unseen-in-train share of test rows):")
    for c in ("plan", "billing_cycle", "region", "age_band", "acquisition_channel", "referral_code", "payment_method"):
        seen = set(train[c].dropna().unique())
        print(f"  {c:20s} {train[c].nunique():7d} {test[c].nunique():7d}   unseen {(~test[c].dropna().isin(seen)).mean() * 100:5.2f}%   "
              f"missing tr/te {train[c].isna().mean() * 100:5.1f}%/{test[c].isna().mean() * 100:5.1f}%")
    print("\n--- event distribution:")
    print(events["event_type"].value_counts())
    print("session minutes:", events.loc[events.event_type == "session", "value"].describe().round(2).to_dict())
    print("plan_change values:", events.loc[events.event_type == "plan_change", "value"].value_counts().to_dict())
    print("events per customer:", events.groupby("customer_id").size().describe().round(1).to_dict())

    print("\n--- basic integrity / leakage checks")
    print("duplicate customer_id in train:", int(train["customer_id"].duplicated().sum()),
          "| in test:", int(test["customer_id"].duplicated().sum()),
          "| customers in BOTH train and test:", len(set(train["customer_id"]) & set(test["customer_id"])))
    print("exact duplicate event rows:", int(events.duplicated().sum()))
    print("events whose customer is not in train/test:", int((~events["customer_id"].isin(set(train["customer_id"]) | set(test["customer_id"]))).sum()))
    print("signup after snapshot:", int((train["signup_date"] > train["snapshot_date"]).sum()), "(train),",
          int((test["signup_date"] > test["snapshot_date"]).sum()), "(test)")
    sm = events.loc[events.event_type == "session", "value"]
    print("session minutes <=0:", int((sm <= 0).sum()), "| missing:", int(sm.isna().sum()), "| > 99.9th pct x5:", int((sm > 5 * sm.quantile(.999)).sum()))
    if pd.api.types.is_numeric_dtype(train["customer_id"]):
        print("corr(customer_id, churn):", round(float(np.corrcoef(train["customer_id"], train["churn"])[0, 1]), 4), "(should be ~0)")
    print("sample_submission columns:", list(ss.columns), "| rows:", len(ss), "| ids match test:", set(ss.iloc[:, 0]) == set(test["customer_id"]))

# =============================================================================
# 3. CUSTOMER FEATURES
# =============================================================================
CAT_RAW = ["plan", "billing_cycle", "region", "age_band", "acquisition_channel", "payment_method"]

def _cycle_days(s):
    s = s.astype(str).str.lower()
    return np.where(s.str.contains("year|annual"), 365, np.where(s.str.contains("quarter"), 90, np.where(s.str.contains("semi|half"), 182, 30)))

def build_customer_features(df):
    """df = concatenated train+test with row_id. Only uses columns of the row itself plus label-free
    plan-level price medians and a CUMULATIVE (by snapshot month) referral-code count."""
    out = pd.DataFrame(index=df.index)
    ten = (df["snapshot_date"] - df["signup_date"]).dt.total_seconds() / 86400
    out["tenure_days"] = ten; out["tenure_months"] = ten / 30.4375
    out["obs_days"] = np.minimum(ten, max(WINDOWS))                          # how much event history can exist
    for c in CAT_RAW: out[c] = df[c].astype(str).fillna("NA")
    out["monthly_price"] = df["monthly_price"]; out["num_devices"] = df["num_devices"]
    out["family_sharing"] = df["family_sharing"].astype(str).str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}).astype(float)
    ref = df["referral_code"]
    out["has_referral"] = (ref.notna() & (ref.astype(str).str.strip() != "")).astype(int)
    # referral code count using ONLY rows with snapshot month <= this row's month (no future rows)
    mon = df["snapshot_date"].dt.to_period("M"); cnt = pd.Series(np.nan, index=df.index)
    for m in sorted(mon.unique()):
        sub = ref[(mon <= m) & ref.notna()].value_counts()
        idx = mon == m
        cnt[idx] = ref[idx].map(sub).values
    out["referral_code_count_so_far"] = cnt.where(out["has_referral"] == 1)
    if ref.nunique() <= 300:                                                  # low-card -> usable as categorical
        out["referral_code"] = ref.astype(str).fillna("NA")
    # interactions
    out["price_per_device"] = out["monthly_price"] / out["num_devices"].replace(0, np.nan)
    out["price_vs_plan_median"] = out["monthly_price"] - out.groupby(df["plan"])["monthly_price"].transform("median")
    cyc = _cycle_days(df["billing_cycle"])
    out["cycle_days"] = cyc
    out["days_to_renewal"] = cyc - (ten % cyc)
    out["is_first_cycle"] = (ten < cyc).astype(int)
    out["tenure_in_cycles"] = ten / cyc
    out["plan_x_cycle"] = df["plan"].astype(str) + "|" + df["billing_cycle"].astype(str)
    out["plan_x_payment"] = df["plan"].astype(str) + "|" + df["payment_method"].astype(str)
    out["channel_x_plan"] = df["acquisition_channel"].astype(str) + "|" + df["plan"].astype(str)
    out["family_x_devices"] = out["family_sharing"].fillna(0) * out["num_devices"].fillna(0)
    cats = CAT_RAW + (["referral_code"] if "referral_code" in out else []) + ["plan_x_cycle", "plan_x_payment", "channel_x_plan"]
    return out, cats

# =============================================================================
# 4. EVENT FEATURES  (snapshot-aware, vectorised)
# =============================================================================
def _eff_snapshot(snap):
    return snap["snapshot_date"] + (pd.Timedelta(days=1) if SNAPSHOT_END_OF_DAY else pd.Timedelta(0))

def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan); np.divide(a, b, out=out, where=(b > 0) & np.isfinite(b)); return out

def build_event_features(snap, events, verbose=True):
    """
    snap   : DataFrame[row_id, customer_id, snapshot_date]   (one row per customer snapshot)
    events : DataFrame[customer_id, timestamp, event_type, value]
    Returns (features indexed by row_id, group dict, diagnostics).
    LEAKAGE GUARD: every event is joined to the snapshot of ITS customer row and kept only if
    timestamp <= effective snapshot; an assertion re-checks this after filtering.
    """
    snap = snap[["row_id", "customer_id", "snapshot_date"]].copy()
    snap["eff"] = _eff_snapshot(snap)
    m = events[["customer_id", "timestamp", "event_type", "value"]].merge(snap[["row_id", "customer_id", "eff"]], on="customer_id", how="inner")
    age_all = ((m["eff"] - m["timestamp"]).dt.total_seconds() / 86400).values.astype("float32")
    keep = age_all >= 0
    diag = {"events_joined": int(len(m)), "events_after_snapshot_dropped": int((~keep).sum()),
            "of_which_within_24h_after_snapshot": int(((age_all < 0) & (age_all >= -1)).sum()),
            "max_age_days": float(age_all.max()) if len(age_all) else np.nan}
    m = m.loc[keep, ["row_id", "timestamp", "event_type", "value", "eff"]].reset_index(drop=True)
    assert (m["timestamp"] <= m["eff"]).all(), "FUTURE EVENT LEAKED INTO FEATURES"
    et = m["event_type"].astype(str).values; val = m["value"].astype(float).values
    m["age"] = age_all[keep]
    m["is_session"] = (et == "session").astype("float32"); m["is_ticket"] = (et == "support_ticket").astype("float32")
    m["is_fail"] = (et == "payment_failed").astype("float32"); m["is_pc"] = (et == "plan_change").astype("float32")
    m["up"] = ((et == "plan_change") & (val > 0)).astype("float32"); m["down"] = ((et == "plan_change") & (val < 0)).astype("float32")
    m["minutes"] = np.where(et == "session", val, np.nan).astype("float32")
    m["pc_val"] = np.where(et == "plan_change", val, 0).astype("float32")
    ids = snap["row_id"].values
    F = pd.DataFrame(index=pd.Index(ids, name="row_id"))
    groups = {"counts": [], "recency": [], "windows": []}
    def add(name, ser, grp, fill=None):
        F[name] = ser.reindex(ids).values if fill is None else ser.reindex(ids).fillna(fill).values; groups[grp].append(name)

    # ---- totals over all available (non-future) history
    g = m.groupby("row_id")
    add("total_events", g.size(), "counts", 0)
    add("total_sessions", g["is_session"].sum(), "counts", 0)
    add("total_support_tickets", g["is_ticket"].sum(), "counts", 0)
    add("total_payment_failures", g["is_fail"].sum(), "counts", 0)
    add("total_plan_changes", g["is_pc"].sum(), "counts", 0)
    add("upgrade_count", g["up"].sum(), "counts", 0)
    add("downgrade_count", g["down"].sum(), "counts", 0)
    add("net_plan_change", g["pc_val"].sum(), "counts", 0)
    s = m.loc[m["is_session"] == 1]; gs = s.groupby("row_id")["minutes"]
    add("total_session_minutes", gs.sum(), "counts", 0)
    add("mean_session_minutes", gs.mean(), "counts"); add("median_session_minutes", gs.median(), "counts")
    add("max_session_minutes", gs.max(), "counts"); add("min_session_minutes", gs.min(), "counts"); add("std_session_minutes", gs.std(), "counts")
    add("active_session_days_total", s.assign(d=np.floor(s["age"])).groupby("row_id")["d"].nunique(), "counts", 0)

    # ---- recency (days since last event of each type, relative to snapshot)
    add("days_since_last_event", g["age"].min(), "recency")
    for nm, col in (("session", "is_session"), ("support_ticket", "is_ticket"), ("payment_failure", "is_fail"), ("plan_change", "is_pc")):
        add(f"days_since_last_{nm}", m.loc[m[col] == 1].groupby("row_id")["age"].min(), "recency")
    add("days_since_first_event", g["age"].max(), "recency")

    # ---- windowed features
    for w in WINDOWS:
        sw = m.loc[m["age"] <= w]; gw = sw.groupby("row_id")
        add(f"sessions_{w}d", gw["is_session"].sum(), "windows", 0)
        add(f"support_tickets_{w}d", gw["is_ticket"].sum(), "windows", 0)
        add(f"payment_failures_{w}d", gw["is_fail"].sum(), "windows", 0)
        add(f"plan_changes_{w}d", gw["is_pc"].sum(), "windows", 0)
        add(f"upgrades_{w}d", gw["up"].sum(), "windows", 0)
        add(f"downgrades_{w}d", gw["down"].sum(), "windows", 0)
        ssw = sw.loc[sw["is_session"] == 1]; gsw = ssw.groupby("row_id")["minutes"]
        add(f"session_minutes_{w}d", gsw.sum(), "windows", 0)
        add(f"mean_session_minutes_{w}d", gsw.mean(), "windows"); add(f"max_session_minutes_{w}d", gsw.max(), "windows")
        add(f"active_session_days_{w}d", ssw.assign(d=np.floor(ssw["age"])).groupby("row_id")["d"].nunique(), "windows", 0)
        if verbose: log(f"   window {w:3d}d done")

    # ---- ratios / trends (division by zero -> NaN, never inf)
    def ratio(name, a, b): F[name] = _safe_div(F[a], F[b]); groups["windows"].append(name)
    def diff(name, a, b): F[name] = F[a] - F[b]; groups["windows"].append(name)
    ratio("ratio_sessions_7d_30d", "sessions_7d", "sessions_30d")
    ratio("ratio_sessions_14d_60d", "sessions_14d", "sessions_60d")
    ratio("ratio_sessions_30d_90d", "sessions_30d", "sessions_90d")
    ratio("ratio_session_minutes_7d_30d", "session_minutes_7d", "session_minutes_30d")
    ratio("ratio_session_minutes_30d_90d", "session_minutes_30d", "session_minutes_90d")
    ratio("ratio_tickets_30d_total", "support_tickets_30d", "total_support_tickets")
    ratio("ratio_failures_30d_total", "payment_failures_30d", "total_payment_failures")
    F["sessions_prev30d"] = F["sessions_60d"] - F["sessions_30d"]; groups["windows"].append("sessions_prev30d")
    F["session_minutes_prev30d"] = F["session_minutes_60d"] - F["session_minutes_30d"]; groups["windows"].append("session_minutes_prev30d")
    F["sessions_prev7d"] = F["sessions_14d"] - F["sessions_7d"]; groups["windows"].append("sessions_prev7d")
    diff("trend_sessions_30d_vs_prev30d", "sessions_30d", "sessions_prev30d")
    diff("trend_sessions_7d_vs_prev7d", "sessions_7d", "sessions_prev7d")
    ratio("ratio_sessions_30d_vs_prev30d", "sessions_30d", "sessions_prev30d")
    ratio("ratio_session_minutes_30d_vs_prev30d", "session_minutes_30d", "session_minutes_prev30d")
    ratio("ratio_sessions_7d_vs_prev7d", "sessions_7d", "sessions_prev7d")
    F["net_plan_change_30d"] = F["upgrades_30d"] - F["downgrades_30d"]; groups["windows"].append("net_plan_change_30d")
    return F, groups, diag

def _cache_key(files, extra):
    h = hashlib.md5()
    for f in files:
        st = os.stat(os.path.join(DATA_DIR, f)); h.update(f"{f}{st.st_size}{int(st.st_mtime)}".encode())
    h.update(str((WINDOWS, SNAPSHOT_END_OF_DAY, extra)).encode()); return h.hexdigest()[:12]

def build_features(train, test, events):
    """Returns (full, feature_cols, cat_cols, groups). `full` = train+test stacked with meta columns."""
    log("Building features ...")
    full = pd.concat([train.assign(is_train=1), test.assign(is_train=0, churn=np.nan)], ignore_index=True)
    full["row_id"] = np.arange(len(full)); full["month"] = full["snapshot_date"].dt.to_period("M")
    cust, cat_cols = build_customer_features(full)
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(["train.csv", "test.csv", "events.csv"], "v3")
    cpath = os.path.join(CACHE_DIR, f"event_feats_{key}.pkl")
    if os.path.exists(cpath) and not FORCE_REBUILD:
        log(f"   loading cached event features: {cpath}"); F, groups, diag = pd.read_pickle(cpath)
    else:
        t = time.time(); F, groups, diag = build_event_features(full, events)
        pd.to_pickle((F, groups, diag), cpath); log(f"   event features built in {time.time() - t:.1f}s -> cached")
    print("   event-join diagnostics:", diag)
    ev_f = F.reset_index(drop=True)
    X = pd.concat([cust.reset_index(drop=True), ev_f], axis=1)
    # derived rates need both blocks
    X["sessions_per_obs_day_30d"] = _safe_div(X["sessions_30d"], np.minimum(X["obs_days"], 30)); groups["windows"].append("sessions_per_obs_day_30d")
    X["session_minutes_per_obs_day_90d"] = _safe_div(X["session_minutes_90d"], np.minimum(X["obs_days"], 90)); groups["windows"].append("session_minutes_per_obs_day_90d")
    groups["customer"] = list(cust.columns)
    for c in cat_cols:                                    # shared category set across train/val/test
        X[c] = pd.Categorical(X[c].fillna("nan").astype(str), categories=sorted(X[c].fillna("nan").astype(str).unique()))
    const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const:
        print("   dropping constant columns:", const); X = X.drop(columns=const)
        for k in groups: groups[k] = [c for c in groups[k] if c not in const]
        cat_cols = [c for c in cat_cols if c not in const]
    meta = full[["row_id", "customer_id", "snapshot_date", "month", "is_train", "churn"]].reset_index(drop=True)
    full = pd.concat([meta, X], axis=1)
    feature_cols = list(X.columns)
    log(f"   {len(feature_cols)} features: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    return full, feature_cols, cat_cols, groups

# =============================================================================
# 5. LEAKAGE CHECKS
# =============================================================================
def run_leakage_checks(full, feature_cols, events, groups, n_rows=300):
    print("\n" + "=" * 70 + "\nLEAKAGE CHECKS\n" + "=" * 70)
    rng = np.random.RandomState(RANDOM_SEED)
    # (a) brute-force recomputation for random rows (loops are fine for a few hundred rows)
    snap = full[["row_id", "customer_id", "snapshot_date"]]
    sample = snap.iloc[rng.choice(len(snap), size=min(n_rows, len(snap)), replace=False)]
    evg = {c: g for c, g in events[events["customer_id"].isin(set(sample["customer_id"]))].groupby("customer_id")}
    bad = 0; checked = 0
    for _, r in sample.iterrows():
        e = evg.get(r["customer_id"]); eff = r["snapshot_date"] + (pd.Timedelta(days=1) if SNAPSHOT_END_OF_DAY else pd.Timedelta(0))
        e = e[e["timestamp"] <= eff] if e is not None else events.iloc[0:0].copy()
        age = (eff - e["timestamp"]).dt.total_seconds() / 86400
        s = e["event_type"].astype(str) == "session"
        exp = {"total_events": len(e), "sessions_30d": int((s & (age <= 30)).sum()),
               "session_minutes_30d": float(e.loc[s & (age <= 30), "value"].sum()),
               "payment_failures_60d": int(((e["event_type"].astype(str) == "payment_failed") & (age <= 60)).sum()),
               "net_plan_change": float(e.loc[e["event_type"].astype(str) == "plan_change", "value"].sum()),
               "days_since_last_support_ticket": float(age[e["event_type"].astype(str) == "support_ticket"].min()) if (e["event_type"].astype(str) == "support_ticket").any() else np.nan}
        got = full.loc[full["row_id"] == r["row_id"], list(exp)].iloc[0]
        for k, v in exp.items():
            checked += 1
            if not (np.isclose(float(got[k]), v, equal_nan=True, rtol=1e-4, atol=1e-3)): bad += 1
    print(f"(a) brute-force recomputation: {checked} values checked on {len(sample)} rows -> mismatches: {bad}")
    assert bad == 0, "Vectorised features disagree with brute-force recomputation!"
    # (b) poison test: inject extreme events AFTER each snapshot; features must not change
    sub = sample.reset_index(drop=True).copy(); sub["row_id"] = np.arange(len(sub))
    ev_sub = events[events["customer_id"].isin(set(sub["customer_id"]))]
    base, _, _ = build_event_features(sub, ev_sub, verbose=False)
    eff = _eff_snapshot(sub) + pd.Timedelta(hours=1 if not SNAPSHOT_END_OF_DAY else 1)
    px = pd.concat([pd.DataFrame({"customer_id": sub["customer_id"], "timestamp": eff, "event_type": t, "value": v})
                    for t, v in (("session", 99999.0), ("payment_failed", np.nan), ("support_ticket", np.nan), ("plan_change", -1.0))])
    px["event_type"] = px["event_type"].astype(ev_sub["event_type"].dtype) if str(ev_sub["event_type"].dtype) == "category" else px["event_type"]
    ev_p = pd.concat([ev_sub.assign(event_type=ev_sub["event_type"].astype(str)), px.assign(event_type=px["event_type"].astype(str))], ignore_index=True)
    poisoned, _, _ = build_event_features(sub, ev_p, verbose=False)
    same = np.allclose(base.values.astype(float), poisoned.values.astype(float), equal_nan=True)
    print(f"(b) poison test (extreme events injected after snapshot): features unchanged -> {same}")
    assert same, "Features changed when post-snapshot events were added -> FUTURE LEAK"
    # (c) suspicious single features (univariate AUC on training rows)
    tr = full[full["is_train"] == 1]; y = tr["churn"].astype(int).values; rows = []
    for c in feature_cols:
        if str(full[c].dtype) == "category": continue
        x = tr[c].astype(float); x = x.fillna(x.median())
        if x.nunique() < 2: continue
        a = roc_auc_score(y, x); rows.append((c, max(a, 1 - a), a))
    sf = pd.DataFrame(rows, columns=["feature", "auc_abs", "auc_raw"]).sort_values("auc_abs", ascending=False)
    print("(c) strongest single numeric features (train, univariate):\n", sf.head(15).round(4).to_string(index=False))
    print("    suspicious (>0.85):", sf.loc[sf.auc_abs > 0.85, "feature"].tolist() or "none")
    # (d) NaN / inf scan
    num = full[[c for c in feature_cols if str(full[c].dtype) != "category"]].astype(float)
    print("(d) inf values:", int(np.isinf(num.values).sum()), "| columns >50% NaN:", [c for c in num.columns if num[c].isna().mean() > .5])
    return sf

# =============================================================================
# 6. TEMPORAL SPLITS
# =============================================================================
def make_temporal_splits(full):
    tr = full[full["is_train"] == 1]; months = sorted(tr["month"].unique())
    splits = []
    for i in range(1, len(months)):
        va_idx = tr.index[tr["month"] == months[i]].values
        tr_mask = tr["month"].isin(months[:i])
        if GAP_DAYS:
            tr_mask &= tr["snapshot_date"] < (tr.loc[va_idx, "snapshot_date"].min() - pd.Timedelta(days=GAP_DAYS))
        tr_idx = tr.index[tr_mask].values
        assert full.loc[tr_idx, "snapshot_date"].max() < full.loc[va_idx, "snapshot_date"].min(), "train/valid temporal overlap"
        splits.append(dict(name=f"{'+'.join(str(m)[5:] for m in months[:i])} -> {months[i]}", train_idx=tr_idx, valid_idx=va_idx,
                           valid_month=str(months[i]), primary=(i == len(months) - 1)))
    log("Temporal splits:"); [print(f"   {s['name']:24s} train={len(s['train_idx']):6d} valid={len(s['valid_idx']):6d}{'  <- PRIMARY' if s['primary'] else ''}") for s in splits]
    return splits

# =============================================================================
# 7. MODELS
# =============================================================================
def _inner_split(n_idx, y):
    return train_test_split(np.arange(n_idx), test_size=ES_HOLDOUT, stratify=y, random_state=RANDOM_SEED)

def _cb_frame(X, cats):
    X = X.copy()
    for c in cats: X[c] = X[c].astype(str)
    return X

def _train_auc(predict_proba, Xtr, ytr, a, n=20000):
    """In-sample AUC on (up to) 20k of the rows the model was fitted on -- only a gauge of overfitting."""
    sub = a[:n]; return roc_auc_score(ytr[sub], predict_proba(Xtr.iloc[sub])[:, 1])

def train_lightgbm(Xtr, ytr, Xva, yva_unused, cats, params=None):
    import lightgbm as lgb
    p = dict(LIGHTGBM_PARAMS, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    m = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, **p)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="auc", callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    return m, m.predict_proba(Xva)[:, 1], _train_auc(m.predict_proba, Xtr, ytr, a), m.best_iteration_

def train_xgboost(Xtr, ytr, Xva, yva_unused, cats, params=None):
    import xgboost as xgb
    p = dict(XGBOOST_PARAMS, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    m = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, early_stopping_rounds=ES_ROUNDS, **p)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], verbose=False)
    return m, m.predict_proba(Xva)[:, 1], _train_auc(m.predict_proba, Xtr, ytr, a), m.best_iteration

def train_catboost(Xtr, ytr, Xva, yva_unused, cats, params=None):
    from catboost import CatBoostClassifier
    p = dict(CATBOOST_PARAMS, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    Xt, Xv = _cb_frame(Xtr, cats), _cb_frame(Xva, cats)
    m = CatBoostClassifier(random_seed=RANDOM_SEED, thread_count=NUM_THREADS, cat_features=cats, **p)
    m.fit(Xt.iloc[a], ytr[a], eval_set=(Xt.iloc[b], ytr[b]), use_best_model=True)
    return m, m.predict_proba(Xv)[:, 1], _train_auc(lambda X: m.predict_proba(_cb_frame(X, cats)), Xtr, ytr, a), m.get_best_iteration()

TRAINERS = {"catboost": train_catboost, "lightgbm": train_lightgbm, "xgboost": train_xgboost}

def evaluate_model(name, full, cols, cats, splits, only_primary=False, params=None, verbose=True):
    """Trains `name` on each split's train part, scores ROC-AUC on the validation month. Returns dict."""
    fn = TRAINERS[name]; cats = [c for c in cats if c in cols]; res = dict(aucs={}, oof=[], models={}, time=0.0, train_auc=[], iters=[])
    for sp in splits:
        if only_primary and not sp["primary"]: continue
        t = time.time(); Xtr = full.loc[sp["train_idx"], cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
        Xva = full.loc[sp["valid_idx"], cols]; yva = full.loc[sp["valid_idx"], "churn"].astype(int).values
        m, pv, tr_auc, bi = fn(Xtr, ytr, Xva, yva, cats, params)
        auc = roc_auc_score(yva, pv); res["aucs"][sp["valid_month"]] = auc; res["models"][sp["valid_month"]] = m
        res["train_auc"].append(tr_auc); res["iters"].append(bi)
        res["oof"].append(pd.DataFrame({"row_id": full.loc[sp["valid_idx"], "row_id"].values, "fold": sp["valid_month"], "y": yva, "pred": pv}))
        res["time"] += time.time() - t
        if verbose: print(f"      {name:9s} {sp['name']:22s} AUC={auc:.5f}  train AUC={tr_auc:.4f}, iters={bi}, {time.time() - t:.0f}s)")
    a = np.array(list(res["aucs"].values())); res["mean"], res["std"] = a.mean(), a.std()
    prim = [sp["valid_month"] for sp in splits if sp["primary"]][0]; res["primary"] = res["aucs"].get(prim, np.nan)
    return res

def _ladder_model():
    for m in (LADDER_MODEL, "lightgbm", "catboost", "xgboost"):
        if HAVE.get(m): return m
    sys.exit("No boosting library installed. pip install lightgbm catboost xgboost")

# =============================================================================
# 8. EXPERIMENTS
# =============================================================================
def run_experiments(full, feature_cols, cat_cols, groups, splits):
    rows = []; oof_frames = {}; models = {}
    lad = _ladder_model(); lp = LADDER_LGBM_PARAMS if lad == "lightgbm" else None
    cust, cnt, rec, win = groups["customer"], groups["counts"], groups["recency"], groups["windows"]
    ladder = [("1. Customer baseline", cust), ("2. + Event counts", cust + cnt), ("3. + Recency", cust + cnt + rec),
              ("4. + Window features", cust + cnt + rec + win)]
    print("\n" + "=" * 70 + f"\nEXPERIMENT LADDER ({lad}, rolling temporal validation)\n" + "=" * 70)
    for nm, cols in ladder:
        log(nm + f"  [{len(cols)} features]")
        r = evaluate_model(lad, full, cols, cat_cols, splits, params=lp, verbose=True)
        rows.append(dict(experiment=nm, model=lad, features=len(cols), primary_auc=r["primary"], rolling_mean=r["mean"], rolling_std=r["std"],
                         train_auc=float(np.mean(r["train_auc"])), time_s=round(r["time"], 1), **{f"auc_{k}": v for k, v in r["aucs"].items()}))
    print("\n" + "=" * 70 + "\nFULL FEATURE SET: CatBoost / LightGBM / XGBoost\n" + "=" * 70)
    full_cols = cust + cnt + rec + win; names = {"catboost": "5. CatBoost", "lightgbm": "6. LightGBM", "xgboost": "7. XGBoost"}
    preds = {}
    for k in ("catboost", "lightgbm", "xgboost"):
        if not HAVE[k]: log(f"{names[k]} skipped (not installed)"); continue
        log(names[k] + f"  [{len(full_cols)} features]")
        r = evaluate_model(k, full, full_cols, cat_cols, splits, only_primary=not ROLLING_FOR_BOOSTERS)
        rows.append(dict(experiment=names[k], model=k, features=len(full_cols), primary_auc=r["primary"], rolling_mean=r["mean"], rolling_std=r["std"],
                         train_auc=float(np.mean(r["train_auc"])), time_s=round(r["time"], 1), **{f"auc_{kk}": v for kk, v in r["aucs"].items()}))
        preds[k] = pd.concat(r["oof"], ignore_index=True); models[k] = r["models"]
    # ---- ensemble (rank average within each validation month; weights equal)
    ens_rows = None
    if len(preds) >= 2:
        base = list(preds.values())[0][["row_id", "fold", "y"]].copy()
        for k, d in preds.items(): base[k] = d["pred"].values
        for k in preds: base[k + "_rank"] = base.groupby("fold")[k].rank(pct=True)
        base["ensemble"] = base[[k + "_rank" for k in preds]].mean(axis=1)
        aucs = {f: roc_auc_score(g["y"], g["ensemble"]) for f, g in base.groupby("fold")}
        prim = [s["valid_month"] for s in splits if s["primary"]][0]
        rows.append(dict(experiment="8. Ensemble (rank avg)", model="+".join(preds), features=len(full_cols), primary_auc=aucs[prim],
                         rolling_mean=np.mean(list(aucs.values())), rolling_std=np.std(list(aucs.values())), time_s=0.0, **{f"auc_{k}": v for k, v in aucs.items()}))
        pc = base[list(preds)].corr(method="spearman"); print("\nValidation-prediction rank correlation between models:\n", pc.round(4))
        ens_rows = base.drop(columns=[k + "_rank" for k in preds])
    else:
        ens_rows = list(preds.values())[0].rename(columns={"pred": list(preds)[0]}) if preds else None
    res = pd.DataFrame(rows)
    return res, ens_rows, models

# =============================================================================
# 9. ABLATION, IMPORTANCE, SHIFT
# =============================================================================
ABLATIONS = {
    "payment (failures)": lambda c, g: "payment_fail" in c,
    "sessions (counts+minutes)": lambda c, g: "session" in c,
    "support tickets": lambda c, g: "support_ticket" in c or "tickets" in c,
    "plan changes (up/down)": lambda c, g: any(k in c for k in ("plan_change", "upgrade", "downgrade")),
    "recency (days_since_*)": lambda c, g: c.startswith("days_since"),
    "tenure/renewal": lambda c, g: any(k in c for k in ("tenure", "renewal", "obs_days", "first_cycle", "cycle_days")),
    "plan & price": lambda c, g: c in g["customer"] and any(k in c for k in ("plan", "price", "billing")) and "plan_change" not in c,
    "ALL event features": lambda c, g: c not in g["customer"],
}

def run_group_ablation(full, feature_cols, cat_cols, groups, splits):
    lad = _ladder_model(); lp = LADDER_LGBM_PARAMS if lad == "lightgbm" else None
    print("\n" + "=" * 70 + f"\nGROUP ABLATION ({lad}): drop one family from the FULL feature set\n" + "=" * 70)
    base = evaluate_model(lad, full, feature_cols, cat_cols, splits, params=lp, verbose=False)
    rows = [dict(dropped="(none: full set)", n_dropped=0, primary_auc=base["primary"], rolling_mean=base["mean"], delta_rolling=0.0)]
    for nm, pred in ABLATIONS.items():
        drop = [c for c in feature_cols if pred(c, groups)]
        if not drop: continue
        r = evaluate_model(lad, full, [c for c in feature_cols if c not in drop], cat_cols, splits, params=lp, verbose=False)
        rows.append(dict(dropped=nm, n_dropped=len(drop), primary_auc=r["primary"], rolling_mean=r["mean"], delta_rolling=r["mean"] - base["mean"]))
        log(f"   dropped {nm:28s} ({len(drop):3d} cols): rolling AUC {r['mean']:.5f}  (delta {r['mean'] - base['mean']:+.5f})")
    return pd.DataFrame(rows)

def model_importance(name, model, cols):
    if name == "lightgbm": v = model.booster_.feature_importance("gain")
    elif name == "xgboost": s = model.get_booster().get_score(importance_type="gain"); v = np.array([s.get(c, 0.0) for c in cols])
    else: v = model.get_feature_importance()
    v = np.asarray(v, dtype=float); return pd.DataFrame({"feature": cols, "importance": v, "importance_pct": 100 * v / v.sum()}).sort_values("importance", ascending=False)

def report_importance(best, models, full, feature_cols, groups, splits):
    prim = [s["valid_month"] for s in splits if s["primary"]][0]
    imp = model_importance(best, models[best][prim], feature_cols); os.makedirs(OUT_DIR, exist_ok=True)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    print("\n" + "=" * 70 + f"\nTOP 30 FEATURES ({best}, primary split)\n" + "=" * 70)
    print(imp.head(30).round(3).to_string(index=False))
    fam = {k: imp.loc[imp.feature.apply(lambda c: p(c, groups)), "importance_pct"].sum() for k, p in ABLATIONS.items() if k != "ALL event features"}
    print("\nImportance share by feature family (families overlap; %):"); [print(f"   {k:28s} {v:6.1f}") for k, v in sorted(fam.items(), key=lambda x: -x[1])]
    if USE_SHAP and HAVE["shap"]:
        try:
            import shap
            sp = [s for s in splits if s["primary"]][0]; X = full.loc[sp["valid_idx"], feature_cols].sample(min(5000, len(sp["valid_idx"])), random_state=RANDOM_SEED)
            m = models[best][prim]
            Xs = _cb_frame(X, [c for c in feature_cols if str(X[c].dtype) == "category"]) if best == "catboost" else X
            sv = shap.TreeExplainer(m).shap_values(Xs); sv = sv[1] if isinstance(sv, list) else sv
            sh = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
            sh.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False); print("\nTop 15 by mean |SHAP|:\n", sh.head(15).round(4).to_string(index=False))
        except Exception as e:
            print("SHAP skipped:", repr(e)[:200])
    return imp

def shift_check(full, feature_cols, cat_cols):
    """Adversarial validation: last train month vs test. High AUC = test looks different (expected somewhat: later months)."""
    if not HAVE["lightgbm"]: return
    import lightgbm as lgb
    last = sorted(full.loc[full.is_train == 1, "month"].unique())[-1]
    d = full[((full.is_train == 1) & (full.month == last)) | (full.is_train == 0)]
    X = d[feature_cols]; y = (d["is_train"] == 0).astype(int).values; oof = np.zeros(len(d)); imp = pd.Series(0.0, index=feature_cols)
    from sklearn.model_selection import StratifiedKFold
    for a, b in StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.7, verbose=-1, n_jobs=NUM_THREADS, random_state=RANDOM_SEED)
        m.fit(X.iloc[a], y[a]); oof[b] = m.predict_proba(X.iloc[b])[:, 1]; imp += pd.Series(m.booster_.feature_importance("gain"), index=feature_cols)
    print("\n" + "=" * 70 + f"\nTRAIN({last}) vs TEST shift check: adversarial AUC = {roc_auc_score(y, oof):.4f}  (0.5 = identical)\n" + "=" * 70)
    print("most shifted features:\n", (100 * imp / imp.sum()).sort_values(ascending=False).head(10).round(2).to_string())

def fit_full_and_predict(best, full, feature_cols, cat_cols, ss):
    tr, te = full[full.is_train == 1], full[full.is_train == 0]
    m, pt, _, bi = TRAINERS[best](tr[feature_cols], tr["churn"].astype(int).values, te[feature_cols], None, [c for c in cat_cols if c in feature_cols])
    sub = pd.DataFrame({ss.columns[0]: te["customer_id"].values, ss.columns[1]: pt})
    assert len(sub) == len(ss) and not sub.isna().any().any(); path = os.path.join(OUT_DIR, "submission_local_test.csv"); sub.to_csv(path, index=False)
    log(f"wrote {path} ({best}, {bi} iters) -- local sanity only")

# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True); np.random.seed(RANDOM_SEED)
    print(f"Libraries available: {HAVE} | threads={NUM_THREADS} | seed={RANDOM_SEED}")
    train, test, events, ss = load_data()
    inspect_data(train, test, events, ss)
    full, feature_cols, cat_cols, groups = build_features(train, test, events)
    run_leakage_checks(full, feature_cols, events, groups)
    splits = make_temporal_splits(full)
    if RUN_SHIFT_CHECK: shift_check(full, feature_cols, cat_cols)
    results, ens_frame, models = run_experiments(full, feature_cols, cat_cols, groups, splits)
    results.to_csv(os.path.join(OUT_DIR, "experiment_results.csv"), index=False)
    # ---- OOF file (each validation row appears once; Jan has no OOF because it is never validated)
    if ens_frame is not None:
        oof = ens_frame.merge(full[["row_id", "customer_id", "snapshot_date"]], on="row_id").rename(columns={"y": "churn"})
        oof.drop(columns="row_id").to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)
    # ---- best model + importance
    best_row = results.loc[results["primary_auc"].idxmax()]
    best_single = results[results.experiment.str[0].isin(list("567"))].sort_values("primary_auc").iloc[-1]["model"]
    imp = report_importance(best_single, models, full, feature_cols, groups, splits)
    abl = run_group_ablation(full, feature_cols, cat_cols, groups, splits) if RUN_GROUP_ABLATION else None
    if abl is not None:
        abl.to_csv(os.path.join(OUT_DIR, "group_ablation.csv"), index=False); print(abl.round(5).to_string(index=False))
    if WRITE_TEST_SUBMISSION: fit_full_and_predict(best_single, full, feature_cols, cat_cols, ss)
    # ---- final summary
    prim_month = [s["valid_month"] for s in splits if s["primary"]][0]; first = [s["valid_month"] for s in splits][0]
    print("\n" + "=" * 50 + "\nSTREAMBOX CHURN LOCAL EXPERIMENT RESULTS\n" + "=" * 50)
    print("\nValidation strategy:\n  Primary : train on all months before the last, validate on", prim_month,
          "\n  Rolling : each month validated on a model trained ONLY on earlier months\n  Early stopping uses a random hold-out of the TRAIN part only.")
    print(f"\n{'Experiment':30s}{'ROC-AUC (' + prim_month + ')':>18s}{'rolling mean':>14s}{'std':>8s}{'train AUC':>11s}{'time(s)':>9s}")
    print("-" * 90)
    for _, r in results.iterrows():
        tra = f"{r.train_auc:11.4f}" if pd.notna(r.get("train_auc", np.nan)) else f"{'-':>11s}"
        print(f"{r.experiment:30s}{r.primary_auc:18.4f}{r.rolling_mean:14.4f}{r.rolling_std:8.4f}{tra}{r.time_s:9.0f}")
    print("\nPer-fold AUC:"); print(results[["experiment"] + [c for c in results.columns if c.startswith("auc_")]].round(4).to_string(index=False))
    print(f"\nBest overall (primary split): {best_row.experiment}  ROC-AUC = {best_row.primary_auc:.4f}")
    print(f"Best single model: {best_single}")
    lad = results.iloc[0]["primary_auc"]; ev = results.iloc[3]["primary_auc"]
    print(f"Event data effect: customer-only {lad:.4f} -> full ladder {ev:.4f}  (delta {ev - lad:+.4f})")
    print("=" * 50)
    print(f"Saved to {OUT_DIR}/: experiment_results.csv, oof_predictions.csv, feature_importance.csv, group_ablation.csv")
    log("done")

if __name__ == "__main__":
    main()