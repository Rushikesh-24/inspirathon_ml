#!/usr/bin/env python
# %% [markdown]
# # StreamBox Churn Prediction — Submission #1
#
# **Metric**: ROC-AUC  
# **Pipeline**: V3 validated ensemble (CatBoost 70% + LightGBM 25% + XGBoost 5%)  
# **Validation**: Temporal out-of-time (Jan→Feb, Jan+Feb→Mar, Jan+Feb+Mar→Apr)
#
# This notebook is fully self-contained and uses only competition-provided data.

# %% [markdown]
# ## Cell 1: Imports and Configuration

# %%
import os, time, warnings, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
T0 = time.time()
def log(msg=""): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# Paths — auto-detect Kaggle vs local
if os.path.exists("/kaggle/input/competitions/inspirathon"):
    INPUT_DIR = "/kaggle/input/competitions/inspirathon"
    OUTPUT_DIR = "/kaggle/working"
elif os.path.exists("data"):
    INPUT_DIR = "data"
    OUTPUT_DIR = "outputs_submission"
else:
    INPUT_DIR = "."
    OUTPUT_DIR = "outputs_submission"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED   = 42
WINDOWS       = [7, 14, 30, 60, 90, 120]
SNAPSHOT_END_OF_DAY = False
ES_HOLDOUT    = 0.15
ES_ROUNDS     = 150
NUM_THREADS   = os.cpu_count() or 1

# V3-tuned hyperparameters
CATBOOST_PARAMS = dict(
    iterations=5000, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
    random_strength=0.5, bagging_temperature=0.3,
    loss_function="Logloss", eval_metric="AUC",
    od_type="Iter", od_wait=200, verbose=0
)
LIGHTGBM_PARAMS = dict(
    n_estimators=5000, learning_rate=0.02, num_leaves=31,
    min_child_samples=30, max_depth=-1, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=0.1, reg_lambda=2.0, verbose=-1
)
XGBOOST_PARAMS = dict(
    n_estimators=5000, learning_rate=0.02, max_depth=6,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.65,
    reg_alpha=0.0, reg_lambda=2.0, gamma=0.1,
    tree_method="hist", enable_categorical=True, eval_metric="auc"
)

# Ensemble weights (validated in V3)
W_CB, W_LGB, W_XGB = 0.70, 0.25, 0.05

np.random.seed(RANDOM_SEED)
print(f"Input: {INPUT_DIR} | Output: {OUTPUT_DIR} | Threads: {NUM_THREADS}")

# %% [markdown]
# ## Cell 2: Load Data

# %%
log("Loading data...")
p = lambda f: os.path.join(INPUT_DIR, f)
train = pd.read_csv(p("train.csv"))
test  = pd.read_csv(p("test.csv"))
events = pd.read_csv(p("events.csv"))
ss     = pd.read_csv(p("sample_submission.csv"))

for d in (train, test):
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"])
    d["signup_date"]   = pd.to_datetime(d["signup_date"])

ts = pd.to_datetime(events["timestamp"],
                     utc=False if events["timestamp"].astype(str).str.len().max() < 30 else True)
if getattr(ts.dt, "tz", None) is not None:
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
events["timestamp"] = ts
events["event_type"] = events["event_type"].astype("category")

if train["churn"].dtype == object or str(train["churn"].dtype).startswith("str"):
    train["churn"] = train["churn"].astype(str).str.lower().map(
        {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
train["churn"] = train["churn"].astype(int)

log(f"train {train.shape} | test {test.shape} | events {events.shape} | sample_sub {ss.shape}")

# %% [markdown]
# ## Cell 3: Data Inspection

# %%
print("=" * 60)
print("DATA INSPECTION")
print("=" * 60)

print(f"\ntrain columns: {list(train.columns)}")
print(f"test columns:  {list(test.columns)}")
print(f"events columns: {list(events.columns)}")
print(f"sample_sub columns: {list(ss.columns)}")

print(f"\nTarget distribution: {train['churn'].value_counts(normalize=True).round(4).to_dict()}")

print(f"\nMissing values (train):")
print(train.isna().sum()[train.isna().sum() > 0])
print(f"\nMissing values (test):")
print(test.isna().sum()[test.isna().sum() > 0])

print(f"\nEvent types: {events['event_type'].value_counts().to_dict()}")
print(f"Events per customer: mean={events.groupby('customer_id').size().mean():.1f}, "
      f"median={events.groupby('customer_id').size().median():.1f}")

print(f"\ntrain snapshot: {train['snapshot_date'].min().date()} .. {train['snapshot_date'].max().date()}")
print(f"test snapshot:  {test['snapshot_date'].min().date()} .. {test['snapshot_date'].max().date()}")
print(f"events range:   {events['timestamp'].min()} .. {events['timestamp'].max()}")

print(f"\nCustomer overlap train/test: {len(set(train['customer_id']) & set(test['customer_id']))}")
print(f"sample_sub IDs match test: {set(ss.iloc[:, 0]) == set(test['customer_id'])}")

# %% [markdown]
# ## Cell 4: Feature Engineering
#
# Self-contained, snapshot-aware pipeline. No event after a customer's snapshot
# date is ever used as a feature.

# %%
log("Building features...")

# ---- Helpers ----
def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=(b > 0) & np.isfinite(b))
    return out

def _cycle_days(s):
    s = s.astype(str).str.lower()
    return np.where(s.str.contains("year|annual"), 365,
           np.where(s.str.contains("quarter"), 90,
           np.where(s.str.contains("semi|half"), 182, 30)))

def _eff_snapshot(snap):
    return snap["snapshot_date"] + (pd.Timedelta(days=1) if SNAPSHOT_END_OF_DAY else pd.Timedelta(0))

# ---- Customer features ----
CAT_RAW = ["plan", "billing_cycle", "region", "age_band",
           "acquisition_channel", "payment_method"]

def build_customer_features(df):
    out = pd.DataFrame(index=df.index)
    ten = (df["snapshot_date"] - df["signup_date"]).dt.total_seconds() / 86400
    out["tenure_days"] = ten
    out["tenure_months"] = ten / 30.4375
    out["obs_days"] = np.minimum(ten, max(WINDOWS))
    for c in CAT_RAW:
        out[c] = df[c].astype(str).fillna("NA")
    out["monthly_price"] = df["monthly_price"]
    out["num_devices"] = df["num_devices"]
    out["family_sharing"] = df["family_sharing"].astype(str).str.lower().map(
        {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}).astype(float)
    ref = df["referral_code"]
    out["has_referral"] = (ref.notna() & (ref.astype(str).str.strip() != "")).astype(int)
    # Referral code count using ONLY rows with snapshot month <= this row's month
    mon = df["snapshot_date"].dt.to_period("M")
    cnt = pd.Series(np.nan, index=df.index)
    for m in sorted(mon.unique()):
        sub = ref[(mon <= m) & ref.notna()].value_counts()
        idx = mon == m
        cnt[idx] = ref[idx].map(sub).values
    out["referral_code_count_so_far"] = cnt.where(out["has_referral"] == 1)
    if ref.nunique() <= 300:
        out["referral_code"] = ref.astype(str).fillna("NA")
    # Derived
    out["price_per_device"] = out["monthly_price"] / out["num_devices"].replace(0, np.nan)
    out["price_vs_plan_median"] = out["monthly_price"] - out.groupby(df["plan"])["monthly_price"].transform("median")
    cyc = _cycle_days(df["billing_cycle"])
    out["cycle_days"] = cyc
    out["days_to_renewal"] = cyc - (ten % cyc)
    out["is_first_cycle"] = (ten < cyc).astype(int)
    out["tenure_in_cycles"] = ten / cyc
    # Interactions
    out["plan_x_cycle"]    = df["plan"].astype(str) + "|" + df["billing_cycle"].astype(str)
    out["plan_x_payment"]  = df["plan"].astype(str) + "|" + df["payment_method"].astype(str)
    out["channel_x_plan"]  = df["acquisition_channel"].astype(str) + "|" + df["plan"].astype(str)
    out["family_x_devices"] = out["family_sharing"].fillna(0) * out["num_devices"].fillna(0)
    cats = CAT_RAW + (["referral_code"] if "referral_code" in out else []) + \
           ["plan_x_cycle", "plan_x_payment", "channel_x_plan"]
    return out, cats

# ---- Event features (snapshot-aware, vectorized) ----
def build_event_features(snap, events, verbose=True):
    snap = snap[["row_id", "customer_id", "snapshot_date"]].copy()
    snap["eff"] = _eff_snapshot(snap)

    # Merge events with snapshots, keep only past events
    m = events[["customer_id", "timestamp", "event_type", "value"]].merge(
        snap[["row_id", "customer_id", "eff"]], on="customer_id", how="inner")
    age_all = ((m["eff"] - m["timestamp"]).dt.total_seconds() / 86400).values.astype("float32")
    keep = age_all >= 0

    diag = {"events_joined": int(len(m)),
            "events_after_snapshot_dropped": int((~keep).sum()),
            "max_age_days": float(age_all.max()) if len(age_all) else np.nan}
    m = m.loc[keep, ["row_id", "timestamp", "event_type", "value", "eff"]].reset_index(drop=True)
    assert (m["timestamp"] <= m["eff"]).all(), "FUTURE EVENT LEAKED"

    et = m["event_type"].astype(str).values
    val = m["value"].astype(float).values
    m["age"] = age_all[keep]
    m["is_session"] = (et == "session").astype("float32")
    m["is_ticket"]  = (et == "support_ticket").astype("float32")
    m["is_fail"]    = (et == "payment_failed").astype("float32")
    m["is_pc"]      = (et == "plan_change").astype("float32")
    m["up"]   = ((et == "plan_change") & (val > 0)).astype("float32")
    m["down"] = ((et == "plan_change") & (val < 0)).astype("float32")
    m["minutes"] = np.where(et == "session", val, np.nan).astype("float32")
    m["pc_val"]  = np.where(et == "plan_change", val, 0).astype("float32")

    ids = snap["row_id"].values
    F = pd.DataFrame(index=pd.Index(ids, name="row_id"))
    groups = {"counts": [], "recency": [], "windows": [],
              "gaps": [], "decay": [], "relative": [],
              "consistency": [], "interactions": []}

    def add(name, ser, grp, fill=None):
        F[name] = ser.reindex(ids).values if fill is None else ser.reindex(ids).fillna(fill).values
        groups[grp].append(name)

    def ratio(name, a, b, grp="windows"):
        F[name] = _safe_div(F[a], F[b]); groups[grp].append(name)

    def diff(name, a, b, grp="windows"):
        F[name] = F[a] - F[b]; groups[grp].append(name)

    # ======== TOTALS ========
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
    add("mean_session_minutes", gs.mean(), "counts")
    add("median_session_minutes", gs.median(), "counts")
    add("max_session_minutes", gs.max(), "counts")
    add("min_session_minutes", gs.min(), "counts")
    add("std_session_minutes", gs.std(), "counts")
    add("active_session_days_total",
        s.assign(d=np.floor(s["age"])).groupby("row_id")["d"].nunique(), "counts", 0)

    # ======== RECENCY ========
    add("days_since_last_event", g["age"].min(), "recency")
    for nm, col in (("session", "is_session"), ("support_ticket", "is_ticket"),
                    ("payment_failure", "is_fail"), ("plan_change", "is_pc")):
        add(f"days_since_last_{nm}", m.loc[m[col] == 1].groupby("row_id")["age"].min(), "recency")
    add("days_since_first_event", g["age"].max(), "recency")

    # ======== WINDOWED FEATURES ========
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
        add(f"mean_session_minutes_{w}d", gsw.mean(), "windows")
        add(f"max_session_minutes_{w}d", gsw.max(), "windows")
        add(f"active_session_days_{w}d",
            ssw.assign(d=np.floor(ssw["age"])).groupby("row_id")["d"].nunique(), "windows", 0)
        if verbose: log(f"   window {w:3d}d done")

    # ======== V1 RATIOS / TRENDS ========
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

    if verbose: log("   V1 features done, building V2/V3 advanced features...")

    # ======== V2: SESSION GAP FEATURES ========
    sess = m.loc[m["is_session"] == 1, ["row_id", "age"]].copy()
    sess_sorted = sess.sort_values(["row_id", "age"])
    sg = sess_sorted.groupby("row_id")["age"]

    add("days_since_2nd_last_session",
        sess_sorted.groupby("row_id")["age"].apply(
            lambda x: x.nsmallest(2).iloc[-1] if len(x) >= 2 else np.nan), "gaps")
    F["gap_last_two_sessions"] = F.get("days_since_2nd_last_session",
        pd.Series(np.nan, index=F.index)) - F.get("days_since_last_session",
        pd.Series(np.nan, index=F.index))
    groups["gaps"].append("gap_last_two_sessions")

    def _gap_stats(ages):
        if len(ages) < 2:
            return pd.Series({"mean_gap": np.nan, "median_gap": np.nan,
                              "std_gap": np.nan, "min_gap": np.nan, "max_gap": np.nan})
        sa = np.sort(ages.values); gaps = np.diff(sa)
        return pd.Series({"mean_gap": gaps.mean(), "median_gap": np.median(gaps),
                          "std_gap": gaps.std(), "min_gap": gaps.min(), "max_gap": gaps.max()})

    gap_stats = sg.apply(_gap_stats).unstack()
    for col in ["mean_gap", "median_gap", "std_gap", "min_gap", "max_gap"]:
        nm = f"session_{col}"
        add(nm, gap_stats[col] if col in gap_stats.columns else pd.Series(np.nan, index=pd.Index(ids)), "gaps")

    sess_30d = sess[sess["age"] <= 30]
    sg30 = sess_30d.sort_values(["row_id", "age"]).groupby("row_id")["age"]

    def _gap_stats_30d(ages):
        if len(ages) < 2:
            return pd.Series({"mean_gap_30d": np.nan, "max_gap_30d": np.nan, "median_gap_30d": np.nan})
        sa = np.sort(ages.values); gaps = np.diff(sa)
        return pd.Series({"mean_gap_30d": gaps.mean(), "max_gap_30d": gaps.max(), "median_gap_30d": np.median(gaps)})

    gap30 = sg30.apply(_gap_stats_30d).unstack()
    for col in ["mean_gap_30d", "max_gap_30d", "median_gap_30d"]:
        nm = f"session_{col}"
        add(nm, gap30[col] if col in gap30.columns else pd.Series(np.nan, index=pd.Index(ids)), "gaps")

    if verbose: log("   session gap features done")

    # ======== V2: ACTIVITY DECAY BUCKETS ========
    buckets = [(0, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 120)]
    for lo, hi in buckets:
        bm = m.loc[(m["age"] >= lo) & (m["age"] <= hi)]
        bg = bm.groupby("row_id")
        bname = f"{lo}_{hi}"
        add(f"sessions_{bname}", bg["is_session"].sum(), "decay", 0)
        add(f"session_minutes_{bname}",
            bm.loc[bm["is_session"] == 1].groupby("row_id")["minutes"].sum(), "decay", 0)
        add(f"active_days_{bname}",
            bm.loc[bm["is_session"] == 1].assign(
                d=np.floor(bm.loc[bm["is_session"] == 1, "age"])).groupby("row_id")["d"].nunique(), "decay", 0)
        add(f"tickets_{bname}", bg["is_ticket"].sum(), "decay", 0)

    # Compound sums
    F["sessions_0_14"] = F["sessions_0_7"] + F["sessions_8_14"]; groups["decay"].append("sessions_0_14")
    F["sessions_0_30"] = F["sessions_0_14"] + F["sessions_15_30"]; groups["decay"].append("sessions_0_30")
    F["session_minutes_0_14"] = F["session_minutes_0_7"] + F["session_minutes_8_14"]; groups["decay"].append("session_minutes_0_14")
    F["session_minutes_0_30"] = F["session_minutes_0_14"] + F["session_minutes_15_30"]; groups["decay"].append("session_minutes_0_30")

    ratio("decay_sessions_0_7_vs_8_14", "sessions_0_7", "sessions_8_14", "decay")
    ratio("decay_sessions_0_14_vs_15_30", "sessions_0_14", "sessions_15_30", "decay")
    ratio("decay_sessions_0_30_vs_31_60", "sessions_0_30", "sessions_31_60", "decay")
    ratio("decay_minutes_0_30_vs_31_60", "session_minutes_0_30", "session_minutes_31_60", "decay")
    ratio("decay_minutes_0_7_vs_8_14", "session_minutes_0_7", "session_minutes_8_14", "decay")
    diff("diff_sessions_0_7_vs_8_14", "sessions_0_7", "sessions_8_14", "decay")
    diff("diff_sessions_0_14_vs_15_30", "sessions_0_14", "sessions_15_30", "decay")
    diff("diff_sessions_0_30_vs_31_60", "sessions_0_30", "sessions_31_60", "decay")
    diff("diff_minutes_0_30_vs_31_60", "session_minutes_0_30", "session_minutes_31_60", "decay")

    if verbose: log("   activity decay features done")

    # ======== V2: CUSTOMER-RELATIVE BASELINES ========
    ratio("recent_7d_vs_total_sessions", "sessions_7d", "total_sessions", "relative")
    ratio("recent_14d_vs_total_sessions", "sessions_14d", "total_sessions", "relative")
    ratio("recent_30d_vs_total_sessions", "sessions_30d", "total_sessions", "relative")
    ratio("recent_7d_vs_120d_sessions", "sessions_7d", "sessions_120d", "relative")
    ratio("recent_14d_vs_120d_sessions", "sessions_14d", "sessions_120d", "relative")
    ratio("recent_30d_vs_total_minutes", "session_minutes_30d", "total_session_minutes", "relative")
    ratio("recent_7d_vs_120d_minutes", "session_minutes_7d", "session_minutes_120d", "relative")
    ratio("recent_30d_vs_total_active_days", "active_session_days_30d", "active_session_days_total", "relative")
    ratio("recent_14d_vs_total_active_days", "active_session_days_14d", "active_session_days_total", "relative")

    if verbose: log("   customer-relative features done")

    # ======== V2: CONSISTENCY ========
    ratio("sessions_per_active_day_7d", "sessions_7d", "active_session_days_7d", "consistency")
    ratio("sessions_per_active_day_30d", "sessions_30d", "active_session_days_30d", "consistency")
    ratio("sessions_per_active_day_60d", "sessions_60d", "active_session_days_60d", "consistency")
    ratio("minutes_per_session_7d", "session_minutes_7d", "sessions_7d", "consistency")
    ratio("minutes_per_session_30d", "session_minutes_30d", "sessions_30d", "consistency")
    ratio("minutes_per_session_60d", "session_minutes_60d", "sessions_60d", "consistency")
    ratio("minutes_per_active_day_30d", "session_minutes_30d", "active_session_days_30d", "consistency")

    sess_30 = m.loc[(m["is_session"] == 1) & (m["age"] <= 30)].copy()
    sess_30["day"] = np.floor(sess_30["age"]).astype(int)
    daily_counts = sess_30.groupby(["row_id", "day"]).size().reset_index(name="cnt")
    add("std_daily_sessions_30d", daily_counts.groupby("row_id")["cnt"].std(), "consistency")
    dc_std = daily_counts.groupby("row_id")["cnt"].std()
    dc_mean = daily_counts.groupby("row_id")["cnt"].mean()
    F["cv_daily_sessions_30d"] = (dc_std / dc_mean.replace(0, np.nan)).reindex(ids).values
    groups["consistency"].append("cv_daily_sessions_30d")

    if verbose: log("   consistency features done")

    # ======== V2: ACTIVITY TREND (SLOPE) ========
    for window, label in [(14, "14d"), (30, "30d")]:
        sess_w = m.loc[(m["is_session"] == 1) & (m["age"] <= window)].copy()
        sess_w["day"] = np.floor(sess_w["age"]).astype(int)
        daily = sess_w.groupby(["row_id", "day"]).size().reset_index(name="cnt")
        def _slope(g):
            if len(g) < 3: return np.nan
            x = g["day"].values.astype(float); y = g["cnt"].values.astype(float)
            xm = x.mean(); ym = y.mean()
            denom = ((x - xm) ** 2).sum()
            if denom == 0: return 0.0
            return float(((x - xm) * (y - ym)).sum() / denom)
        slopes = daily.groupby("row_id").apply(_slope)
        add(f"session_slope_{label}", slopes, "decay")
    if verbose: log("   activity trend features done")

    # ======== V2: RECENCY INTERACTIONS ========
    F["no_session_7d"]  = (F["sessions_7d"]  == 0).astype(float); groups["interactions"].append("no_session_7d")
    F["no_session_14d"] = (F["sessions_14d"] == 0).astype(float); groups["interactions"].append("no_session_14d")
    F["no_session_30d"] = (F["sessions_30d"] == 0).astype(float); groups["interactions"].append("no_session_30d")
    F["recency_x_sessions_30d"] = F["days_since_last_session"] * F["sessions_30d"]; groups["interactions"].append("recency_x_sessions_30d")
    F["recency_x_minutes_30d"]  = F["days_since_last_session"] * F["session_minutes_30d"]; groups["interactions"].append("recency_x_minutes_30d")
    F["recency_div_active_days_30d"] = _safe_div(
        F["days_since_last_session"].values, np.maximum(F["active_session_days_30d"].values, 1))
    groups["interactions"].append("recency_div_active_days_30d")
    F["recency_div_sessions_30d"] = _safe_div(
        F["days_since_last_session"].values, np.maximum(F["sessions_30d"].values, 1))
    groups["interactions"].append("recency_div_sessions_30d")
    if verbose: log("   recency interactions done")

    # ======== V2: SUPPORT INTERACTIONS ========
    ratio("tickets_per_session_30d", "support_tickets_30d", "sessions_30d", "interactions")
    ratio("tickets_per_active_day_30d", "support_tickets_30d", "active_session_days_30d", "interactions")
    ratio("tickets_per_session_total", "total_support_tickets", "total_sessions", "interactions")
    F["tickets_prev30d"] = F["support_tickets_60d"] - F["support_tickets_30d"]; groups["interactions"].append("tickets_prev30d")
    ratio("tickets_30d_vs_prev30d", "support_tickets_30d", "tickets_prev30d", "interactions")
    F["tickets_with_declining_sessions"] = (
        (F["support_tickets_30d"] > 0) & (F["sessions_30d"] < F["sessions_prev30d"])).astype(float)
    groups["interactions"].append("tickets_with_declining_sessions")
    if verbose: log("   support interactions done")

    # ======== V2: PLAN CHANGE INTERACTIONS ========
    F["downgrade_and_declining"] = (
        (F["downgrades_30d"] > 0) & (F["sessions_30d"] < F["sessions_prev30d"])).astype(float)
    groups["interactions"].append("downgrade_and_declining")
    F["downgrade_and_low_activity"] = (
        (F["downgrades_30d"] > 0) & (F["sessions_30d"] <= 2)).astype(float)
    groups["interactions"].append("downgrade_and_low_activity")
    F["plan_change_and_failure"] = (
        (F["plan_changes_30d"] > 0) & (F["payment_failures_30d"] > 0)).astype(float)
    groups["interactions"].append("plan_change_and_failure")
    F["has_recent_downgrade_30d"] = (F["downgrades_30d"] > 0).astype(float)
    groups["interactions"].append("has_recent_downgrade_30d")
    if verbose: log("   plan change interactions done")

    n_feats = sum(len(v) for v in groups.values())
    if verbose: log(f"   Event features total: {n_feats}")
    return F, groups, diag

# ---- Build full feature matrix ----
full = pd.concat([train.assign(is_train=1),
                  test.assign(is_train=0, churn=np.nan)], ignore_index=True)
full["row_id"] = np.arange(len(full))
full["month"] = full["snapshot_date"].dt.to_period("M")

cust, cat_cols = build_customer_features(full)
F, groups, diag = build_event_features(full, events)
print("   event-join diagnostics:", diag)

ev_f = F.reset_index(drop=True)
X = pd.concat([cust.reset_index(drop=True), ev_f], axis=1)

# Cross-block features
X["sessions_per_obs_day_30d"] = _safe_div(X["sessions_30d"], np.minimum(X["obs_days"], 30))
groups["windows"].append("sessions_per_obs_day_30d")
X["session_minutes_per_obs_day_90d"] = _safe_div(X["session_minutes_90d"], np.minimum(X["obs_days"], 90))
groups["windows"].append("session_minutes_per_obs_day_90d")
X["tenure_x_sessions_30d"] = X["tenure_days"] * X["sessions_30d"]
groups["interactions"].append("tenure_x_sessions_30d")
X["tenure_x_decline"] = X["tenure_days"] * (X["sessions_30d"] < X["sessions_prev30d"]).astype(float)
groups["interactions"].append("tenure_x_decline")
X["recency_vs_tenure"] = _safe_div(X["days_since_last_session"].values, np.maximum(X["tenure_days"].values, 1))
groups["interactions"].append("recency_vs_tenure")

groups["customer"] = list(cust.columns)

# Categorical encoding
for c in cat_cols:
    X[c] = pd.Categorical(X[c].fillna("nan").astype(str),
                           categories=sorted(X[c].fillna("nan").astype(str).unique()))

# Drop constants
const = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
if const:
    print(f"   dropping constant columns: {const}")
    X = X.drop(columns=const)
    for k in groups: groups[k] = [c for c in groups[k] if c not in const]
    cat_cols = [c for c in cat_cols if c not in const]

meta = full[["row_id", "customer_id", "snapshot_date", "month", "is_train", "churn"]].reset_index(drop=True)
full = pd.concat([meta, X], axis=1)
feature_cols = list(X.columns)

log(f"Total features: {len(feature_cols)}")
print("   " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

# %% [markdown]
# ## Cell 5: Feature Validation

# %%
print("=" * 60)
print("FEATURE VALIDATION")
print("=" * 60)

num_feats = full[feature_cols].select_dtypes(include=[np.number])
print(f"Total features: {len(feature_cols)}")
print(f"Numeric features: {len(num_feats.columns)}")
print(f"Categorical features: {len([c for c in feature_cols if str(full[c].dtype) == 'category'])}")
print(f"Inf values: {int(np.isinf(num_feats.values).sum())}")
high_nan = [c for c in num_feats.columns if num_feats[c].isna().mean() > 0.5]
print(f"Columns >50% NaN: {high_nan}")
print(f"'churn' in feature_cols: {'churn' in feature_cols}")
print(f"'customer_id' in feature_cols: {'customer_id' in feature_cols}")
assert "churn" not in feature_cols, "TARGET LEAKED INTO FEATURES"
assert "customer_id" not in feature_cols, "customer_id in features"

# Quick leakage check: verify no event after snapshot was used
tr = full[full["is_train"] == 1]
y = tr["churn"].astype(int).values
for c in feature_cols:
    if str(full[c].dtype) == "category": continue
    x = tr[c].astype(float).fillna(tr[c].astype(float).median())
    if x.nunique() < 2: continue
    a = roc_auc_score(y, x)
    if max(a, 1 - a) > 0.90:
        print(f"   WARNING: suspiciously high univariate AUC for '{c}': {max(a, 1-a):.4f}")

print("Feature validation PASSED.")

# %% [markdown]
# ## Cell 6: Temporal Validation

# %%
print("=" * 60)
print("TEMPORAL VALIDATION")
print("=" * 60)

tr = full[full["is_train"] == 1]
months = sorted(tr["month"].unique())
splits = []
for i in range(1, len(months)):
    va_idx = tr.index[tr["month"] == months[i]].values
    tr_idx = tr.index[tr["month"].isin(months[:i])].values
    assert full.loc[tr_idx, "snapshot_date"].max() < full.loc[va_idx, "snapshot_date"].min()
    splits.append(dict(
        name=f"{'+'.join(str(m)[5:] for m in months[:i])} -> {months[i]}",
        train_idx=tr_idx, valid_idx=va_idx,
        valid_month=str(months[i]),
        primary=(i == len(months) - 1)
    ))

log("Temporal splits:")
for s in splits:
    print(f"   {s['name']:24s} train={len(s['train_idx']):6d} valid={len(s['valid_idx']):6d}"
          f"{'  <- PRIMARY' if s['primary'] else ''}")

# %% [markdown]
# ## Cell 7: CatBoost Validation

# %%
from catboost import CatBoostClassifier

def _cb_frame(X, cats):
    X = X.copy()
    for c in cats: X[c] = X[c].astype(str)
    return X

cats_used = [c for c in cat_cols if c in feature_cols]

log("CatBoost temporal validation")
cb_oof = []
cb_models = {}
for sp in splits:
    t = time.time()
    Xtr = full.loc[sp["train_idx"], feature_cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
    Xva = full.loc[sp["valid_idx"], feature_cols]; yva = full.loc[sp["valid_idx"], "churn"].astype(int).values
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=RANDOM_SEED)
    Xt, Xv = _cb_frame(Xtr, cats_used), _cb_frame(Xva, cats_used)
    m = CatBoostClassifier(random_seed=RANDOM_SEED, thread_count=NUM_THREADS,
                           cat_features=cats_used, **CATBOOST_PARAMS)
    m.fit(Xt.iloc[a], ytr[a], eval_set=(Xt.iloc[b], ytr[b]), use_best_model=True)
    pv = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(yva, pv)
    cb_oof.append(pd.DataFrame({"row_id": full.loc[sp["valid_idx"], "row_id"].values,
                                 "fold": sp["valid_month"], "y": yva, "pred": pv}))
    cb_models[sp["valid_month"]] = m
    print(f"   {sp['name']:24s} AUC={auc:.5f} iters={m.get_best_iteration()} {time.time()-t:.0f}s")

cb_aucs = {f: roc_auc_score(g["y"], g["pred"]) for f, g in pd.concat(cb_oof).groupby("fold")}
print(f"   Rolling mean: {np.mean(list(cb_aucs.values())):.5f}")

# %% [markdown]
# ## Cell 8: LightGBM Validation

# %%
import lightgbm as lgb

log("LightGBM temporal validation")
lgb_oof = []
lgb_models = {}
for sp in splits:
    t = time.time()
    Xtr = full.loc[sp["train_idx"], feature_cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
    Xva = full.loc[sp["valid_idx"], feature_cols]; yva = full.loc[sp["valid_idx"], "churn"].astype(int).values
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=RANDOM_SEED)
    m = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, **LIGHTGBM_PARAMS)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    pv = m.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, pv)
    lgb_oof.append(pd.DataFrame({"row_id": full.loc[sp["valid_idx"], "row_id"].values,
                                  "fold": sp["valid_month"], "y": yva, "pred": pv}))
    lgb_models[sp["valid_month"]] = m
    print(f"   {sp['name']:24s} AUC={auc:.5f} iters={m.best_iteration_} {time.time()-t:.0f}s")

lgb_aucs = {f: roc_auc_score(g["y"], g["pred"]) for f, g in pd.concat(lgb_oof).groupby("fold")}
print(f"   Rolling mean: {np.mean(list(lgb_aucs.values())):.5f}")

# %% [markdown]
# ## Cell 9: XGBoost Validation

# %%
import xgboost as xgb

log("XGBoost temporal validation")
xgb_oof = []
xgb_models = {}
for sp in splits:
    t = time.time()
    Xtr = full.loc[sp["train_idx"], feature_cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
    Xva = full.loc[sp["valid_idx"], feature_cols]; yva = full.loc[sp["valid_idx"], "churn"].astype(int).values
    a, b = train_test_split(np.arange(len(Xtr)), test_size=ES_HOLDOUT, stratify=ytr, random_state=RANDOM_SEED)
    m = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS,
                           early_stopping_rounds=ES_ROUNDS, **XGBOOST_PARAMS)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], verbose=False)
    pv = m.predict_proba(Xva)[:, 1]
    auc = roc_auc_score(yva, pv)
    xgb_oof.append(pd.DataFrame({"row_id": full.loc[sp["valid_idx"], "row_id"].values,
                                  "fold": sp["valid_month"], "y": yva, "pred": pv}))
    xgb_models[sp["valid_month"]] = m
    print(f"   {sp['name']:24s} AUC={auc:.5f} iters={m.best_iteration} {time.time()-t:.0f}s")

xgb_aucs = {f: roc_auc_score(g["y"], g["pred"]) for f, g in pd.concat(xgb_oof).groupby("fold")}
print(f"   Rolling mean: {np.mean(list(xgb_aucs.values())):.5f}")

# %% [markdown]
# ## Cell 10: Validation Ensemble

# %%
print("=" * 60)
print("VALIDATION ENSEMBLE")
print("=" * 60)

oof_all = pd.concat(cb_oof)[["row_id", "fold", "y"]].copy()
oof_all["cb"]  = pd.concat(cb_oof)["pred"].values
oof_all["lgb"] = pd.concat(lgb_oof)["pred"].values
oof_all["xgb"] = pd.concat(xgb_oof)["pred"].values
oof_all["ensemble"] = W_CB * oof_all["cb"] + W_LGB * oof_all["lgb"] + W_XGB * oof_all["xgb"]

prim_month = [s["valid_month"] for s in splits if s["primary"]][0]

print(f"\nEnsemble weights: CatBoost={W_CB} LightGBM={W_LGB} XGBoost={W_XGB}")
print(f"\n{'Model':20s} {'Feb':>8s} {'Mar':>8s} {'Apr':>8s} {'Mean':>8s}")
print("-" * 55)
for model_name, aucs_dict in [("CatBoost", cb_aucs), ("LightGBM", lgb_aucs), ("XGBoost", xgb_aucs)]:
    vals = list(aucs_dict.values())
    print(f"{model_name:20s} {vals[0]:8.5f} {vals[1]:8.5f} {vals[2]:8.5f} {np.mean(vals):8.5f}")

ens_aucs = {f: roc_auc_score(g["y"], g["ensemble"]) for f, g in oof_all.groupby("fold")}
vals = list(ens_aucs.values())
print(f"{'Ensemble':20s} {vals[0]:8.5f} {vals[1]:8.5f} {vals[2]:8.5f} {np.mean(vals):8.5f}")
print(f"\nPrimary (April) AUC: {ens_aucs[prim_month]:.5f}")
print(f"Rolling mean AUC:    {np.mean(vals):.5f}")

# %% [markdown]
# ## Cell 11: Train Final Models on Full Training Data

# %%
log("Training final models on FULL training data...")

train_mask = full["is_train"] == 1
test_mask  = full["is_train"] == 0

Xtr_full = full.loc[train_mask, feature_cols]
ytr_full = full.loc[train_mask, "churn"].astype(int).values
Xte_full = full.loc[test_mask, feature_cols]

# Use best iteration count from primary fold as guide for full training
# CatBoost
cb_iters = cb_models[prim_month].get_best_iteration()
cb_final_params = dict(CATBOOST_PARAMS, iterations=int(cb_iters * 1.1))  # slight increase for more data
a, b = train_test_split(np.arange(len(Xtr_full)), test_size=ES_HOLDOUT, stratify=ytr_full, random_state=RANDOM_SEED)
Xt_f = _cb_frame(Xtr_full, cats_used)
cb_final = CatBoostClassifier(random_seed=RANDOM_SEED, thread_count=NUM_THREADS,
                               cat_features=cats_used, **cb_final_params)
cb_final.fit(Xt_f.iloc[a], ytr_full[a], eval_set=(Xt_f.iloc[b], ytr_full[b]), use_best_model=True)
cb_test_pred = cb_final.predict_proba(_cb_frame(Xte_full, cats_used))[:, 1]
log(f"CatBoost final: {cb_final.get_best_iteration()} iterations")

# LightGBM
lgb_iters = lgb_models[prim_month].best_iteration_
lgb_final_params = dict(LIGHTGBM_PARAMS, n_estimators=int(lgb_iters * 1.1))
a, b = train_test_split(np.arange(len(Xtr_full)), test_size=ES_HOLDOUT, stratify=ytr_full, random_state=RANDOM_SEED)
lgb_final = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, **lgb_final_params)
lgb_final.fit(Xtr_full.iloc[a], ytr_full[a], eval_set=[(Xtr_full.iloc[b], ytr_full[b])],
              eval_metric="auc", callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
lgb_test_pred = lgb_final.predict_proba(Xte_full)[:, 1]
log(f"LightGBM final: {lgb_final.best_iteration_} iterations")

# XGBoost
xgb_iters = xgb_models[prim_month].best_iteration
xgb_final_params = dict(XGBOOST_PARAMS, n_estimators=int(xgb_iters * 1.1))
a, b = train_test_split(np.arange(len(Xtr_full)), test_size=ES_HOLDOUT, stratify=ytr_full, random_state=RANDOM_SEED)
xgb_final = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS,
                                early_stopping_rounds=ES_ROUNDS, **xgb_final_params)
xgb_final.fit(Xtr_full.iloc[a], ytr_full[a], eval_set=[(Xtr_full.iloc[b], ytr_full[b])], verbose=False)
xgb_test_pred = xgb_final.predict_proba(Xte_full)[:, 1]
log(f"XGBoost final: {xgb_final.best_iteration} iterations")

# %% [markdown]
# ## Cell 12: Generate Test Predictions (Ensemble)

# %%
ensemble_pred = W_CB * cb_test_pred + W_LGB * lgb_test_pred + W_XGB * xgb_test_pred

print("Test prediction statistics:")
print(f"  CatBoost: min={cb_test_pred.min():.4f} max={cb_test_pred.max():.4f} mean={cb_test_pred.mean():.4f}")
print(f"  LightGBM: min={lgb_test_pred.min():.4f} max={lgb_test_pred.max():.4f} mean={lgb_test_pred.mean():.4f}")
print(f"  XGBoost:  min={xgb_test_pred.min():.4f} max={xgb_test_pred.max():.4f} mean={xgb_test_pred.mean():.4f}")
print(f"  Ensemble: min={ensemble_pred.min():.4f} max={ensemble_pred.max():.4f} mean={ensemble_pred.mean():.4f}")

# Correlation between models
from scipy.stats import spearmanr
r_cb_lgb, _ = spearmanr(cb_test_pred, lgb_test_pred)
r_cb_xgb, _ = spearmanr(cb_test_pred, xgb_test_pred)
r_lgb_xgb, _ = spearmanr(lgb_test_pred, xgb_test_pred)
print(f"\nSpearman rank correlation:")
print(f"  CB-LGB: {r_cb_lgb:.4f} | CB-XGB: {r_cb_xgb:.4f} | LGB-XGB: {r_lgb_xgb:.4f}")

# %% [markdown]
# ## Cell 13: Create Submission

# %%
# Match sample_submission structure and customer_id ordering
test_customers = full.loc[test_mask, "customer_id"].values
submission = pd.DataFrame({
    ss.columns[0]: test_customers,
    ss.columns[1]: ensemble_pred
})

# Reindex to match sample_submission ordering exactly
submission = submission.set_index(ss.columns[0]).reindex(ss.iloc[:, 0]).reset_index()

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
log(f"Submission saved to {submission_path}")

# %% [markdown]
# ## Cell 14: Final Verification

# %%
print("=" * 50)
print("SUBMISSION VERIFICATION")
print("=" * 50)

sub = pd.read_csv(submission_path)
print(f"Path:             {submission_path}")
print(f"Shape:            {sub.shape}")
print(f"Columns:          {list(sub.columns)}")
print(f"Prediction min:   {sub.iloc[:, 1].min():.6f}")
print(f"Prediction max:   {sub.iloc[:, 1].max():.6f}")
print(f"Prediction mean:  {sub.iloc[:, 1].mean():.6f}")
print(f"Unique customers: {sub.iloc[:, 0].nunique()}")
print(f"NaN predictions:  {sub.iloc[:, 1].isna().sum()}")

# Assertions
assert len(sub) == len(test), f"Expected {len(test)} rows, got {len(sub)}"
assert set(sub.iloc[:, 0]) == set(test["customer_id"]), "Customer IDs don't match test"
assert sub.iloc[:, 1].notna().all(), "NaN predictions found"
assert ((sub.iloc[:, 1] >= 0) & (sub.iloc[:, 1] <= 1)).all(), "Predictions outside [0, 1]"

print(f"\nAll assertions PASSED.")
print(f"\nValidation summary:")
print(f"  CatBoost rolling:  {np.mean(list(cb_aucs.values())):.5f}")
print(f"  LightGBM rolling:  {np.mean(list(lgb_aucs.values())):.5f}")
print(f"  XGBoost rolling:   {np.mean(list(xgb_aucs.values())):.5f}")
print(f"  Ensemble rolling:  {np.mean(list(ens_aucs.values())):.5f}")
print(f"  Ensemble primary:  {ens_aucs[prim_month]:.5f}")

print("\n" + "=" * 50)
print("SUBMISSION READY")
print("=" * 50)
print(f"Path: {submission_path}")
print(f"Rows: {len(sub)}")
print(f"Columns: {list(sub.columns)}")
print(f"Prediction min: {sub.iloc[:, 1].min():.6f}")
print(f"Prediction max: {sub.iloc[:, 1].max():.6f}")
print(f"Prediction mean: {sub.iloc[:, 1].mean():.6f}")
print(f"Unique customers: {sub.iloc[:, 0].nunique()}")
print(f"NaN predictions: 0")
print("=" * 50)

log("DONE")
