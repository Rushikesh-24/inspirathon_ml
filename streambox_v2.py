#!/usr/bin/env python
"""
StreamBox V2 -- Advanced feature engineering + tuning on top of V1 baseline.
Baseline: primary AUC=0.8669, rolling=0.8627

Strategy:
  1. Load V1 features (preserve all leakage protections)
  2. Add advanced behavioral features directly onto the merged events dataframe
  3. Session gaps, activity decay, customer-relative baselines, consistency,
     recency interactions, support interactions, plan change interactions
  4. Hyperparameter tuning (focused)
  5. Improved ensembling
"""
import os, sys, time, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG (same as V1)
# ============================================================================
DATA_DIR      = "data"
OUT_DIR       = "outputs_v2"
CACHE_DIR     = "cache"
RANDOM_SEED   = 42
N_JOBS        = -1
FORCE_REBUILD = True   # Force rebuild to include new features

WINDOWS       = [7, 14, 30, 60, 90, 120]
SNAPSHOT_END_OF_DAY = False
GAP_DAYS      = 0
ES_HOLDOUT    = 0.15
ES_ROUNDS     = 150
ROLLING_FOR_BOOSTERS = True
NUM_THREADS   = (os.cpu_count() or 1) if N_JOBS in (-1, None) else N_JOBS

T0 = time.time()
def log(msg=""): print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

def _have(mod):
    try: __import__(mod); return True
    except Exception: return False
HAVE = {m: _have(m) for m in ("catboost", "lightgbm", "xgboost", "shap")}

def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan); np.divide(a, b, out=out, where=(b > 0) & np.isfinite(b)); return out

# ============================================================================
# 1. LOAD DATA (same as V1)
# ============================================================================
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

# ============================================================================
# 2. CUSTOMER FEATURES (same as V1)
# ============================================================================
CAT_RAW = ["plan", "billing_cycle", "region", "age_band", "acquisition_channel", "payment_method"]

def _cycle_days(s):
    s = s.astype(str).str.lower()
    return np.where(s.str.contains("year|annual"), 365, np.where(s.str.contains("quarter"), 90, np.where(s.str.contains("semi|half"), 182, 30)))

def _eff_snapshot(snap):
    return snap["snapshot_date"] + (pd.Timedelta(days=1) if SNAPSHOT_END_OF_DAY else pd.Timedelta(0))

def build_customer_features(df):
    out = pd.DataFrame(index=df.index)
    ten = (df["snapshot_date"] - df["signup_date"]).dt.total_seconds() / 86400
    out["tenure_days"] = ten; out["tenure_months"] = ten / 30.4375
    out["obs_days"] = np.minimum(ten, max(WINDOWS))
    for c in CAT_RAW: out[c] = df[c].astype(str).fillna("NA")
    out["monthly_price"] = df["monthly_price"]; out["num_devices"] = df["num_devices"]
    out["family_sharing"] = df["family_sharing"].astype(str).str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}).astype(float)
    ref = df["referral_code"]
    out["has_referral"] = (ref.notna() & (ref.astype(str).str.strip() != "")).astype(int)
    mon = df["snapshot_date"].dt.to_period("M"); cnt = pd.Series(np.nan, index=df.index)
    for m in sorted(mon.unique()):
        sub = ref[(mon <= m) & ref.notna()].value_counts()
        idx = mon == m
        cnt[idx] = ref[idx].map(sub).values
    out["referral_code_count_so_far"] = cnt.where(out["has_referral"] == 1)
    if ref.nunique() <= 300:
        out["referral_code"] = ref.astype(str).fillna("NA")
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

# ============================================================================
# 3. EVENT FEATURES -- V2 (extended from V1)
# ============================================================================
def build_event_features_v2(snap, events, verbose=True):
    """
    V2 event features: everything from V1 + session gaps, activity decay buckets,
    customer-relative baselines, consistency metrics, recency interactions,
    support interactions, plan change interactions.
    """
    snap = snap[["row_id", "customer_id", "snapshot_date"]].copy()
    snap["eff"] = _eff_snapshot(snap)
    m = events[["customer_id", "timestamp", "event_type", "value"]].merge(
        snap[["row_id", "customer_id", "eff"]], on="customer_id", how="inner")
    age_all = ((m["eff"] - m["timestamp"]).dt.total_seconds() / 86400).values.astype("float32")
    keep = age_all >= 0
    diag = {"events_joined": int(len(m)), "events_after_snapshot_dropped": int((~keep).sum()),
            "of_which_within_24h_after_snapshot": int(((age_all < 0) & (age_all >= -1)).sum()),
            "max_age_days": float(age_all.max()) if len(age_all) else np.nan}
    m = m.loc[keep, ["row_id", "timestamp", "event_type", "value", "eff"]].reset_index(drop=True)
    assert (m["timestamp"] <= m["eff"]).all(), "FUTURE EVENT LEAKED INTO FEATURES"
    et = m["event_type"].astype(str).values; val = m["value"].astype(float).values
    m["age"] = age_all[keep]
    m["is_session"] = (et == "session").astype("float32")
    m["is_ticket"] = (et == "support_ticket").astype("float32")
    m["is_fail"] = (et == "payment_failed").astype("float32")
    m["is_pc"] = (et == "plan_change").astype("float32")
    m["up"] = ((et == "plan_change") & (val > 0)).astype("float32")
    m["down"] = ((et == "plan_change") & (val < 0)).astype("float32")
    m["minutes"] = np.where(et == "session", val, np.nan).astype("float32")
    m["pc_val"] = np.where(et == "plan_change", val, 0).astype("float32")
    ids = snap["row_id"].values
    F = pd.DataFrame(index=pd.Index(ids, name="row_id"))
    groups = {"counts": [], "recency": [], "windows": [], "gaps": [], "decay": [],
              "relative": [], "consistency": [], "interactions": []}

    def add(name, ser, grp, fill=None):
        F[name] = ser.reindex(ids).values if fill is None else ser.reindex(ids).fillna(fill).values
        groups[grp].append(name)

    # ==== V1 TOTALS (preserved exactly) ====
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
    add("active_session_days_total", s.assign(d=np.floor(s["age"])).groupby("row_id")["d"].nunique(), "counts", 0)

    # ==== V1 RECENCY ====
    add("days_since_last_event", g["age"].min(), "recency")
    for nm, col in (("session", "is_session"), ("support_ticket", "is_ticket"),
                    ("payment_failure", "is_fail"), ("plan_change", "is_pc")):
        add(f"days_since_last_{nm}", m.loc[m[col] == 1].groupby("row_id")["age"].min(), "recency")
    add("days_since_first_event", g["age"].max(), "recency")

    # ==== V1 WINDOWED FEATURES ====
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

    # ==== V1 RATIOS/TRENDS ====
    def ratio(name, a, b, grp="windows"): F[name] = _safe_div(F[a], F[b]); groups[grp].append(name)
    def diff(name, a, b, grp="windows"): F[name] = F[a] - F[b]; groups[grp].append(name)

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

    if verbose: log("   V1 features done, building V2 advanced features...")

    # =========================================================================
    # NEW V2 FEATURES
    # =========================================================================

    # ---- SESSION GAP FEATURES ----
    sess = m.loc[m["is_session"] == 1, ["row_id", "age"]].copy()
    sess_sorted = sess.sort_values(["row_id", "age"])
    sg = sess_sorted.groupby("row_id")["age"]

    # Days since 2nd-last session
    add("days_since_2nd_last_session",
        sess_sorted.groupby("row_id")["age"].apply(lambda x: x.nsmallest(2).iloc[-1] if len(x) >= 2 else np.nan),
        "gaps")

    # Gap between last two sessions
    F["gap_last_two_sessions"] = F.get("days_since_2nd_last_session", pd.Series(np.nan, index=F.index)) - F.get("days_since_last_session", pd.Series(np.nan, index=F.index))
    groups["gaps"].append("gap_last_two_sessions")

    # Session gap statistics (computed from sorted ages per customer)
    def _gap_stats(ages):
        if len(ages) < 2: return pd.Series({"mean_gap": np.nan, "median_gap": np.nan, "std_gap": np.nan,
                                             "min_gap": np.nan, "max_gap": np.nan})
        sorted_ages = np.sort(ages.values)
        gaps = np.diff(sorted_ages)
        return pd.Series({"mean_gap": gaps.mean(), "median_gap": np.median(gaps), "std_gap": gaps.std(),
                          "min_gap": gaps.min(), "max_gap": gaps.max()})

    gap_stats = sg.apply(_gap_stats).unstack()
    for col in ["mean_gap", "median_gap", "std_gap", "min_gap", "max_gap"]:
        nm = f"session_{col}"
        add(nm, gap_stats[col] if col in gap_stats.columns else pd.Series(np.nan, index=pd.Index(ids)), "gaps")

    # Session gaps within 30d
    sess_30d = sess[sess["age"] <= 30]
    sg30 = sess_30d.sort_values(["row_id", "age"]).groupby("row_id")["age"]

    def _gap_stats_30d(ages):
        if len(ages) < 2: return pd.Series({"mean_gap_30d": np.nan, "max_gap_30d": np.nan, "median_gap_30d": np.nan})
        sorted_ages = np.sort(ages.values)
        gaps = np.diff(sorted_ages)
        return pd.Series({"mean_gap_30d": gaps.mean(), "max_gap_30d": gaps.max(), "median_gap_30d": np.median(gaps)})

    gap30 = sg30.apply(_gap_stats_30d).unstack()
    for col in ["mean_gap_30d", "max_gap_30d", "median_gap_30d"]:
        nm = f"session_{col}"
        add(nm, gap30[col] if col in gap30.columns else pd.Series(np.nan, index=pd.Index(ids)), "gaps")

    # recency_vs_tenure is computed later in build_features() where tenure_days is available

    if verbose: log("   session gap features done")

    # ---- ACTIVITY DECAY BUCKETS ----
    buckets = [(0, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 120)]
    for lo, hi in buckets:
        bm = m.loc[(m["age"] >= lo) & (m["age"] <= hi)]
        bg = bm.groupby("row_id")
        bname = f"{lo}_{hi}"
        add(f"sessions_{bname}", bg["is_session"].sum(), "decay", 0)
        add(f"session_minutes_{bname}", bm.loc[bm["is_session"] == 1].groupby("row_id")["minutes"].sum(), "decay", 0)
        add(f"active_days_{bname}",
            bm.loc[bm["is_session"] == 1].assign(d=np.floor(bm.loc[bm["is_session"] == 1, "age"])).groupby("row_id")["d"].nunique(),
            "decay", 0)
        add(f"tickets_{bname}", bg["is_ticket"].sum(), "decay", 0)

    # Compound bucket sums (individual buckets are 0_7, 8_14, 15_30, etc.)
    F["sessions_0_14"] = F["sessions_0_7"] + F["sessions_8_14"]; groups["decay"].append("sessions_0_14")
    F["sessions_0_30"] = F["sessions_0_14"] + F["sessions_15_30"]; groups["decay"].append("sessions_0_30")
    F["session_minutes_0_14"] = F["session_minutes_0_7"] + F["session_minutes_8_14"]; groups["decay"].append("session_minutes_0_14")
    F["session_minutes_0_30"] = F["session_minutes_0_14"] + F["session_minutes_15_30"]; groups["decay"].append("session_minutes_0_30")

    # Decay ratios
    ratio("decay_sessions_0_7_vs_8_14", "sessions_0_7", "sessions_8_14", "decay")
    ratio("decay_sessions_0_14_vs_15_30", "sessions_0_14", "sessions_15_30", "decay")
    ratio("decay_sessions_0_30_vs_31_60", "sessions_0_30", "sessions_31_60", "decay")
    ratio("decay_minutes_0_30_vs_31_60", "session_minutes_0_30", "session_minutes_31_60", "decay")
    ratio("decay_minutes_0_7_vs_8_14", "session_minutes_0_7", "session_minutes_8_14", "decay")

    # Decay differences
    diff("diff_sessions_0_7_vs_8_14", "sessions_0_7", "sessions_8_14", "decay")
    diff("diff_sessions_0_14_vs_15_30", "sessions_0_14", "sessions_15_30", "decay")
    diff("diff_sessions_0_30_vs_31_60", "sessions_0_30", "sessions_31_60", "decay")
    diff("diff_minutes_0_30_vs_31_60", "session_minutes_0_30", "session_minutes_31_60", "decay")

    if verbose: log("   activity decay features done")

    # ---- CUSTOMER-RELATIVE BASELINES ----
    # Compare recent activity to full-history baseline
    ratio("recent_7d_vs_total_sessions", "sessions_7d", "total_sessions", "relative")
    ratio("recent_14d_vs_total_sessions", "sessions_14d", "total_sessions", "relative")
    ratio("recent_30d_vs_total_sessions", "sessions_30d", "total_sessions", "relative")
    ratio("recent_7d_vs_120d_sessions", "sessions_7d", "sessions_120d", "relative")
    ratio("recent_14d_vs_120d_sessions", "sessions_14d", "sessions_120d", "relative")

    # Minutes relative
    ratio("recent_30d_vs_total_minutes", "session_minutes_30d", "total_session_minutes", "relative")
    ratio("recent_7d_vs_120d_minutes", "session_minutes_7d", "session_minutes_120d", "relative")

    # Active days relative
    ratio("recent_30d_vs_total_active_days", "active_session_days_30d", "active_session_days_total", "relative")
    ratio("recent_14d_vs_total_active_days", "active_session_days_14d", "active_session_days_total", "relative")

    if verbose: log("   customer-relative features done")

    # ---- CONSISTENCY FEATURES ----
    # Sessions per active day
    ratio("sessions_per_active_day_7d", "sessions_7d", "active_session_days_7d", "consistency")
    ratio("sessions_per_active_day_30d", "sessions_30d", "active_session_days_30d", "consistency")
    ratio("sessions_per_active_day_60d", "sessions_60d", "active_session_days_60d", "consistency")

    # Minutes per session
    ratio("minutes_per_session_7d", "session_minutes_7d", "sessions_7d", "consistency")
    ratio("minutes_per_session_30d", "session_minutes_30d", "sessions_30d", "consistency")
    ratio("minutes_per_session_60d", "session_minutes_60d", "sessions_60d", "consistency")

    # Minutes per active day
    ratio("minutes_per_active_day_30d", "session_minutes_30d", "active_session_days_30d", "consistency")

    # Daily session variability (std of per-day session counts within 30d)
    sess_30 = m.loc[(m["is_session"] == 1) & (m["age"] <= 30)].copy()
    sess_30["day"] = np.floor(sess_30["age"]).astype(int)
    daily_counts = sess_30.groupby(["row_id", "day"]).size().reset_index(name="cnt")
    add("std_daily_sessions_30d", daily_counts.groupby("row_id")["cnt"].std(), "consistency")
    dc_std = daily_counts.groupby("row_id")["cnt"].std()
    dc_mean = daily_counts.groupby("row_id")["cnt"].mean()
    F["cv_daily_sessions_30d"] = (dc_std / dc_mean.replace(0, np.nan)).reindex(ids).values
    groups["consistency"].append("cv_daily_sessions_30d")

    if verbose: log("   consistency features done")

    # ---- ACTIVITY TREND (LINEAR SLOPE) ----
    # Compute session count slope over last 30d using simple linear regression
    # Aggregate by day, then regress count ~ day
    for window, label in [(14, "14d"), (30, "30d")]:
        sess_w = m.loc[(m["is_session"] == 1) & (m["age"] <= window)].copy()
        sess_w["day"] = np.floor(sess_w["age"]).astype(int)
        daily = sess_w.groupby(["row_id", "day"]).size().reset_index(name="cnt")

        def _slope(g):
            if len(g) < 3: return np.nan
            x = g["day"].values.astype(float)
            y = g["cnt"].values.astype(float)
            xm = x.mean(); ym = y.mean()
            denom = ((x - xm) ** 2).sum()
            if denom == 0: return 0.0
            return float(((x - xm) * (y - ym)).sum() / denom)

        slopes = daily.groupby("row_id").apply(_slope)
        add(f"session_slope_{label}", slopes, "decay")

    if verbose: log("   activity trend features done")

    # ---- RECENCY INTERACTIONS ----
    # Binary indicators
    F["no_session_7d"] = (F["sessions_7d"] == 0).astype(float); groups["interactions"].append("no_session_7d")
    F["no_session_14d"] = (F["sessions_14d"] == 0).astype(float); groups["interactions"].append("no_session_14d")
    F["no_session_30d"] = (F["sessions_30d"] == 0).astype(float); groups["interactions"].append("no_session_30d")

    # Recency * activity interactions
    F["recency_x_sessions_30d"] = F["days_since_last_session"] * F["sessions_30d"]; groups["interactions"].append("recency_x_sessions_30d")
    F["recency_x_minutes_30d"] = F["days_since_last_session"] * F["session_minutes_30d"]; groups["interactions"].append("recency_x_minutes_30d")
    F["recency_div_active_days_30d"] = _safe_div(F["days_since_last_session"].values, np.maximum(F["active_session_days_30d"].values, 1))
    groups["interactions"].append("recency_div_active_days_30d")
    F["recency_div_sessions_30d"] = _safe_div(F["days_since_last_session"].values, np.maximum(F["sessions_30d"].values, 1))
    groups["interactions"].append("recency_div_sessions_30d")

    if verbose: log("   recency interaction features done")

    # ---- SUPPORT INTERACTIONS ----
    # Support per session / active day
    ratio("tickets_per_session_30d", "support_tickets_30d", "sessions_30d", "interactions")
    ratio("tickets_per_active_day_30d", "support_tickets_30d", "active_session_days_30d", "interactions")
    ratio("tickets_per_session_total", "total_support_tickets", "total_sessions", "interactions")

    # Recent vs historical support
    F["tickets_prev30d"] = F["support_tickets_60d"] - F["support_tickets_30d"]; groups["interactions"].append("tickets_prev30d")
    ratio("tickets_30d_vs_prev30d", "support_tickets_30d", "tickets_prev30d", "interactions")

    # Support + declining sessions indicator
    F["tickets_with_declining_sessions"] = (
        (F["support_tickets_30d"] > 0) & (F["sessions_30d"] < F["sessions_prev30d"])
    ).astype(float)
    groups["interactions"].append("tickets_with_declining_sessions")

    if verbose: log("   support interaction features done")

    # ---- PLAN CHANGE INTERACTIONS ----
    # Downgrade + activity decline
    F["downgrade_and_declining"] = (
        (F["downgrades_30d"] > 0) & (F["sessions_30d"] < F["sessions_prev30d"])
    ).astype(float)
    groups["interactions"].append("downgrade_and_declining")

    # Downgrade + low activity
    F["downgrade_and_low_activity"] = (
        (F["downgrades_30d"] > 0) & (F["sessions_30d"] <= 2)
    ).astype(float)
    groups["interactions"].append("downgrade_and_low_activity")

    # Plan change + payment failure
    F["plan_change_and_failure"] = (
        (F["plan_changes_30d"] > 0) & (F["payment_failures_30d"] > 0)
    ).astype(float)
    groups["interactions"].append("plan_change_and_failure")

    # Days since plan change (already exists in recency, but explicitly add to interactions)
    F["has_recent_downgrade_30d"] = (F["downgrades_30d"] > 0).astype(float)
    groups["interactions"].append("has_recent_downgrade_30d")

    if verbose: log("   plan change interaction features done")

    n_v2 = sum(len(v) for v in groups.values()) - (len(groups.get("counts", [])) + len(groups.get("recency", [])) +
               sum(1 for c in groups.get("windows", []) if not any(c.startswith(p) for p in ["decay_", "diff_"])))
    if verbose: log(f"   V2 total features in event block: {sum(len(v) for v in groups.values())} (V2 new: ~{n_v2})")

    return F, groups, diag

# ============================================================================
# 4. BUILD FEATURES (same structure as V1 but using V2 event features)
# ============================================================================
def build_features(train, test, events):
    log("Building features ...")
    full = pd.concat([train.assign(is_train=1), test.assign(is_train=0, churn=np.nan)], ignore_index=True)
    full["row_id"] = np.arange(len(full)); full["month"] = full["snapshot_date"].dt.to_period("M")
    cust, cat_cols = build_customer_features(full)
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(["train.csv", "test.csv", "events.csv"], "v2_advanced")
    cpath = os.path.join(CACHE_DIR, f"event_feats_{key}.pkl")
    if os.path.exists(cpath) and not FORCE_REBUILD:
        log(f"   loading cached event features: {cpath}"); F, groups, diag = pd.read_pickle(cpath)
    else:
        t = time.time(); F, groups, diag = build_event_features_v2(full, events)
        pd.to_pickle((F, groups, diag), cpath); log(f"   event features built in {time.time() - t:.1f}s -> cached")
    print("   event-join diagnostics:", diag)
    ev_f = F.reset_index(drop=True)
    X = pd.concat([cust.reset_index(drop=True), ev_f], axis=1)
    # Derived rates needing both blocks
    X["sessions_per_obs_day_30d"] = _safe_div(X["sessions_30d"], np.minimum(X["obs_days"], 30)); groups["windows"].append("sessions_per_obs_day_30d")
    X["session_minutes_per_obs_day_90d"] = _safe_div(X["session_minutes_90d"], np.minimum(X["obs_days"], 90)); groups["windows"].append("session_minutes_per_obs_day_90d")

    # V2: tenure interactions (need customer+event features)
    X["tenure_x_sessions_30d"] = X["tenure_days"] * X["sessions_30d"]; groups["interactions"].append("tenure_x_sessions_30d")
    X["tenure_x_decline"] = X["tenure_days"] * (X["sessions_30d"] < X["sessions_prev30d"]).astype(float); groups["interactions"].append("tenure_x_decline")
    X["recency_vs_tenure"] = _safe_div(X["days_since_last_session"].values, np.maximum(X["tenure_days"].values, 1)); groups["interactions"].append("recency_vs_tenure_cust")

    groups["customer"] = list(cust.columns)
    for c in cat_cols:
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

def _cache_key(files, extra):
    h = hashlib.md5()
    for f in files:
        st = os.stat(os.path.join(DATA_DIR, f)); h.update(f"{f}{st.st_size}{int(st.st_mtime)}".encode())
    h.update(str((WINDOWS, SNAPSHOT_END_OF_DAY, extra)).encode()); return h.hexdigest()[:12]

# ============================================================================
# 5. LEAKAGE CHECKS (same as V1 -- critical to preserve)
# ============================================================================
def run_leakage_checks(full, feature_cols, events, groups, n_rows=300):
    print("\n" + "=" * 70 + "\nLEAKAGE CHECKS\n" + "=" * 70)
    rng = np.random.RandomState(RANDOM_SEED)
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

    # (b) poison test
    sub = sample.reset_index(drop=True).copy(); sub["row_id"] = np.arange(len(sub))
    ev_sub = events[events["customer_id"].isin(set(sub["customer_id"]))]
    base, _, _ = build_event_features_v2(sub, ev_sub, verbose=False)
    eff = _eff_snapshot(sub) + pd.Timedelta(hours=1 if not SNAPSHOT_END_OF_DAY else 1)
    px = pd.concat([pd.DataFrame({"customer_id": sub["customer_id"], "timestamp": eff, "event_type": t, "value": v})
                    for t, v in (("session", 99999.0), ("payment_failed", np.nan), ("support_ticket", np.nan), ("plan_change", -1.0))])
    px["event_type"] = px["event_type"].astype(ev_sub["event_type"].dtype) if str(ev_sub["event_type"].dtype) == "category" else px["event_type"]
    ev_p = pd.concat([ev_sub.assign(event_type=ev_sub["event_type"].astype(str)), px.assign(event_type=px["event_type"].astype(str))], ignore_index=True)
    poisoned, _, _ = build_event_features_v2(sub, ev_p, verbose=False)
    same = np.allclose(base.values.astype(float), poisoned.values.astype(float), equal_nan=True)
    print(f"(b) poison test (extreme events injected after snapshot): features unchanged -> {same}")
    assert same, "Features changed when post-snapshot events were added -> FUTURE LEAK"
    print("Leakage checks PASSED.")
    return

# ============================================================================
# 6. TEMPORAL SPLITS (same as V1)
# ============================================================================
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

# ============================================================================
# 7. MODELS (V2 hyperparameters -- tuned)
# ============================================================================
def _inner_split(n_idx, y):
    return train_test_split(np.arange(n_idx), test_size=ES_HOLDOUT, stratify=y, random_state=RANDOM_SEED)

def _cb_frame(X, cats):
    X = X.copy()
    for c in cats: X[c] = X[c].astype(str)
    return X

def _train_auc(predict_proba, Xtr, ytr, a, n=20000):
    sub = a[:n]; return roc_auc_score(ytr[sub], predict_proba(Xtr.iloc[sub])[:, 1])

# V2 tuned hyperparameters
CATBOOST_PARAMS_V2 = dict(iterations=4000, learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
                          random_strength=0.5, bagging_temperature=0.3,
                          eval_metric="AUC", od_type="Iter", od_wait=200, verbose=0)
LIGHTGBM_PARAMS_V2 = dict(n_estimators=5000, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                           max_depth=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           reg_alpha=0.1, reg_lambda=2.0, verbose=-1)
XGBOOST_PARAMS_V2  = dict(n_estimators=5000, learning_rate=0.02, max_depth=7, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0,
                           tree_method="hist", enable_categorical=True, eval_metric="auc")
LADDER_LGBM_PARAMS = dict(LIGHTGBM_PARAMS_V2, learning_rate=0.06)

def train_lightgbm(Xtr, ytr, Xva, yva_unused, cats, params=None):
    import lightgbm as lgb
    p = dict(LIGHTGBM_PARAMS_V2, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    m = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, **p)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    return m, m.predict_proba(Xva)[:, 1], _train_auc(m.predict_proba, Xtr, ytr, a), m.best_iteration_

def train_xgboost(Xtr, ytr, Xva, yva_unused, cats, params=None):
    import xgboost as xgb
    p = dict(XGBOOST_PARAMS_V2, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    m = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=NUM_THREADS, early_stopping_rounds=ES_ROUNDS, **p)
    m.fit(Xtr.iloc[a], ytr[a], eval_set=[(Xtr.iloc[b], ytr[b])], verbose=False)
    return m, m.predict_proba(Xva)[:, 1], _train_auc(m.predict_proba, Xtr, ytr, a), m.best_iteration

def train_catboost(Xtr, ytr, Xva, yva_unused, cats, params=None):
    from catboost import CatBoostClassifier
    p = dict(CATBOOST_PARAMS_V2, **(params or {})); a, b = _inner_split(len(Xtr), ytr)
    Xt, Xv = _cb_frame(Xtr, cats), _cb_frame(Xva, cats)
    m = CatBoostClassifier(random_seed=RANDOM_SEED, thread_count=NUM_THREADS, cat_features=cats, **p)
    m.fit(Xt.iloc[a], ytr[a], eval_set=(Xt.iloc[b], ytr[b]), use_best_model=True)
    return m, m.predict_proba(Xv)[:, 1], _train_auc(lambda X: m.predict_proba(_cb_frame(X, cats)), Xtr, ytr, a), m.get_best_iteration()

TRAINERS = {"catboost": train_catboost, "lightgbm": train_lightgbm, "xgboost": train_xgboost}

def evaluate_model(name, full, cols, cats, splits, only_primary=False, params=None, verbose=True):
    fn = TRAINERS[name]; cats = [c for c in cats if c in cols]
    res = dict(aucs={}, oof=[], models={}, time=0.0, train_auc=[], iters=[])
    for sp in splits:
        if only_primary and not sp["primary"]: continue
        t = time.time()
        Xtr = full.loc[sp["train_idx"], cols]; ytr = full.loc[sp["train_idx"], "churn"].astype(int).values
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
    for m in ("lightgbm", "catboost", "xgboost"):
        if HAVE.get(m): return m
    sys.exit("No boosting library installed. pip install lightgbm catboost xgboost")

# ============================================================================
# 8. EXPERIMENTS
# ============================================================================
def run_experiments(full, feature_cols, cat_cols, groups, splits):
    rows = []; oof_frames = {}; models = {}
    lad = _ladder_model(); lp = LADDER_LGBM_PARAMS if lad == "lightgbm" else None

    # Flatten all group keys for the full feature set
    all_feats = feature_cols

    # Define feature subsets for comparison
    cust = groups["customer"]
    v1_counts = groups["counts"]
    v1_recency = groups["recency"]
    v1_windows = groups["windows"]
    v2_gaps = groups.get("gaps", [])
    v2_decay = groups.get("decay", [])
    v2_relative = groups.get("relative", [])
    v2_consistency = groups.get("consistency", [])
    v2_interactions = groups.get("interactions", [])

    v1_feats = cust + v1_counts + v1_recency + v1_windows
    v2_new = v2_gaps + v2_decay + v2_relative + v2_consistency + v2_interactions
    v2_full = v1_feats + v2_new

    # Deduplicate while preserving order
    seen = set()
    v2_full_dedup = []
    for c in v2_full:
        if c not in seen and c in full.columns:
            seen.add(c); v2_full_dedup.append(c)
    v2_full = v2_full_dedup

    # ---- Feature subset experiments ----
    experiments = [
        ("A. V1 features (baseline)", v1_feats),
        ("B. V1 + session gaps", [c for c in v1_feats + v2_gaps if c in full.columns]),
        ("C. V1 + decay/trends", [c for c in v1_feats + v2_decay if c in full.columns]),
        ("D. V1 + gaps+decay+relative", [c for c in v1_feats + v2_gaps + v2_decay + v2_relative if c in full.columns]),
        ("E. Full V2 features", v2_full),
    ]

    print("\n" + "=" * 70 + f"\nFEATURE SUBSET EXPERIMENTS ({lad})\n" + "=" * 70)
    for nm, cols in experiments:
        cols_dedup = list(dict.fromkeys(c for c in cols if c in full.columns))
        log(nm + f"  [{len(cols_dedup)} features]")
        r = evaluate_model(lad, full, cols_dedup, cat_cols, splits, params=lp, verbose=True)
        rows.append(dict(experiment=nm, model=lad, features=len(cols_dedup),
                         primary_auc=r["primary"], rolling_mean=r["mean"], rolling_std=r["std"],
                         train_auc=float(np.mean(r["train_auc"])), time_s=round(r["time"], 1),
                         **{f"auc_{k}": v for k, v in r["aucs"].items()}))

    # ---- Full V2 features with each model ----
    print("\n" + "=" * 70 + "\nFULL V2 FEATURE SET: CatBoost / LightGBM / XGBoost\n" + "=" * 70)
    names = {"catboost": "F. CatBoost V2", "lightgbm": "G. LightGBM V2", "xgboost": "H. XGBoost V2"}
    preds = {}
    for k in ("catboost", "lightgbm", "xgboost"):
        if not HAVE[k]: log(f"{names[k]} skipped (not installed)"); continue
        log(names[k] + f"  [{len(v2_full)} features]")
        r = evaluate_model(k, full, v2_full, cat_cols, splits, only_primary=not ROLLING_FOR_BOOSTERS)
        rows.append(dict(experiment=names[k], model=k, features=len(v2_full),
                         primary_auc=r["primary"], rolling_mean=r["mean"], rolling_std=r["std"],
                         train_auc=float(np.mean(r["train_auc"])), time_s=round(r["time"], 1),
                         **{f"auc_{kk}": v for kk, v in r["aucs"].items()}))
        preds[k] = pd.concat(r["oof"], ignore_index=True); models[k] = r["models"]

    # ---- Ensemble ----
    ens_rows = None
    if len(preds) >= 2:
        base = list(preds.values())[0][["row_id", "fold", "y"]].copy()
        for k, d in preds.items(): base[k] = d["pred"].values
        for k in preds: base[k + "_rank"] = base.groupby("fold")[k].rank(pct=True)
        base["ensemble_rank"] = base[[k + "_rank" for k in preds]].mean(axis=1)
        base["ensemble_prob"] = base[list(preds)].mean(axis=1)

        # Also try weighted: catboost weighted higher since it was best single in V1
        if "catboost" in preds and "lightgbm" in preds and "xgboost" in preds:
            base["ensemble_weighted"] = (0.45 * base["catboost_rank"] + 0.30 * base["lightgbm_rank"] + 0.25 * base["xgboost_rank"])

        for ens_name, ens_col in [("I. Ensemble (rank avg)", "ensemble_rank"),
                                    ("J. Ensemble (prob avg)", "ensemble_prob"),
                                    ("K. Ensemble (weighted)", "ensemble_weighted")]:
            if ens_col not in base.columns: continue
            aucs = {f: roc_auc_score(g["y"], g[ens_col]) for f, g in base.groupby("fold")}
            prim = [s["valid_month"] for s in splits if s["primary"]][0]
            rows.append(dict(experiment=ens_name, model="+".join(preds), features=len(v2_full),
                             primary_auc=aucs[prim], rolling_mean=np.mean(list(aucs.values())),
                             rolling_std=np.std(list(aucs.values())), time_s=0.0,
                             **{f"auc_{k}": v for k, v in aucs.items()}))
        pc = base[list(preds)].corr(method="spearman")
        print("\nValidation-prediction rank correlation between models:\n", pc.round(4))
        ens_rows = base

    res = pd.DataFrame(rows)
    return res, ens_rows, models, v2_full

# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
def model_importance(name, model, cols):
    if name == "lightgbm": v = model.booster_.feature_importance("gain")
    elif name == "xgboost": s = model.get_booster().get_score(importance_type="gain"); v = np.array([s.get(c, 0.0) for c in cols])
    else: v = model.get_feature_importance()
    v = np.asarray(v, dtype=float)
    return pd.DataFrame({"feature": cols, "importance": v, "importance_pct": 100 * v / v.sum()}).sort_values("importance", ascending=False)

# ============================================================================
# 10. DISTRIBUTION SHIFT ANALYSIS
# ============================================================================
def analyze_shift(full, feature_cols):
    print("\n" + "=" * 70 + "\nDISTRIBUTION SHIFT ANALYSIS\n" + "=" * 70)
    tr = full[full["is_train"] == 1]
    te = full[full["is_train"] == 0]
    apr = tr[tr["month"].astype(str) == "2026-04"]
    may = te[te["month"].astype(str) == "2026-05"]
    jun = te[te["month"].astype(str) == "2026-06"]

    shifted_features = [
        "referral_code_count_so_far", "tenure_days", "tenure_months",
        "min_session_minutes", "days_since_first_event",
        "days_since_last_support_ticket", "mean_session_minutes_30d",
        "price_per_device", "sessions_30d", "active_session_days_30d"
    ]
    shifted_features = [f for f in shifted_features if f in full.columns and str(full[f].dtype) != "category"]

    print(f"\n{'Feature':<35s} {'Apr median':>12s} {'May median':>12s} {'Jun median':>12s} {'Apr->May shift':>15s}")
    print("-" * 90)
    for f in shifted_features:
        a = apr[f].median(); m_ = may[f].median(); j = jun[f].median()
        shift = (m_ - a) / max(abs(a), 1e-6) * 100
        print(f"{f:<35s} {a:12.2f} {m_:12.2f} {j:12.2f} {shift:14.1f}%")

# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True); np.random.seed(RANDOM_SEED)
    print(f"Libraries available: {HAVE} | threads={NUM_THREADS} | seed={RANDOM_SEED}")
    print("=" * 70)
    print("STREAMBOX V2 -- Advanced Feature Engineering")
    print("Baseline (V1): primary AUC=0.8669, rolling mean=0.8627")
    print("=" * 70)

    train, test, events, ss = load_data()
    full, feature_cols, cat_cols, groups = build_features(train, test, events)

    # Leakage checks
    run_leakage_checks(full, feature_cols, events, groups)

    # Temporal splits
    splits = make_temporal_splits(full)

    # Distribution shift analysis
    analyze_shift(full, feature_cols)

    # Run experiments
    results, ens_frame, models, v2_full = run_experiments(full, feature_cols, cat_cols, groups, splits)

    # Save results
    results.to_csv(os.path.join(OUT_DIR, "experiment_results_v2.csv"), index=False)

    # Feature importance for best single model
    best_rows = results[results.experiment.str.startswith(("F.", "G.", "H."))].sort_values("primary_auc")
    if len(best_rows) > 0:
        best_single = best_rows.iloc[-1]["model"]
        best_exp = best_rows.iloc[-1]["experiment"]
        prim = [s["valid_month"] for s in splits if s["primary"]][0]
        imp = model_importance(best_single, models[best_single][prim], v2_full)
        imp.to_csv(os.path.join(OUT_DIR, "feature_importance_v2.csv"), index=False)
        print("\n" + "=" * 70 + f"\nTOP 50 FEATURES ({best_exp}, primary split)\n" + "=" * 70)
        print(imp.head(50).round(3).to_string(index=False))

        # Check which V2 features appear in top 50
        v2_new_names = set(groups.get("gaps", []) + groups.get("decay", []) + groups.get("relative", []) +
                           groups.get("consistency", []) + groups.get("interactions", []))
        v2_in_top50 = imp.head(50)[imp.head(50)["feature"].isin(v2_new_names)]
        print(f"\n--- V2 NEW features in top 50: {len(v2_in_top50)}")
        if len(v2_in_top50) > 0:
            print(v2_in_top50[["feature", "importance_pct"]].round(3).to_string(index=False))

    # ---- FINAL SUMMARY ----
    print("\n" + "=" * 60)
    print("STREAMBOX V2 RESULTS")
    print("=" * 60)

    print(f"\n{'Experiment':35s} {'Primary AUC':>14s} {'Rolling Mean':>14s} {'Rolling Std':>14s}")
    print("-" * 80)
    for _, r in results.iterrows():
        tra = f"{r.get('train_auc', np.nan):.4f}" if pd.notna(r.get("train_auc", np.nan)) else "-"
        print(f"{r.experiment:35s} {r.primary_auc:14.4f} {r.rolling_mean:14.4f} {r.rolling_std:14.4f}")

    best_row = results.loc[results["primary_auc"].idxmax()]
    v1_primary = 0.8669
    v1_rolling = 0.8627

    print(f"\nV1 baseline:")
    print(f"    Best primary AUC = {v1_primary}")
    print(f"    Best rolling mean = {v1_rolling}")
    print(f"\nV2 best:")
    print(f"    Experiment = {best_row.experiment}")
    print(f"    Primary AUC = {best_row.primary_auc:.4f}")
    print(f"    Rolling mean = {best_row.rolling_mean:.4f}")
    print(f"    Rolling std = {best_row.rolling_std:.4f}")
    print(f"\nImprovement:")
    print(f"    Primary = {best_row.primary_auc - v1_primary:+.4f}")
    print(f"    Rolling = {best_row.rolling_mean - v1_rolling:+.4f}")

    # Per-fold AUC
    fold_cols = [c for c in results.columns if c.startswith("auc_")]
    print("\nPer-fold AUC:")
    print(results[["experiment"] + fold_cols].round(4).to_string(index=False))

    print("=" * 60)
    print(f"Saved to {OUT_DIR}/: experiment_results_v2.csv, feature_importance_v2.csv")
    log("done")

if __name__ == "__main__":
    main()
