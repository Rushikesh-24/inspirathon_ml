.venv) kaggle % python claude.py                     
Libraries available: {'catboost': True, 'lightgbm': True, 'xgboost': True, 'shap': True} | threads=8 | seed=42
[    1.7s] Loading data ...
[    3.7s] train (60000, 14) | test (40000, 13) | events (2613721, 4) | sample_submission (40000, 2)

======================================================================
DATA INSPECTION
======================================================================

--- train: shape=(60000, 14), memory=9.6 MB
                              dtype  missing  missing%  nunique
customer_id                     str        0      0.00    60000
snapshot_date        datetime64[us]        0      0.00        4
signup_date          datetime64[us]        0      0.00     2050
plan                            str        0      0.00        3
billing_cycle                   str        0      0.00        2
monthly_price               float64        0      0.00     2171
region                          str        0      0.00        8
age_band                        str     3636      6.06        6
acquisition_channel             str        0      0.00        7
referral_code                   str    45554     75.92      300
payment_method                  str     4757      7.93        3
num_devices                   int64        0      0.00       10
family_sharing                int64        0      0.00        2
churn                         int64        0      0.00        2

--- test: shape=(40000, 13), memory=6.1 MB
                              dtype  missing  missing%  nunique
customer_id                     str        0      0.00    40000
snapshot_date        datetime64[us]        0      0.00        2
signup_date          datetime64[us]        0      0.00     1901
plan                            str        0      0.00        3
billing_cycle                   str        0      0.00        2
monthly_price               float64        0      0.00     2145
region                          str        0      0.00        8
age_band                        str     2401      6.00        6
acquisition_channel             str        0      0.00        7
referral_code                   str    30884     77.21      300
payment_method                  str     3131      7.83        3
num_devices                   int64        0      0.00       10
family_sharing                int64        0      0.00        2

--- events: shape=(2613721, 4), memory=88.9 MB
                      dtype  missing  missing%  nunique
customer_id             str        0      0.00    97991
timestamp    datetime64[us]        0      0.00  2450623
event_type         category        0      0.00        4
value               float64   113208      4.33     2966

--- target (train): {0: 0.8806, 1: 0.1194}

churn rate by snapshot month (train):
                 rows  churn_rate
snapshot_date                   
2026-01        14993      0.1173
2026-02        15180      0.1199
2026-03        15014      0.1228
2026-04        14813      0.1177

test rows by snapshot month:
 snapshot_date
2026-05    20058
2026-06    19942
Freq: M, Name: count, dtype: int64
train: snapshot 2026-01-31 .. 2026-04-30 | signup 2020-08-10 .. 2026-04-29
test: snapshot 2026-05-31 .. 2026-06-30 | signup 2020-12-08 .. 2026-06-29
events: 2025-10-03 00:00:17 .. 2026-06-29 23:59:36

--- categorical cardinalities (train / test / unseen-in-train share of test rows):
  plan                       3       3   unseen  0.00%   missing tr/te   0.0%/  0.0%
  billing_cycle              2       2   unseen  0.00%   missing tr/te   0.0%/  0.0%
  region                     8       8   unseen  0.00%   missing tr/te   0.0%/  0.0%
  age_band                   6       6   unseen  0.00%   missing tr/te   6.1%/  6.0%
  acquisition_channel        7       7   unseen  0.00%   missing tr/te   0.0%/  0.0%
  referral_code            300     300   unseen  0.00%   missing tr/te  75.9%/ 77.2%
  payment_method             3       3   unseen  0.00%   missing tr/te   7.9%/  7.8%

--- event distribution:
event_type
session           2484909
support_ticket      75821
payment_failed      37387
plan_change         15604
Name: count, dtype: int64
session minutes: {'count': 2484909.0, 'mean': 30.61, 'std': 23.27, 'min': 0.8, '25%': 15.5, '50%': 24.4, '75%': 38.4, 'max': 806.0}
plan_change values: {1.0: 8939, -1.0: 6665}
events per customer: {'count': 97991.0, 'mean': 26.7, 'std': 20.0, 'min': 1.0, '25%': 13.0, '50%': 22.0, '75%': 35.0, 'max': 390.0}

--- basic integrity / leakage checks
duplicate customer_id in train: 0 | in test: 0 | customers in BOTH train and test: 0
exact duplicate event rows: 0
events whose customer is not in train/test: 0
signup after snapshot: 0 (train), 0 (test)
session minutes <=0: 0 | missing: 0 | > 99.9th pct x5: 0
sample_submission columns: ['customer_id', 'churn'] | rows: 40000 | ids match test: True
[    4.7s] Building features ...
[    6.3s]    window   7d done
[    6.4s]    window  14d done
[    6.6s]    window  30d done
[    6.9s]    window  60d done
[    7.3s]    window  90d done
[    7.7s]    window 120d done
[    7.8s]    event features built in 3.0s -> cached
   event-join diagnostics: {'events_joined': 2613721, 'events_after_snapshot_dropped': 0, 'of_which_within_24h_after_snapshot': 0, 'max_age_days': 119.9999771118164}
[    7.9s]    124 features: counts=15, recency=6, windows=78, customer=25

======================================================================
LEAKAGE CHECKS
======================================================================
(a) brute-force recomputation: 1800 values checked on 300 rows -> mismatches: 0
(b) poison test (extreme events injected after snapshot): features unchanged -> True
(c) strongest single numeric features (train, univariate):
                              feature  auc_abs  auc_raw
             active_session_days_30d   0.7377   0.2623
                        sessions_30d   0.7330   0.2670
                 session_minutes_30d   0.7073   0.2927
       ratio_sessions_30d_vs_prev30d   0.7020   0.2980
                         tenure_days   0.7009   0.2991
                       tenure_months   0.7009   0.2991
ratio_session_minutes_30d_vs_prev30d   0.6924   0.3076
                        sessions_14d   0.6923   0.3077
             active_session_days_14d   0.6921   0.3079
            sessions_per_obs_day_30d   0.6890   0.3110
             active_session_days_60d   0.6855   0.3145
                 session_minutes_14d   0.6838   0.3162
                        sessions_60d   0.6757   0.3243
             active_session_days_90d   0.6681   0.3319
                     sessions_prev7d   0.6656   0.3344
    suspicious (>0.85): none
(d) inf values: 0 | columns >50% NaN: ['referral_code_count_so_far', 'days_since_last_support_ticket', 'days_since_last_payment_failure', 'days_since_last_plan_change', 'ratio_tickets_30d_total', 'ratio_failures_30d_total']
[    9.4s] Temporal splits:
   01 -> 2026-02            train= 14993 valid= 15180
   01+02 -> 2026-03         train= 30173 valid= 15014
   01+02+03 -> 2026-04      train= 45187 valid= 14813  <- PRIMARY

======================================================================
TRAIN(2026-04) vs TEST shift check: adversarial AUC = 0.7339  (0.5 = identical)
======================================================================
most shifted features:
 referral_code_count_so_far        45.56
referral_code                     14.46
tenure_days                        3.99
channel_x_plan                     1.87
tenure_months                      1.82
min_session_minutes                0.99
days_since_first_event             0.96
days_since_last_support_ticket     0.95
mean_session_minutes_30d           0.91
price_per_device                   0.89

======================================================================
EXPERIMENT LADDER (lightgbm, rolling temporal validation)
======================================================================
[   12.5s] 1. Customer baseline  [25 features]
      lightgbm  01 -> 2026-02          AUC=0.73470  train AUC=0.7687, iters=4, 1s)
      lightgbm  01+02 -> 2026-03       AUC=0.72764  train AUC=0.8473, iters=107, 1s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.73560  train AUC=0.7898, iters=46, 1s)
[   15.7s] 2. + Event counts  [40 features]
      lightgbm  01 -> 2026-02          AUC=0.78723  train AUC=0.9047, iters=70, 1s)
      lightgbm  01+02 -> 2026-03       AUC=0.77867  train AUC=0.8754, iters=86, 1s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.78474  train AUC=0.8357, iters=48, 1s)
[   19.2s] 3. + Recency  [46 features]
      lightgbm  01 -> 2026-02          AUC=0.84384  train AUC=0.9286, iters=65, 1s)
      lightgbm  01+02 -> 2026-03       AUC=0.83842  train AUC=0.9157, iters=100, 1s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.84467  train AUC=0.8829, iters=56, 1s)
[   23.0s] 4. + Window features  [124 features]
      lightgbm  01 -> 2026-02          AUC=0.85646  train AUC=0.9371, iters=59, 1s)
      lightgbm  01+02 -> 2026-03       AUC=0.85415  train AUC=0.9283, iters=95, 2s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86476  train AUC=0.9130, iters=90, 2s)

======================================================================
FULL FEATURE SET: CatBoost / LightGBM / XGBoost
======================================================================
[   28.0s] 5. CatBoost  [124 features]
      catboost  01 -> 2026-02          AUC=0.86486  train AUC=0.9006, iters=199, 4s)
      catboost  01+02 -> 2026-03       AUC=0.85881  train AUC=0.9085, iters=447, 10s)
      catboost  01+02+03 -> 2026-04    AUC=0.86560  train AUC=0.8885, iters=280, 9s)
[   51.0s] 6. LightGBM  [124 features]
      lightgbm  01 -> 2026-02          AUC=0.85940  train AUC=0.9441, iters=130, 2s)
      lightgbm  01+02 -> 2026-03       AUC=0.85522  train AUC=0.9274, iters=180, 2s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86551  train AUC=0.9001, iters=125, 2s)
[   57.4s] 7. XGBoost  [124 features]
      xgboost   01 -> 2026-02          AUC=0.85968  train AUC=0.9404, iters=138, 1s)
      xgboost   01+02 -> 2026-03       AUC=0.85258  train AUC=0.9449, iters=296, 3s)
      xgboost   01+02+03 -> 2026-04    AUC=0.86412  train AUC=0.9178, iters=213, 3s)

Validation-prediction rank correlation between models:
           catboost  lightgbm  xgboost
catboost    1.0000    0.9582   0.9663
lightgbm    0.9582    1.0000   0.9703
xgboost     0.9663    0.9703   1.0000

======================================================================
TOP 30 FEATURES (catboost, primary split)
======================================================================
                             feature  importance  importance_pct
       ratio_sessions_30d_vs_prev30d       7.787           7.787
                         tenure_days       7.174           7.174
             days_since_last_session       5.859           5.859
                        plan_x_cycle       5.350           5.350
              ratio_sessions_30d_90d       5.039           5.039
                       tenure_months       4.787           4.787
                      payment_method       3.567           3.567
                 support_tickets_60d       3.478           3.478
ratio_session_minutes_30d_vs_prev30d       3.186           3.186
                    family_x_devices       3.080           3.080
       trend_sessions_30d_vs_prev30d       2.419           2.419
                          cycle_days       2.288           2.288
              ratio_sessions_14d_60d       2.275           2.275
                            obs_days       2.275           2.275
                            age_band       1.865           1.865
                              region       1.852           1.852
       ratio_session_minutes_30d_90d       1.759           1.759
                       billing_cycle       1.721           1.721
                       referral_code       1.256           1.256
                     days_to_renewal       1.235           1.235
                payment_failures_30d       1.233           1.233
             active_session_days_30d       1.115           1.115
            ratio_failures_30d_total       1.072           1.072
                    tenure_in_cycles       1.061           1.061
     session_minutes_per_obs_day_90d       1.022           1.022
                        sessions_14d       1.004           1.004
               total_session_minutes       0.922           0.922
                        sessions_30d       0.865           0.865
                      plan_x_payment       0.764           0.764
                                plan       0.745           0.745

Importance share by feature family (families overlap; %):
   sessions (counts+minutes)      45.5
   tenure/renewal                 18.8
   plan & price                   10.7
   recency (days_since_*)          7.5
   support tickets                 6.3
   payment (failures)              2.4
   plan changes (up/down)          1.1

======================================================================
GROUP ABLATION (lightgbm): drop one family from the FULL feature set
======================================================================
[   75.3s]    dropped payment (failures)           (  8 cols): rolling AUC 0.85901  (delta +0.00056)
[   79.5s]    dropped sessions (counts+minutes)    ( 54 cols): rolling AUC 0.83396  (delta -0.02450)
[   85.0s]    dropped support tickets              (  9 cols): rolling AUC 0.85332  (delta -0.00514)
[   91.1s]    dropped plan changes (up/down)       ( 24 cols): rolling AUC 0.85729  (delta -0.00116)
[   96.6s]    dropped recency (days_since_*)       (  6 cols): rolling AUC 0.85588  (delta -0.00257)
[  101.4s]    dropped tenure/renewal               (  7 cols): rolling AUC 0.85262  (delta -0.00584)
[  106.2s]    dropped plan & price                 (  8 cols): rolling AUC 0.85827  (delta -0.00019)
[  109.3s]    dropped ALL event features           ( 99 cols): rolling AUC 0.73265  (delta -0.12581)
                  dropped  n_dropped  primary_auc  rolling_mean  delta_rolling
         (none: full set)          0      0.86476       0.85846        0.00000
       payment (failures)          8      0.86515       0.85901        0.00056
sessions (counts+minutes)         54      0.83735       0.83396       -0.02450
          support tickets          9      0.85944       0.85332       -0.00514
   plan changes (up/down)         24      0.86392       0.85729       -0.00116
   recency (days_since_*)          6      0.86312       0.85588       -0.00257
           tenure/renewal          7      0.85892       0.85262       -0.00584
             plan & price          8      0.86495       0.85827       -0.00019
       ALL event features         99      0.73560       0.73265       -0.12581

==================================================
STREAMBOX CHURN LOCAL EXPERIMENT RESULTS
==================================================

Validation strategy:
  Primary : train on all months before the last, validate on 2026-04 
  Rolling : each month validated on a model trained ONLY on earlier months
  Early stopping uses a random hold-out of the TRAIN part only.

Experiment                     ROC-AUC (2026-04)  rolling mean     std  train AUC  time(s)
------------------------------------------------------------------------------------------
1. Customer baseline                      0.7356        0.7326  0.0036     0.8020        3
2. + Event counts                         0.7847        0.7835  0.0036     0.8719        4
3. + Recency                              0.8447        0.8423  0.0028     0.9091        4
4. + Window features                      0.8648        0.8585  0.0046     0.9261        5
5. CatBoost                               0.8656        0.8631  0.0030     0.8992       23
6. LightGBM                               0.8655        0.8600  0.0042     0.9239        6
7. XGBoost                                0.8641        0.8588  0.0048     0.9344        7
8. Ensemble (rank avg)                    0.8669        0.8627  0.0039          -        0

Per-fold AUC:
            experiment  auc_2026-02  auc_2026-03  auc_2026-04
  1. Customer baseline       0.7347       0.7276       0.7356
     2. + Event counts       0.7872       0.7787       0.7847
          3. + Recency       0.8438       0.8384       0.8447
  4. + Window features       0.8565       0.8542       0.8648
           5. CatBoost       0.8649       0.8588       0.8656
           6. LightGBM       0.8594       0.8552       0.8655
            7. XGBoost       0.8597       0.8526       0.8641
8. Ensemble (rank avg)       0.8636       0.8575       0.8669

Best overall (primary split): 8. Ensemble (rank avg)  ROC-AUC = 0.8669
Best single model: catboost
Event data effect: customer-only 0.7356 -> full ladder 0.8648  (delta +0.1292)
==================================================
Saved to outputs/: experiment_results.csv, oof_predictions.csv, feature_importance.csv, group_ablation.csv
[  109.3s] done

(.venv) kaggle % python streambox_v2.py
Libraries available: {'catboost': True, 'lightgbm': True, 'xgboost': True, 'shap': True} | threads=8 | seed=42
======================================================================
STREAMBOX V2 -- Advanced Feature Engineering
Baseline (V1): primary AUC=0.8669, rolling mean=0.8627
======================================================================
[    1.6s] Loading data ...
[    3.4s] train (60000, 14) | test (40000, 13) | events (2613721, 4) | sample_submission (40000, 2)
[    3.4s] Building features ...
[    5.0s]    window   7d done
[    5.1s]    window  14d done
[    5.3s]    window  30d done
[    5.6s]    window  60d done
[    6.0s]    window  90d done
[    6.4s]    window 120d done
[    6.4s]    V1 features done, building V2 advanced features...
[   37.8s]    session gap features done
[   38.6s]    activity decay features done
[   38.6s]    customer-relative features done
[   38.8s]    consistency features done
[   42.2s]    activity trend features done
[   42.2s]    recency interaction features done
[   42.2s]    support interaction features done
[   42.2s]    plan change interaction features done
[   42.2s]    V2 total features in event block: 181 (V2 new: ~84)
[   42.3s]    event features built in 38.8s -> cached
   event-join diagnostics: {'events_joined': 2613721, 'events_after_snapshot_dropped': 0, 'of_which_within_24h_after_snapshot': 0, 'max_age_days': 119.9999771118164}
[   42.5s]    211 features: counts=15, recency=6, windows=78, gaps=10, decay=39, relative=9, consistency=9, interactions=20, customer=25

======================================================================
LEAKAGE CHECKS
======================================================================
(a) brute-force recomputation: 1800 values checked on 300 rows -> mismatches: 0
(b) poison test (extreme events injected after snapshot): features unchanged -> True
Leakage checks PASSED.
[   43.4s] Temporal splits:
   01 -> 2026-02            train= 14993 valid= 15180
   01+02 -> 2026-03         train= 30173 valid= 15014
   01+02+03 -> 2026-04      train= 45187 valid= 14813  <- PRIMARY

======================================================================
DISTRIBUTION SHIFT ANALYSIS
======================================================================

Feature                               Apr median   May median   Jun median  Apr->May shift
------------------------------------------------------------------------------------------
referral_code_count_so_far                 79.00       108.00       120.00           36.7%
tenure_days                               323.00       239.50       241.00          -25.9%
tenure_months                              10.61         7.87         7.92          -25.9%
min_session_minutes                         6.90         6.90         6.90            0.0%
days_since_first_event                    115.73       114.69       114.84           -0.9%
days_since_last_support_ticket             41.19        40.91        40.37           -0.7%
mean_session_minutes_30d                   23.21        23.10        23.19           -0.5%
price_per_device                            6.41         6.45         6.38            0.5%
sessions_30d                                4.00         4.00         4.00            0.0%
active_session_days_30d                     4.00         4.00         4.00            0.0%

======================================================================
FEATURE SUBSET EXPERIMENTS (lightgbm)
======================================================================
[   43.5s] A. V1 features (baseline)  [124 features]
      lightgbm  01 -> 2026-02          AUC=0.85622  train AUC=0.9479, iters=42, 2s)
      lightgbm  01+02 -> 2026-03       AUC=0.85192  train AUC=0.9629, iters=102, 3s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86450  train AUC=0.9223, iters=61, 3s)
[   52.4s] B. V1 + session gaps  [134 features]
      lightgbm  01 -> 2026-02          AUC=0.85540  train AUC=0.9553, iters=46, 3s)
      lightgbm  01+02 -> 2026-03       AUC=0.85312  train AUC=0.9442, iters=68, 3s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86574  train AUC=0.9306, iters=69, 3s)
[   61.5s] C. V1 + decay/trends  [163 features]
      lightgbm  01 -> 2026-02          AUC=0.85540  train AUC=0.9678, iters=58, 3s)
      lightgbm  01+02 -> 2026-03       AUC=0.85267  train AUC=0.9549, iters=82, 4s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86373  train AUC=0.9248, iters=62, 4s)
[   71.4s] D. V1 + gaps+decay+relative  [182 features]
      lightgbm  01 -> 2026-02          AUC=0.85630  train AUC=0.9680, iters=57, 3s)
      lightgbm  01+02 -> 2026-03       AUC=0.85300  train AUC=0.9567, iters=83, 4s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86592  train AUC=0.9346, iters=74, 4s)
[   82.2s] E. Full V2 features  [210 features]
      lightgbm  01 -> 2026-02          AUC=0.85589  train AUC=0.9746, iters=64, 3s)
      lightgbm  01+02 -> 2026-03       AUC=0.85351  train AUC=0.9509, iters=72, 4s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86558  train AUC=0.9411, iters=82, 4s)

======================================================================
FULL V2 FEATURE SET: CatBoost / LightGBM / XGBoost
======================================================================
[   93.5s] F. CatBoost V2  [210 features]
      catboost  01 -> 2026-02          AUC=0.86457  train AUC=0.9296, iters=407, 11s)
      catboost  01+02 -> 2026-03       AUC=0.85789  train AUC=0.9196, iters=635, 23s)
      catboost  01+02+03 -> 2026-04    AUC=0.86975  train AUC=0.9081, iters=636, 30s)
[  156.8s] G. LightGBM V2  [210 features]
      lightgbm  01 -> 2026-02          AUC=0.85735  train AUC=0.9833, iters=223, 6s)
      lightgbm  01+02 -> 2026-03       AUC=0.85366  train AUC=0.9792, iters=364, 9s)
      lightgbm  01+02+03 -> 2026-04    AUC=0.86767  train AUC=0.9317, iters=193, 6s)
[  177.6s] H. XGBoost V2  [210 features]
      xgboost   01 -> 2026-02          AUC=0.86008  train AUC=0.9426, iters=227, 3s)
      xgboost   01+02 -> 2026-03       AUC=0.85434  train AUC=0.9452, iters=367, 5s)
      xgboost   01+02+03 -> 2026-04    AUC=0.86534  train AUC=0.9244, iters=266, 6s)

Validation-prediction rank correlation between models:
           catboost  lightgbm  xgboost
catboost    1.0000    0.9459   0.9664
lightgbm    0.9459    1.0000   0.9684
xgboost     0.9664    0.9684   1.0000

======================================================================
TOP 50 FEATURES (F. CatBoost V2, primary split)
======================================================================
                             feature  importance  importance_pct
                         tenure_days       5.263           5.263
       ratio_sessions_30d_vs_prev30d       3.807           3.807
                        plan_x_cycle       3.577           3.577
             days_since_last_session       3.465           3.465
               tenure_x_sessions_30d       3.390           3.390
                       tenure_months       3.292           3.292
                    family_x_devices       2.831           2.831
                       billing_cycle       2.321           2.321
                     days_to_renewal       2.265           2.265
                              region       2.208           2.208
                            age_band       2.059           2.059
                 support_tickets_60d       2.034           2.034
              ratio_sessions_30d_90d       1.989           1.989
                      plan_x_payment       1.965           1.965
                          cycle_days       1.954           1.954
                      payment_method       1.744           1.744
         days_since_2nd_last_session       1.594           1.594
                            obs_days       1.580           1.580
            ratio_failures_30d_total       1.546           1.546
                       referral_code       1.301           1.301
ratio_session_minutes_30d_vs_prev30d       1.184           1.184
       trend_sessions_30d_vs_prev30d       1.154           1.154
                price_vs_plan_median       1.148           1.148
         recency_div_active_days_30d       1.097           1.097
                    tenure_in_cycles       1.093           1.093
                    session_mean_gap       1.058           1.058
        recent_30d_vs_total_sessions       1.052           1.052
         diff_sessions_0_30_vs_31_60       1.012           1.012
     recent_30d_vs_total_active_days       0.986           0.986
                      channel_x_plan       0.985           0.985
       ratio_session_minutes_30d_90d       0.980           0.980
                       monthly_price       0.963           0.963
                                plan       0.927           0.927
                     session_min_gap       0.883           0.883
                       tickets_31_60       0.858           0.858
              days_since_first_event       0.755           0.755
                payment_failures_30d       0.732           0.732
         recent_30d_vs_total_minutes       0.730           0.730
               days_since_last_event       0.710           0.710
                 min_session_minutes       0.687           0.687
               gap_last_two_sessions       0.687           0.687
                  session_median_gap       0.622           0.622
        decay_sessions_0_30_vs_31_60       0.601           0.601
     session_minutes_per_obs_day_90d       0.598           0.598
                    price_per_device       0.594           0.594
         decay_minutes_0_30_vs_31_60       0.558           0.558
                      downgrades_60d       0.540           0.540
         sessions_per_active_day_60d       0.533           0.533
              median_session_minutes       0.513           0.513
                     tickets_prev30d       0.507           0.507

--- V2 NEW features in top 50: 16
                        feature  importance_pct
          tenure_x_sessions_30d           3.390
    days_since_2nd_last_session           1.594
    recency_div_active_days_30d           1.097
               session_mean_gap           1.058
   recent_30d_vs_total_sessions           1.052
    diff_sessions_0_30_vs_31_60           1.012
recent_30d_vs_total_active_days           0.986
                session_min_gap           0.883
                  tickets_31_60           0.858
    recent_30d_vs_total_minutes           0.730
          gap_last_two_sessions           0.687
             session_median_gap           0.622
   decay_sessions_0_30_vs_31_60           0.601
    decay_minutes_0_30_vs_31_60           0.558
    sessions_per_active_day_60d           0.533
                tickets_prev30d           0.507

============================================================
STREAMBOX V2 RESULTS
============================================================

Experiment                             Primary AUC   Rolling Mean    Rolling Std
--------------------------------------------------------------------------------
A. V1 features (baseline)                   0.8645         0.8575         0.0052
B. V1 + session gaps                        0.8657         0.8581         0.0055
C. V1 + decay/trends                        0.8637         0.8573         0.0047
D. V1 + gaps+decay+relative                 0.8659         0.8584         0.0055
E. Full V2 features                         0.8656         0.8583         0.0052
F. CatBoost V2                              0.8697         0.8641         0.0049
G. LightGBM V2                              0.8677         0.8596         0.0059
H. XGBoost V2                               0.8653         0.8599         0.0045
I. Ensemble (rank avg)                      0.8693         0.8631         0.0049
J. Ensemble (prob avg)                      0.8698         0.8635         0.0049
K. Ensemble (weighted)                      0.8697         0.8637         0.0049

V1 baseline:
    Best primary AUC = 0.8669
    Best rolling mean = 0.8627

V2 best:
    Experiment = J. Ensemble (prob avg)
    Primary AUC = 0.8698
    Rolling mean = 0.8635
    Rolling std = 0.0049

Improvement:
    Primary = +0.0029
    Rolling = +0.0008

Per-fold AUC:
                 experiment  auc_2026-02  auc_2026-03  auc_2026-04
  A. V1 features (baseline)       0.8562       0.8519       0.8645
       B. V1 + session gaps       0.8554       0.8531       0.8657
       C. V1 + decay/trends       0.8554       0.8527       0.8637
D. V1 + gaps+decay+relative       0.8563       0.8530       0.8659
        E. Full V2 features       0.8559       0.8535       0.8656
             F. CatBoost V2       0.8646       0.8579       0.8697
             G. LightGBM V2       0.8573       0.8537       0.8677
              H. XGBoost V2       0.8601       0.8543       0.8653
     I. Ensemble (rank avg)       0.8629       0.8573       0.8693
     J. Ensemble (prob avg)       0.8629       0.8577       0.8698
     K. Ensemble (weighted)       0.8636       0.8577       0.8697
============================================================
Saved to outputs_v2/: experiment_results_v2.csv, feature_importance_v2.csv
[  191.5s] done

kaggle % ml
(.venv) kaggle % python streambox_v3.py
======================================================================
STREAMBOX V3 -- Hyperparameter Tuning & Feature Selection
V2 baseline: CatBoost primary=0.8697, rolling=0.8641
             Ensemble primary=0.8698, rolling=0.8635
Libraries: {'catboost': True, 'lightgbm': True, 'xgboost': True, 'shap': True} | threads=8
======================================================================
[    1.9s] Loading data ...
[    3.8s] train (60000, 14) | test (40000, 13) | events (2613721, 4) | sample_submission (40000, 2)
[    3.8s] Building features ...
[    3.9s]    loading cached event features: cache/event_feats_b77bbab6487d.pkl
   event-join diagnostics: {'events_joined': 2613721, 'events_after_snapshot_dropped': 0, 'of_which_within_24h_after_snapshot': 0, 'max_age_days': 119.9999771118164}
[    4.2s]    211 features: counts=15, recency=6, windows=78, gaps=10, decay=39, relative=9, consistency=9, interactions=20, customer=25
[    4.3s] Temporal splits:
   01 -> 2026-02            train= 14993 valid= 15180
   01+02 -> 2026-03         train= 30173 valid= 15014
   01+02+03 -> 2026-04      train= 45187 valid= 14813  <- PRIMARY

======================================================================
STEP 0: VERIFY V2 BASELINE
======================================================================
[    2.4s] V2 CatBoost baseline
      catboost  01 -> 2026-02          AUC=0.86319  train=0.9243 iters=364 11s
      catboost  01+02 -> 2026-03       AUC=0.85896  train=0.9199 iters=638 22s
      catboost  01+02+03 -> 2026-04    AUC=0.86985  train=0.9082 iters=651 27s
   -> primary=0.86985  rolling=0.86400  std=0.00449
[   62.9s] V2 LightGBM baseline
      lightgbm  01 -> 2026-02          AUC=0.85714  train=0.9821 iters=217 5s
      lightgbm  01+02 -> 2026-03       AUC=0.85543  train=0.9695 iters=293 7s
      lightgbm  01+02+03 -> 2026-04    AUC=0.86704  train=0.9388 iters=224 7s
   -> primary=0.86704  rolling=0.85987  std=0.00512
[   83.1s] V2 XGBoost baseline
      xgboost   01 -> 2026-02          AUC=0.85871  train=0.9407 iters=209 2s
      xgboost   01+02 -> 2026-03       AUC=0.85396  train=0.9483 iters=401 6s
      xgboost   01+02+03 -> 2026-04    AUC=0.86517  train=0.9160 iters=208 5s
   -> primary=0.86517  rolling=0.85928  std=0.00459

======================================================================
STEP 1: CATBOOST HYPERPARAMETER SEARCH
======================================================================
[   96.5s] Screening CatBoost candidates on PRIMARY fold only...
   [ 0] d=6 lr=0.03 l2=3.0 rs=0.5 bt=0.3 -> April=0.86977 32s
   [ 1] d=7 lr=0.02 l2=5.0 rs=0.5 bt=0.3 -> April=0.86988 46s
   [ 2] d=8 lr=0.02 l2=5.0 rs=0.5 bt=0.5 -> April=0.86888 42s
   [ 3] d=8 lr=0.03 l2=3.0 rs=1.0 bt=0.5 -> April=0.86784 29s
   [ 4] d=7 lr=0.03 l2=10.0 rs=0.5 bt=0.3 -> April=0.86909 25s
   [ 5] d=9 lr=0.02 l2=10.0 rs=1.0 bt=0.5 -> April=0.86901 58s
   [ 6] d=6 lr=0.05 l2=5.0 rs=0 bt=0 -> April=0.86888 20s
   [ 7] d=7 lr=0.05 l2=3.0 rs=0 bt=0 -> April=0.86772 25s
   [ 8] d=8 lr=0.03 l2=10.0 rs=0.5 bt=1.0 -> April=0.86930 35s
   [ 9] d=10 lr=0.02 l2=20.0 rs=1.0 bt=1.0 -> April=0.86843 90s
   [10] d=7 lr=0.03 l2=5.0 rs=0 bt=0.5 -> April=0.86908 26s
   [11] d=8 lr=0.02 l2=3.0 rs=0.5 bt=0 -> April=0.86837 81s

   Top 4 candidates -> full rolling evaluation:
   [1] primary=0.86988 rolling=0.86413 std=0.00468
   [0] primary=0.86977 rolling=0.86439 std=0.00430
   [8] primary=0.86930 rolling=0.86380 std=0.00451
   [4] primary=0.86909 rolling=0.86368 std=0.00429
[  979.2s] Best CatBoost: rolling=0.86439 primary=0.86977

======================================================================
STEP 2: LIGHTGBM HYPERPARAMETER SEARCH
======================================================================
[  979.2s] Screening LightGBM candidates on PRIMARY fold...
   [ 0] nl=31 mcs=30 ss=0.8 cbt=0.6 -> April=0.86851 6s
   [ 1] nl=63 mcs=50 ss=0.8 cbt=0.7 -> April=0.86630 7s
   [ 2] nl=63 mcs=20 ss=0.7 cbt=0.5 -> April=0.86855 8s
   [ 3] nl=127 mcs=50 ss=0.8 cbt=0.6 -> April=0.86634 13s
   [ 4] nl=63 mcs=30 ss=0.85 cbt=0.65 -> April=0.86723 7s
   [ 5] nl=47 mcs=40 ss=0.8 cbt=0.6 -> April=0.86807 6s
   [ 6] nl=63 mcs=30 ss=0.75 cbt=0.55 -> April=0.86852 8s
   [ 7] nl=95 mcs=30 ss=0.8 cbt=0.6 -> April=0.86591 10s

   Top 4 -> full rolling:
   [2] primary=0.86855 rolling=0.86001 std=0.00606
   [6] primary=0.86852 rolling=0.86046 std=0.00571
   [0] primary=0.86851 rolling=0.86111 std=0.00538
   [5] primary=0.86807 rolling=0.86088 std=0.00514
[ 1121.8s] Best LightGBM: rolling=0.86111 primary=0.86851

======================================================================
STEP 3: XGBOOST HYPERPARAMETER SEARCH
======================================================================
[ 1121.8s] Screening XGBoost candidates on PRIMARY fold...
   [ 0] md=6 mcw=10 ss=0.8 cbt=0.6 -> April=0.86579 8s
   [ 1] md=7 mcw=5 ss=0.8 cbt=0.7 -> April=0.86561 9s
   [ 2] md=8 mcw=10 ss=0.75 cbt=0.55 -> April=0.86639 9s
   [ 3] md=7 mcw=15 ss=0.85 cbt=0.6 -> April=0.86575 8s
   [ 4] md=6 mcw=5 ss=0.8 cbt=0.65 -> April=0.86573 8s
   [ 5] md=8 mcw=5 ss=0.8 cbt=0.6 -> April=0.86548 9s
   [ 6] md=9 mcw=15 ss=0.8 cbt=0.5 -> April=0.86433 9s
   [ 7] md=7 mcw=10 ss=0.8 cbt=0.6 -> April=0.86522 8s

   Top 4 -> full rolling:
   [2] primary=0.86639 rolling=0.85970 std=0.00516
   [0] primary=0.86579 rolling=0.85996 std=0.00453
   [3] primary=0.86575 rolling=0.85984 std=0.00452
   [4] primary=0.86573 rolling=0.86010 std=0.00455
[ 1264.3s] Best XGBoost: rolling=0.86010 primary=0.86573

======================================================================
STEP 4: FEATURE SELECTION (CatBoost)
======================================================================
[ 1307.9s] Feature selection: top 150
    top-150: primary=0.86976 rolling=0.86439 std=0.00433
[ 1376.4s] Feature selection: top 100
   top-100: primary=0.86937 rolling=0.86445 std=0.00404
[ 1420.3s] Feature selection: top 75
   top-75: primary=0.86973 rolling=0.86537 std=0.00383
[ 1458.8s] Feature selection: top 50
   top-50: primary=0.86713 rolling=0.86475 std=0.00271

======================================================================
STEP 5: BEHAVIORAL vs CUSTOMER-HEAVY FEATURES
======================================================================
[ 1491.4s] Behavioral-heavy (drop high-shift customer feats)
   behavioral-heavy: primary=0.86846 rolling=0.86189
[ 1547.5s] Customer-only (no event features)
   customer-only: primary=0.73709 rolling=0.74031

======================================================================
STEP 6: ENSEMBLE OPTIMIZATION
======================================================================
[ 1574.9s] Collecting OOF predictions from best tuned models...

Top 10 ensemble weight combinations:
 w_cb  w_lgb  w_xgb  rolling_prob  rolling_rank  best_rolling
 0.70   0.25   0.05      0.865125      0.864885      0.865125
 0.65   0.30   0.05      0.865119      0.864866      0.865119
 0.75   0.20   0.05      0.865092      0.864864      0.865092
 0.60   0.35   0.05      0.865072      0.864809      0.865072
 0.65   0.25   0.10      0.865061      0.864817      0.865061
 0.70   0.20   0.10      0.865050      0.864821      0.865050
 0.60   0.30   0.10      0.865039      0.864771      0.865039
 0.80   0.15   0.05      0.865020      0.864820      0.865020
 0.75   0.15   0.10      0.864995      0.864792      0.864995
 0.65   0.20   0.15      0.864983      0.864741      0.864983

Best ensemble: type=prob weights=CB:0.7 LGB:0.25 XGB:0.05
   Per-fold: {'2026-02': 0.8642705797196816, '2026-03': 0.8600099318605192, '2026-04': 0.8710949592880696}
   Primary=0.87109  Rolling=0.86513  Std=0.00457

Pre-specified ensemble combos:
   prob CB=0.70 LGB=0.20 XGB=0.10 -> rolling=0.86505 primary=0.87094
   rank CB=0.70 LGB=0.20 XGB=0.10 -> rolling=0.86482 primary=0.87045
   prob CB=0.60 LGB=0.25 XGB=0.15 -> rolling=0.86498 primary=0.87101
   rank CB=0.60 LGB=0.25 XGB=0.15 -> rolling=0.86471 primary=0.87044
   prob CB=0.80 LGB=0.10 XGB=0.10 -> rolling=0.86489 primary=0.87055
   rank CB=0.80 LGB=0.10 XGB=0.10 -> rolling=0.86474 primary=0.87020
   prob CB=0.50 LGB=0.30 XGB=0.20 -> rolling=0.86479 primary=0.87097
   rank CB=0.50 LGB=0.30 XGB=0.20 -> rolling=0.86448 primary=0.87031
   prob CB=0.33 LGB=0.33 XGB=0.33 -> rolling=0.86416 primary=0.87051
   rank CB=0.33 LGB=0.33 XGB=0.33 -> rolling=0.86377 primary=0.86973

======================================================================
STEP 7: SEED ROBUSTNESS CHECK
======================================================================
   seed=42: primary=0.86977 rolling=0.86439 std=0.00430
   seed=123: primary=0.87055 rolling=0.86494 std=0.00485
   seed=2026: primary=0.87009 rolling=0.86507 std=0.00439
   Primary range: 0.86977 - 0.87055 (spread=0.00078)
   Rolling range: 0.86439 - 0.86507 (spread=0.00067)

======================================================================
STEP 8: TOP 30 FEATURES (Best CatBoost)
======================================================================
                             feature  importance_pct
                      payment_method           3.829
                         tenure_days           3.722
                        plan_x_cycle           3.404
               tenure_x_sessions_30d           3.342
       ratio_sessions_30d_vs_prev30d           3.239
                    family_x_devices           2.888
             days_since_last_session           2.840
                       tenure_months           2.727
        decay_sessions_0_30_vs_31_60           2.418
                       billing_cycle           2.350
                   recency_vs_tenure           2.203
                          cycle_days           2.081
                              region           1.985
ratio_session_minutes_30d_vs_prev30d           1.764
                 support_tickets_60d           1.685
                            age_band           1.647
                            obs_days           1.494
            ratio_failures_30d_total           1.385
                     days_to_renewal           1.343
                       referral_code           1.283
         days_since_2nd_last_session           1.182
       trend_sessions_30d_vs_prev30d           1.181
         recency_div_active_days_30d           1.162
       ratio_session_minutes_30d_90d           1.159
                       monthly_price           1.158
                price_vs_plan_median           1.149
                    session_mean_gap           1.092
     recent_30d_vs_total_active_days           1.063
                 min_session_minutes           1.006
              ratio_sessions_30d_90d           1.004

============================================================
STREAMBOX V3 FINAL RESULTS
============================================================

Experiment                                         Feb      Mar      Apr     Mean      Std
-------------------------------------------------------------------------------------
V2 CatBoost                                     0.8632   0.8590   0.8699   0.8640   0.0045
V2 LightGBM                                     0.8571   0.8554   0.8670   0.8599   0.0051
V2 XGBoost                                      0.8587   0.8540   0.8652   0.8593   0.0046
V3 CatBoost tuned                               0.8642   0.8592   0.8698   0.8644   0.0043
V3 LightGBM tuned                               0.8589   0.8559   0.8685   0.8611   0.0054
V3 XGBoost tuned                                0.8600   0.8546   0.8657   0.8601   0.0045
V3 CB top-150                                   0.8643   0.8592   0.8698   0.8644   0.0043
V3 CB top-100                                   0.8645   0.8595   0.8694   0.8645   0.0040
V3 CB top-75                                    0.8660   0.8604   0.8697   0.8654   0.0038
V3 CB top-50                                    0.8662   0.8610   0.8671   0.8647   0.0027
V3 behavioral-heavy                             0.8621   0.8551   0.8685   0.8619   0.0055
V3 customer-only                                0.7482   0.7356   0.7371   0.7403   0.0056
V3 Ensemble (prob 0.3333333333333333/0.3333333333333333/0.3333333333333333)   0.8643   0.8600   0.8711   0.8651   0.0046
V3 Equal prob                                   0.8630   0.8589   0.8705   0.8642   0.0048
V3 Equal rank                                   0.8630   0.8586   0.8697   0.8638   0.0046

V2 baseline:
    Best primary AUC = 0.8698
    Best rolling mean = 0.8641

V3 best (by rolling mean):
    Experiment = V3 CB top-75
    Primary AUC = 0.86973
    Rolling mean = 0.86537
    Rolling std = 0.00383

Improvement over V2:
    Primary = -0.00007
    Rolling = +0.00127

Best CatBoost params: {'iterations': 5000, 'eval_metric': 'AUC', 'od_type': 'Iter', 'od_wait': 200, 'verbose': 0, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0, 'random_strength': 0.5, 'bagging_temperature': 0.3}
Best LightGBM params: {'n_estimators': 5000, 'learning_rate': 0.02, 'subsample_freq': 1, 'verbose': -1, 'num_leaves': 31, 'min_child_samples': 30, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 0.1, 'reg_lambda': 2.0}
Best XGBoost params: {'n_estimators': 5000, 'learning_rate': 0.02, 'tree_method': 'hist', 'enable_categorical': True, 'eval_metric': 'auc', 'max_depth': 6, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.65, 'reg_alpha': 0.0, 'reg_lambda': 2.0, 'gamma': 0.1}
Best ensemble: type=prob weights=CB:0.7 LGB:0.25 XGB:0.05

Seed stability (CatBoost):
    Primary spread = 0.00078
    Rolling spread = 0.00067

Leakage status: V2 leakage checks passed (brute-force + poison test)
Feature count: 211

============================================================
RECOMMENDATION
============================================================

Best single model: V3 CB top-75
    Rolling = 0.86537, Primary = 0.86973

Best ensemble: V3 Ensemble (prob 0.3333333333333333/0.3333333333333333/0.3333333333333333)
    Rolling = 0.86513, Primary = 0.87109

-> RECOMMEND: V3 CB top-75 for final Kaggle notebook

Saved to outputs_v3/: v3_results.csv, feature_importance_v3.csv
[ 1816.8s] done
(.venv) kaggle %  python submission_notebook.py 
Input: data | Output: outputs_submission | Threads: 8
[   0.0s] Loading data...
[   1.9s] train (60000, 14) | test (40000, 13) | events (2613721, 4) | sample_sub (40000, 2)
============================================================
DATA INSPECTION
============================================================

train columns: ['customer_id', 'snapshot_date', 'signup_date', 'plan', 'billing_cycle', 'monthly_price', 'region', 'age_band', 'acquisition_channel', 'referral_code', 'payment_method', 'num_devices', 'family_sharing', 'churn']
test columns:  ['customer_id', 'snapshot_date', 'signup_date', 'plan', 'billing_cycle', 'monthly_price', 'region', 'age_band', 'acquisition_channel', 'referral_code', 'payment_method', 'num_devices', 'family_sharing']
events columns: ['customer_id', 'timestamp', 'event_type', 'value']
sample_sub columns: ['customer_id', 'churn']

Target distribution: {0: 0.8806, 1: 0.1194}

Missing values (train):
age_band           3636
referral_code     45554
payment_method     4757
dtype: int64

Missing values (test):
age_band           2401
referral_code     30884
payment_method     3131
dtype: int64

Event types: {'session': 2484909, 'support_ticket': 75821, 'payment_failed': 37387, 'plan_change': 15604}
Events per customer: mean=26.7, median=22.0

train snapshot: 2026-01-31 .. 2026-04-30
test snapshot:  2026-05-31 .. 2026-06-30
events range:   2025-10-03 00:00:17 .. 2026-06-29 23:59:36

Customer overlap train/test: 0
sample_sub IDs match test: True
[   2.2s] Building features...
[   4.0s]    window   7d done
[   4.1s]    window  14d done
[   4.2s]    window  30d done
[   4.6s]    window  60d done
[   5.0s]    window  90d done
[   5.5s]    window 120d done
[   5.5s]    V1 features done, building V2/V3 advanced features...
[  38.1s]    session gap features done
[  38.9s]    activity decay features done
[  38.9s]    customer-relative features done
[  39.1s]    consistency features done
[  42.6s]    activity trend features done
[  42.6s]    recency interactions done
[  42.6s]    support interactions done
[  42.6s]    plan change interactions done
[  42.6s]    Event features total: 181
   event-join diagnostics: {'events_joined': 2613721, 'events_after_snapshot_dropped': 0, 'max_age_days': 119.9999771118164}
[  42.9s] Total features: 211
   counts=15, recency=6, windows=78, gaps=10, decay=39, relative=9, consistency=9, interactions=20, customer=25
============================================================
FEATURE VALIDATION
============================================================
Total features: 211
Numeric features: 201
Categorical features: 10
Inf values: 0
Columns >50% NaN: ['referral_code_count_so_far', 'days_since_last_support_ticket', 'days_since_last_payment_failure', 'days_since_last_plan_change', 'ratio_tickets_30d_total', 'ratio_failures_30d_total', 'session_slope_14d', 'tickets_30d_vs_prev30d']
'churn' in feature_cols: False
'customer_id' in feature_cols: False
Feature validation PASSED.
============================================================
TEMPORAL VALIDATION
============================================================
[  44.6s] Temporal splits:
   01 -> 2026-02            train= 14993 valid= 15180
   01+02 -> 2026-03         train= 30173 valid= 15014
   01+02+03 -> 2026-04      train= 45187 valid= 14813  <- PRIMARY
[  45.0s] CatBoost temporal validation
   01 -> 2026-02            AUC=0.86418 iters=526 13s
   01+02 -> 2026-03         AUC=0.85923 iters=708 22s
   01+02+03 -> 2026-04      AUC=0.86977 iters=992 38s
   Rolling mean: 0.86439
[ 117.1s] LightGBM temporal validation
   01 -> 2026-02            AUC=0.85890 iters=257 4s
   01+02 -> 2026-03         AUC=0.85591 iters=491 7s
   01+02+03 -> 2026-04      AUC=0.86851 iters=283 7s
   Rolling mean: 0.86111
[ 135.2s] XGBoost temporal validation
   01 -> 2026-02            AUC=0.85996 iters=256 4s
   01+02 -> 2026-03         AUC=0.85460 iters=564 11s
   01+02+03 -> 2026-04      AUC=0.86573 iters=280 9s
   Rolling mean: 0.86010
============================================================
VALIDATION ENSEMBLE
============================================================

Ensemble weights: CatBoost=0.7 LightGBM=0.25 XGBoost=0.05

Model                     Feb      Mar      Apr     Mean
-------------------------------------------------------
CatBoost              0.86418  0.85923  0.86977  0.86439
LightGBM              0.85890  0.85591  0.86851  0.86111
XGBoost               0.85996  0.85460  0.86573  0.86010
Ensemble              0.86427  0.86001  0.87109  0.86513

Primary (April) AUC: 0.87109
Rolling mean AUC:    0.86513
[ 158.8s] Training final models on FULL training data...
[ 204.5s] CatBoost final: 641 iterations
[ 211.1s] LightGBM final: 300 iterations
[ 220.5s] XGBoost final: 296 iterations
Test prediction statistics:
  CatBoost: min=0.0015 max=0.9601 mean=0.1269
  LightGBM: min=0.0049 max=0.8812 mean=0.1272
  XGBoost:  min=0.0029 max=0.9030 mean=0.1262
  Ensemble: min=0.0025 max=0.9336 mean=0.1269

Spearman rank correlation:
  CB-LGB: 0.9694 | CB-XGB: 0.9756 | LGB-XGB: 0.9789
[ 220.6s] Submission saved to outputs_submission/submission.csv
==================================================
SUBMISSION VERIFICATION
==================================================
Path:             outputs_submission/submission.csv
Shape:            (40000, 2)
Columns:          ['customer_id', 'churn']
Prediction min:   0.002544
Prediction max:   0.933639
Prediction mean:  0.126923
Unique customers: 40000
NaN predictions:  0

All assertions PASSED.

Validation summary:
  CatBoost rolling:  0.86439
  LightGBM rolling:  0.86111
  XGBoost rolling:   0.86010
  Ensemble rolling:  0.86513
  Ensemble primary:  0.87109

==================================================
SUBMISSION READY
==================================================
Path: outputs_submission/submission.csv
Rows: 40000
Columns: ['customer_id', 'churn']
Prediction min: 0.002544
Prediction max: 0.933639
Prediction mean: 0.126923
Unique customers: 40000
NaN predictions: 0
==================================================
[ 220.7s] DONE