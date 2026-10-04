# StreamBox Churn Prediction

## Overview
StreamBox is a fictional subscription service. Each month some customers cancel their subscription. Your job is to predict, for each customer, the probability that they will cancel (churn) in the 30 days after a given snapshot date.

This is a 24-hour selection round. The 12 best-ranked eligible teams will go through to the next stage.

You get two kinds of data: a customer profile table and a large activity log. Strong solutions will need to turn the activity log into useful features.

## How selection works
When the competition ends, teams are ranked on the private leaderboard.
- The top 20 teams go through verification
- The 12 highest-ranked teams that pass verification are selected.
- Ties are broken by the earliest submission time.

## Evaluation
Submissions are evaluated on the Area Under the ROC Curve (AUC) between the predicted probability and the observed target.

## Submission File
For each `customer_id` in the test set, predict the probability that the customer churns. The file must contain a header and have the following format:

```csv
customer_id,churn
C3bed7656,0.12
C8b536ea2,0.03
...
```

The public leaderboard is calculated on about 30% of the test data. The final ranking uses the remaining 70% (private leaderboard). You may select up to 2 final submissions.

## Dataset Description
Each customer appears once, at a snapshot date (the last day of a month). The target is whether they cancel within the 30 days after that date. Training customers have snapshot dates from January to April 2026; test customers have snapshot dates in May and June 2026.

Only the data provided in this competition may be used.

### Files
- `train.csv`: 60,000 customers with the target churn
- `test.csv`: 40,000 customers; predict churn for these
- `events.csv`: about 2.6 million activity events for all customers, covering up to 120 days before each customer's snapshot date
- `sample_submission.csv`: a correctly formatted submission

### train.csv / test.csv
| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `snapshot_date` | Date the prediction is made |
| `signup_date` | Date the customer signed up |
| `plan` | Subscription tier: basic, standard, premium |
| `billing_cycle` | monthly or annual |
| `monthly_price` | Effective monthly price paid |
| `region` | Customer region |
| `age_band` | Age group (may be missing) |
| `acquisition_channel` | How the customer was acquired |
| `referral_code` | Referral or partner code used at signup (only for some channels) |
| `payment_method` | Payment method on file (may be missing) |
| `num_devices` | Number of registered devices |
| `family_sharing` | 1 if family sharing is enabled |
| `churn` | Target (train only): 1 if the customer cancelled within 30 days after snapshot_date |

### events.csv
| Column | Description |
|---|---|
| `customer_id` | Customer identifier (joins to train/test) |
| `timestamp` | Date and time of the event |
| `event_type` | `session`, `support_ticket`, `payment_failed`, `plan_change` |
| `value` | For `session`: minutes used. For `plan_change`: +1 upgrade, -1 downgrade. Otherwise empty |
