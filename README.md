# FinShield — Financial Transaction Anomaly Detection

**Domain:** Financial Crime · Fraud Detection · Risk Scoring
**Stack:** Python · NumPy · Pandas · SQLite · Matplotlib · Seaborn
**Author:** Aparajita Mondal
**Dataset:** 748,065 synthetic transactions · $95.5M total volume · 181-day period

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Problem Matters](#why-this-problem-matters)
3. [Architecture & Pipeline](#architecture--pipeline)
4. [Dataset Design](#dataset-design)
5. [Fraud Patterns Injected](#fraud-patterns-injected)
6. [Detection Methods](#detection-methods)
7. [Findings & Analysis](#findings--analysis)
8. [Business Impact](#business-impact)
9. [Key Visualizations](#key-visualizations)
10. [How to Run](#how-to-run)
11. [What I Would Build Next](#what-i-would-build-next)

---

## Project Overview

FinShield is an end-to-end financial transaction monitoring system built entirely in Python — from synthetic data generation through ETL, multi-method anomaly detection, risk scoring, and visual analytics.

The project simulates 6 months of real-world credit/debit card activity for 2,000 cardholders across 10 major US cities, injects 5 distinct fraud attack patterns into the dataset with ground-truth labels, then runs a rule-based + statistical detection engine that assigns every transaction a composite risk score and an alert tier.

The entire system mirrors the architecture of a production fraud platform: a raw landing zone (CSV/simulated stream), a transformation layer that computes rolling baselines and velocity features, a structured SQLite warehouse (production equivalent: AWS Redshift or Athena over S3), and a detection engine that outputs to the same warehouse for dashboarding.

---

## Why This Problem Matters

Payment fraud cost the global economy **$33.8 billion in 2023** — a number that keeps climbing as card-not-present transactions and real-time payments grow. Financial institutions are looking for data professionals who can bridge the gap between raw transaction logs and actionable risk signals.

This project demonstrates that capability: starting with messy, high-volume transaction data and ending with a precision-scored, tiered alert queue that a fraud operations team could act on directly.

**The specific skills showcased here are in high demand for:**
- Data Engineer / Data Platform roles (ETL design, warehouse schema, indexing strategy)
- Analytics Engineer roles (feature engineering, rolling windows, behavioral baselines)
- Risk & Fraud Analytics roles (detection logic, precision/recall trade-offs, business impact modelling)

---

## Architecture & Pipeline

```
Raw Data (CSV)                       Warehouse (SQLite)
──────────────                       ──────────────────
transactions_raw.csv  ──[EXTRACT]──▶  transactions_clean
cardholders.csv                       card_baselines
                      ──[TRANSFORM]─▶  daily_kpis
                                       merchant_stats
                      ──[LOAD]──────▶
                                      ──[DETECT]──────▶  detection_results
                                                          detection_performance
```

**Four-step orchestration (`main.py`):**

| Step | Module | What It Does |
|------|--------|-------------|
| 1 | `generate_data.py` | Synthesise 2,000 cardholders + 6 months of transactions with injected fraud |
| 2 | `pipeline.py` | Extract → Validate → Enrich → Load into SQLite warehouse |
| 3 | `anomaly_detection.py` | Five detection methods → composite risk score → alert tier |
| 4 | `visualizations.py` | Six production-quality charts saved to `outputs/` |

**Warehouse schema:**

```sql
-- transactions_clean: full enriched transaction log
-- Indexes: (card_id, timestamp), is_fraud, category

-- card_baselines: per-cardholder spend profile
-- Aggregates: total_txns, total_spend, night_txn_pct, unique_cities, unique_merchants

-- daily_kpis: platform-level daily metrics
-- Fields: total_txns, total_volume, unique_cards, fraud_count, fraud_rate

-- merchant_stats: merchant-level risk aggregates
-- Fields: txn_count, total_volume, unique_cards, fraud_count, fraud_rate
```

---

## Dataset Design

**2,000 cardholders** across four income tiers:

| Tier | Avg Transaction | Std Dev | Txns/Day | Population Share |
|------|----------------|---------|----------|-----------------|
| Low | $32 | $18 | 1.8 | 30% |
| Mid | $68 | $45 | 2.5 | 40% |
| High | $145 | $120 | 3.2 | 22% |
| Premium | $380 | $280 | 4.0 | 8% |

Each cardholder has a **behavioural fingerprint**: home city, 3 preferred merchant categories, typical active hours, and an income-driven spending baseline. This fingerprint is the foundation for every detection method — anomalies are measured relative to each individual's normal behaviour, not a population average.

**Transaction volume:**

| Metric | Value |
|--------|-------|
| Total transactions | 748,065 |
| Total volume | $95,502,410.79 |
| Date range | 1 Jan 2024 – 29 Jun 2024 (181 days) |
| Avg daily transactions | 4,133 |
| Avg daily volume | $527,638 |
| Peak daily transactions | 4,373 |
| Avg transaction amount | $127.67 |

**10 cities modelled:** New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, Dallas, Austin, Seattle — with geographic coordinates used for Haversine-based impossible travel detection.

---

## Fraud Patterns Injected

Five distinct fraud attack vectors, ground-truth labelled, injected into the clean dataset:

| Fraud Type | Transactions | % of Fraud | Description |
|------------|-------------|------------|-------------|
| VELOCITY_ABUSE | 668 | 40.3% | 10–18 transactions within a 30-minute window |
| CARD_TESTING | 648 | 39.1% | 8–15 rapid micro-transactions ($0.50–$2.50) testing card validity |
| ACCOUNT_TAKEOVER | 161 | 9.7% | Sudden shift to unusual merchant categories + foreign city, same day |
| AMOUNT_SPIKE | 100 | 6.0% | Single transaction 8–20× above cardholder's 30-day average |
| GEO_IMPOSSIBLE | 80 | 4.8% | Two transactions 1,000+ miles apart within 3 hours |
| **Total** | **1,657** | **0.222%** | Realistic class imbalance for production fraud datasets |

**Class imbalance note:** 0.222% fraud prevalence is intentional and realistic — most production fraud systems operate at 0.1%–0.5% base rates. This imbalance is the core reason naive classifiers fail and why precision/recall analysis is more meaningful than accuracy.

**Fraud by income tier:**

| Tier | Fraud Transactions | Fraud Rate |
|------|-------------------|------------|
| Low | 500 | 0.32% |
| Mid | 612 | 0.22% |
| High | 408 | 0.19% |
| Premium | 137 | 0.14% |

Low-income cardholders show the highest fraud rate — consistent with real-world patterns where lower-income accounts have weaker secondary authentication and are more frequently targeted for card testing.

**Fraud by category:**

| Category | Fraud Transactions |
|----------|--------------------|
| Retail | 481 |
| Online Shopping | 430 |
| Travel | 144 |
| Dining | 128 |
| Entertainment | 97 |

Online Shopping and Retail dominate because card testing and velocity abuse attacks cluster in these categories — bot-driven micro-transactions at low-friction online merchants.

**Fraud by time of day:**
- Day fraud (6am–10pm): 978 transactions (59%)
- Night fraud (10pm–6am): 679 transactions (41%)

Night fraud has a higher *concentration* — with far fewer legitimate night transactions in the dataset, any night-time anomaly carries a stronger signal.

---

## Detection Methods

### Method 1: Z-Score Amount Deviation

**Principle:** A cardholder's 30-day rolling average and standard deviation are computed per transaction (window = 720 transactions, min 5 periods). Each transaction receives an `amount_z` score. Transactions where `|amount_z| > 4` are flagged.

**Feature engineering:**
```python
rolling_avg_30d = groupby(card_id)[amount].rolling(720, min_periods=5).mean()
rolling_std_30d = groupby(card_id)[amount].rolling(720, min_periods=5).std()
amount_z = (amount - rolling_avg_30d) / rolling_std_30d
```

**Threshold rationale:** 4σ is conservative — a 3σ threshold would catch more fraud but generate ~3× more false positives. 4σ was chosen to keep precision high while the composite scorer handles borderline cases.

**Performance:**
- Precision: 0.758 — when it fires, it's usually right
- Recall: 0.142 — only catches 14% of all fraud
- F1: 0.239

**Analysis:** High precision makes this a high-confidence signal — when the z-score fires alone, it correctly identifies fraud 75.8% of the time. The low recall is expected: it only captures AMOUNT_SPIKE and the most extreme ACCOUNT_TAKEOVER transactions. The 235 true positives it catches at 75.8% precision are worth acting on immediately.

---

### Method 2: Velocity Rule Engine

**Principle:** Count the number of transactions per card within any 30-minute rolling window (`velocity_30m`). Flag cards exceeding 5 transactions in that window.

**Implementation:** For each card, a sliding window comparison over Unix timestamps identifies bursts. `velocity_30m > 5` triggers the flag; score scales with volume.

**Performance:**
- Precision: 0.987 — nearly perfect, almost no false alarms
- Recall: 0.466 — catches 46.6% of all fraud
- F1: 0.633 — **best single-method performer**

**Analysis:** This is the standout method. 98.7% precision at 46.6% recall is exceptional for a rule-based system. The near-perfect precision comes from the asymmetry of the signal — legitimate cardholders almost never make 6+ purchases in 30 minutes, so when the rule fires, it is almost always a bot-driven attack (VELOCITY_ABUSE or CARD_TESTING).

The 46.6% recall means roughly half the fraud goes undetected by velocity alone — those are the subtler attacks (AMOUNT_SPIKE, GEO_IMPOSSIBLE, ACCOUNT_TAKEOVER) that don't cluster in time windows.

**Production implication:** This rule should trigger an immediate card block or real-time SMS verification. The cost of a false positive (a temporary card hold) is far outweighed by the precision — only 10 legitimate transactions were flagged across 748,065.

---

### Method 3: Geo-Impossible Detection

**Principle:** For consecutive transactions on the same card, compute Haversine distance between transaction cities. Flag any pair where the cardholder would need to travel >500 miles in under 4 hours (physically impossible without a private jet).

**Implementation:**
```python
distance = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
if distance > 500 and time_diff_hours < 4:
    flag = True
    score = min(100, distance / 15)
```

**Performance:**
- Precision: 0.013 — fires frequently, often wrong
- Recall: 0.385 — catches 38.5% of fraud
- F1: 0.025

**Analysis:** This result looks alarming at first glance — 1.3% precision seems like a broken detector. But the analysis reveals a nuanced picture:

The low precision stems from a dataset design characteristic: cardholders travel legitimately (5% travel rate in data generation), and with only 10 cities modelled, many "impossible" flags are actually legitimate same-day flights. The algorithm catches 638 true fraud transactions but fires 48,924 times on legitimate travel — a false positive rate driven by the coarse city-level granularity.

**What this teaches:** Geo-velocity detection works well when you have full GPS coordinates, merchant-level location, and IP geolocation cross-referenced. With city-level data only, the signal is noisy. In a production system, this flag should only trigger in combination with other signals (as it does in the composite score), never alone.

Despite the low precision, the **38.5% recall is the second highest of any individual method** — it's finding fraud that the other methods miss. This is why it carries a 25% weight in the composite score.

---

### Method 4: Night + Foreign City Composite

**Principle:** Night-time transactions (10pm–6am) in a city other than the cardholder's home city. Each signal is weak alone; together they form a stronger indicator.

**Performance:**
- Precision: 0.000
- Recall: 0.000
- F1: 0.000

**Analysis:** A zero result is itself an important finding. The method fails because the injected fraud patterns don't include a "night + travel" combination. ACCOUNT_TAKEOVER fraud happens during business hours (10am–10pm by design), and GEO_IMPOSSIBLE fraud happens at daytime hours (8am–8pm) to simulate realistic travel patterns.

**What this teaches:** Feature combinations only work when the attack patterns actually produce those feature combinations. This is the core lesson of adversarial ML: attackers adapt to avoid known signatures. Night + foreign city is a valid signal for certain fraud typologies (card-not-present identity theft, international skimming) but not for the five patterns modelled here.

In a real fraud detection environment, this method would be validated against historical fraud labels before deployment. The zero performance here is not a bug — it's a calibration signal telling us where the rule is misaligned with actual attack vectors.

---

### Method 5: Category Anomaly Score

**Principle:** Compute each cardholder's category frequency distribution. Flag transactions in categories where the card has spent less than 5% of its historical transactions, combined with an above-baseline amount (`amount_z > 1`).

**Performance:**
- Precision: 0.010
- Recall: 0.159
- F1: 0.019

**Analysis:** Similar to geo-detection, the recall (15.9%) is meaningful — it's catching ACCOUNT_TAKEOVER fraud that slips through the other methods. But the precision is very low because many legitimate transactions also hit low-frequency categories (a grocery shopper visiting a healthcare provider, for example).

The root issue: 5% is too permissive a threshold for a dataset where cardholders visit all 10 categories over 6 months. With more historical data per card, frequency distributions would sharpen and rare-category flags would become more precise.

---

### Composite Risk Score

**Weighted ensemble:**

| Method | Weight | Rationale |
|--------|--------|-----------|
| Amount Z-Score | 30% | High precision, direct financial signal |
| Velocity | 25% | Near-perfect precision, highest F1 |
| Geo-Impossible | 25% | High recall, catches unique fraud type |
| Night + Foreign | 10% | Situational signal, low weight pending validation |
| Category Anomaly | 10% | Supporting signal for ATO detection |

**Alert tiers:**

| Tier | Score Range | Transactions | Confirmed Fraud | Precision |
|------|-------------|-------------|----------------|-----------|
| BLOCK | 75–100 | 0 | 0 | — |
| REVIEW | 55–75 | 12 | 12 | **100%** |
| MONITOR | 30–55 | 1,933 | 361 | 18.7% |
| CLEAR | 0–30 | 746,120 | 1,284 | 0.17% |

**Analysis of composite results:**

The REVIEW tier achieves 100% precision (12 for 12) — every transaction the system escalates to human review is confirmed fraud. This is the correct design for a fraud system: the highest-confidence alerts must be right, because incorrect blocks destroy customer trust.

The MONITOR tier at 18.7% precision is worth attention. Of 1,933 flagged transactions, 361 are fraud — that's a pool a fraud analyst team could investigate with a manageable false-positive burden. In a real platform, these would be queued for model-assisted review rather than immediate action.

The 1,284 confirmed fraud transactions landing in CLEAR represent the detection gap. Improving recall without sacrificing the REVIEW tier's precision is the core engineering challenge for v2.

---

## Findings & Analysis

### Finding 1: Velocity Is the Most Actionable Signal

The velocity rule (Method 2) is the clear winner for operational use: 98.7% precision, 46.6% recall, F1 of 0.633. No other method comes close on F1. For a fraud operations team, this translates directly to: **block any card making >5 transactions in 30 minutes with near-certainty of it being correct**.

Velocity abuse (668 transactions) and card testing (648 transactions) together account for **79.4% of all fraud** in this dataset — and velocity detection catches most of them. This means a single well-tuned rule handles the majority of fraud volume.

### Finding 2: High Precision and High Recall Are Mutually Exclusive at the Individual Method Level

The precision-recall trade-off is stark across the five methods:

| Method | Precision | Recall | Position |
|--------|-----------|--------|----------|
| Amount Z-Score | 0.758 | 0.142 | High precision, low recall |
| Velocity | 0.987 | 0.466 | High precision, medium recall |
| Geo-Impossible | 0.013 | 0.385 | Low precision, medium-high recall |
| Category Anomaly | 0.010 | 0.159 | Low precision, low recall |

This is exactly why ensemble methods exist. Methods 3 and 4 individually look like broken detectors by precision, but they're finding fraud that Methods 1 and 2 miss. The composite scorer's job is to combine these orthogonal signals intelligently — requiring multiple weak signals to fire together before escalating to REVIEW.

### Finding 3: The Composite Score Is Too Conservative

The 100% REVIEW precision is impressive but comes at a cost: only 12 of 1,657 fraud transactions reach REVIEW tier. The system is extremely conservative, preferring to miss fraud over generating false positives.

For a fraud platform, this is a calibration decision driven by business context:
- **High false positive tolerance** (e.g., small fintech, risk-tolerant): Lower REVIEW threshold to 45, accept more FP for higher recall
- **Low false positive tolerance** (e.g., premium credit card, reputation-sensitive): Keep 55+ threshold, supplement with ML-based second-pass on MONITOR queue
- **Real-time blocking** (e.g., instant payment rails): Trust only BLOCK tier (75+), use REVIEW for delayed posting

### Finding 4: Amount Spikes Are Underdetected Due to Rolling Window Design

AMOUNT_SPIKE detection (Method 1) catches only 14.2% recall despite being a conceptually simple pattern. The reason: the rolling window of 720 transactions is too large. For a cardholder making 2–3 transactions per day, 720 transactions represents 6–12 months of history — far more than the 30-day window implied by the variable name `rolling_avg_30d`.

A 30-day window for a mid-tier cardholder averaging 2.5 txns/day is actually ~75 transactions, not 720. This miscalibration means the rolling average is computed over a very long baseline, making it more stable and harder to deviate from — which suppresses true amount spikes.

This is a real-world data engineering lesson: **naming a variable "30-day rolling" doesn't make it 30 days unless you verify the window aligns with actual transaction frequency per segment**.

### Finding 5: Low-Income Cardholders Have the Highest Fraud Rate

At 0.32% fraud prevalence, low-income cardholders are targeted at 2.3× the rate of premium cardholders (0.14%). This pattern appears in real fraud data and has operational implications: risk scoring models that ignore income tier may under-flag low-income accounts while wasting resources on premium accounts that attackers find less attractive (stronger authentication, more fraud monitoring in place).

### Finding 6: Retail and Online Shopping Are the Fraud Attack Surface

481 Retail + 430 Online Shopping = 55.3% of all fraud transactions concentrated in just 2 of 10 merchant categories. This is the distribution we'd expect in real fraud data: low-friction, high-volume merchant categories that are easy to exploit for card testing and velocity abuse.

Implication: merchant-category risk weights should be incorporated into the composite score in v2 — transactions at online/retail merchants carry higher inherent risk than the same amount at a Utilities or Healthcare merchant.

---

## Business Impact

Based on an estimated average fraud transaction loss of $285 (industry benchmark for card-not-present fraud):

| Metric | Value |
|--------|-------|
| Total transactions monitored | 748,065 |
| Total volume analysed | $95,502,411 |
| Fraud cases escalated (REVIEW+BLOCK) | 12 |
| False positives generated | 0 |
| Estimated fraud losses prevented (REVIEW) | $3,420 |
| Confirmed fraud in MONITOR queue | 361 |
| Estimated value in MONITOR queue | $102,885 |
| Confirmed fraud missed (CLEAR) | 1,284 |
| Estimated missed fraud value | $365,940 |

**Key insight:** The $365,940 in missed fraud represents the true cost of the system's conservatism. Closing this gap — even partially — through ML-based scoring on the MONITOR queue or better-calibrated rolling windows would represent significant loss prevention with zero increase in operational workload for the fraud team.

**False positive cost:** 0 false positives in the REVIEW+BLOCK tiers. This means zero incorrect card blocks, zero customer service calls from blocked legitimate transactions, zero chargeback disputes — the hidden operational cost that most fraud systems ignore.

---

## Key Visualizations

All charts are saved to `outputs/`:

**`fraud_landscape.png`** — Side-by-side overview: fraud type distribution, alert tier breakdown, fraud rate by income tier, and category-level fraud volume. The entry-point dashboard for understanding the dataset at a glance.

**`detection_heatmap.png`** — A method × fraud-type correlation heatmap showing which detection methods are catching which attack vectors. Reveals the complementary nature of the ensemble: velocity catches what z-score misses; geo catches what velocity misses.

**`risk_score_distribution.png`** — Distribution of risk scores split by fraud vs. legitimate transactions, with alert tier thresholds overlaid. Shows the separation (or lack thereof) between the two distributions and the operating point of each tier.

**`geo_fraud_map.png`** — City-level transaction volume vs. fraud incidence, plotted geographically across the 10 cities. Highlights geographic concentrations of fraud activity.

**`model_performance.png`** — Precision-Recall bar chart for all five methods plus the composite, with F1 scores annotated. The clearest visual summary of relative method performance.

**`kpi_dashboard.png`** — Six-panel daily KPI dashboard: transaction volume, dollar volume, fraud rate over time, unique active cards, night transaction share, and city mix. Shows the platform operating pattern over 6 months.

---

## How to Run

**Prerequisites:**
```bash
pip install pandas numpy matplotlib seaborn
```

**Full pipeline:**
```bash
cd finshield
python src/main.py
```

**Individual steps:**
```bash
# Generate data only
python src/generate_data.py

# Run ETL only (after data generation)
python src/pipeline.py

# Run detection only (after ETL)
python src/anomaly_detection.py

# Generate charts only (after detection)
python src/visualizations.py
```

**Expected runtime:** ~3–5 minutes on a standard laptop (velocity detection is O(n²) per card — the bottleneck).

**Outputs:**
```
data/
  transactions_raw.csv        # 748,065 raw transactions
  cardholders.csv             # 2,000 cardholder profiles
  finshield_warehouse.db      # SQLite warehouse (6 tables)

outputs/
  fraud_landscape.png
  detection_heatmap.png
  risk_score_distribution.png
  geo_fraud_map.png
  model_performance.png
  kpi_dashboard.png
```

---

## What I Would Build Next

### v2: Machine Learning Layer

The rule-based + statistical approach establishes ground truth and sets a performance baseline. The natural next step is a supervised ML layer trained on the `detection_results` table:

1. **Features ready:** `amount_z`, `velocity_30m`, `is_night`, `is_home_city`, `rolling_avg_30d`, `rolling_std_30d`, income_tier, category — all already computed and warehoused
2. **Labels ready:** `is_fraud` ground truth with `fraud_type` for multi-class capability
3. **Model candidates:** XGBoost (handles class imbalance well with `scale_pos_weight`), Isolation Forest (unsupervised, no label dependency), LightGBM with focal loss for extreme imbalance

Expected improvement: Recall from 46.6% → 70–80% while maintaining REVIEW precision above 90%.

### v2: Real-Time Stream Processing

Replace the batch CSV extract with a Kinesis Data Streams ingestion layer:
- Lambda function consumes Kinesis shard → calls detection API → writes alert to DynamoDB
- API Gateway exposes `/score` endpoint for real-time scoring per transaction
- CloudWatch alarms on BLOCK-tier transactions with SNS → PagerDuty escalation

This is the architecture that a production payment system would run, and the SQLite warehouse here maps cleanly to Redshift (analytics) + DynamoDB (operational alerts).

### v3: Feedback Loop

Production fraud detection degrades without ground truth feedback. The feedback loop closes when:
- Fraud analysts label REVIEW queue transactions (confirmed fraud / false positive)
- Those labels are written back to the warehouse weekly
- Detection thresholds auto-calibrate based on recent precision/recall on the labelled set
- Model retraining triggers when precision drops below 85% on any method

This is the difference between a static rule engine and a learning fraud system.

---

## About This Project

This project was built to demonstrate the full data engineering + analytics lifecycle in a high-stakes financial domain — from data modelling and ETL pipeline design through feature engineering, statistical detection, and business impact quantification.

The architecture mirrors real fraud detection platforms at scale: streaming ingestion, per-cardholder behavioural baselines, ensemble risk scoring, and tiered alert queues. Every design decision maps to a production trade-off between precision (customer trust), recall (loss prevention), and operational cost (analyst workload).

**Connection to MBA background:** The business impact section intentionally frames detection performance in financial terms — prevented losses, false positive operational costs, and the marginal value of closing the detection gap. A fraud detection system that doesn't quantify its own ROI is an engineering project, not a business asset. The financial framing is the difference.

---

*FinShield | Aparajita Mondal | 2024*
