"""
Generates 1000-row sample data for FinShield:
  - transactions_sample.csv  (1000 transactions)
  - cardholders_sample.csv   (50 cardholders)

Run: python data/raw/generate_sample.py
"""

import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

# --- Config ---
NUM_CARDHOLDERS = 50
NUM_TRANSACTIONS = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 6, 29)

CITIES = {
    "New York":      (40.7128, -74.0060),
    "Los Angeles":   (34.0522, -118.2437),
    "Chicago":       (41.8781, -87.6298),
    "Houston":       (29.7604, -95.3698),
    "Phoenix":       (33.4484, -112.0740),
    "Philadelphia":  (39.9526, -75.1652),
    "San Antonio":   (29.4241, -98.4936),
    "Dallas":        (32.7767, -96.7970),
    "Austin":        (30.2672, -97.7431),
    "Seattle":       (47.6062, -122.3321),
}
CITY_NAMES = list(CITIES.keys())

INCOME_TIERS = {
    "low":     {"avg": 32,  "std": 18,  "txns_day": 1.8,  "share": 0.30},
    "mid":     {"avg": 68,  "std": 45,  "txns_day": 2.5,  "share": 0.40},
    "high":    {"avg": 145, "std": 120, "txns_day": 3.2,  "share": 0.22},
    "premium": {"avg": 380, "std": 280, "txns_day": 4.0,  "share": 0.08},
}

CATEGORIES = [
    "Retail", "Online Shopping", "Dining", "Travel",
    "Entertainment", "Healthcare", "Grocery", "Utilities",
    "Fuel", "Education",
]

FRAUD_TYPES = ["VELOCITY_ABUSE", "CARD_TESTING", "ACCOUNT_TAKEOVER", "AMOUNT_SPIKE", "GEO_IMPOSSIBLE"]


def pick_tier():
    r = random.random()
    cum = 0
    for tier, cfg in INCOME_TIERS.items():
        cum += cfg["share"]
        if r < cum:
            return tier
    return "mid"


def random_timestamp(start, end):
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── 1. Generate cardholders ────────────────────────────────────────────────────
cardholders = []
for i in range(1, NUM_CARDHOLDERS + 1):
    tier = pick_tier()
    cfg = INCOME_TIERS[tier]
    home_city = random.choice(CITY_NAMES)
    preferred_cats = random.sample(CATEGORIES, 3)
    cardholders.append({
        "card_id":          f"CARD_{i:04d}",
        "income_tier":      tier,
        "home_city":        home_city,
        "home_lat":         CITIES[home_city][0],
        "home_lon":         CITIES[home_city][1],
        "avg_transaction":  round(cfg["avg"], 2),
        "std_transaction":  round(cfg["std"], 2),
        "txns_per_day":     cfg["txns_day"],
        "preferred_cat_1":  preferred_cats[0],
        "preferred_cat_2":  preferred_cats[1],
        "preferred_cat_3":  preferred_cats[2],
        "active_hour_start": random.randint(6, 10),
        "active_hour_end":   random.randint(20, 23),
    })

# ── 2. Generate transactions ───────────────────────────────────────────────────
# Decide how many are fraud (~0.22%)
num_fraud = max(2, round(NUM_TRANSACTIONS * 0.0022 * 10))  # slightly elevated for sample visibility
fraud_indices = set(random.sample(range(NUM_TRANSACTIONS), num_fraud))

transactions = []
for i in range(NUM_TRANSACTIONS):
    txn_id = f"TXN_{i+1:06d}"
    card = random.choice(cardholders)
    tier_cfg = INCOME_TIERS[card["income_tier"]]

    ts = random_timestamp(START_DATE, END_DATE)
    hour = ts.hour
    is_night = 1 if hour < 6 or hour >= 22 else 0

    is_fraud = 1 if i in fraud_indices else 0
    fraud_type = ""

    if is_fraud:
        fraud_type = random.choice(FRAUD_TYPES)
        # Tweak transaction characteristics to reflect fraud pattern
        if fraud_type == "CARD_TESTING":
            amount = round(random.uniform(0.50, 2.50), 2)
            city = random.choice(CITY_NAMES)
        elif fraud_type == "AMOUNT_SPIKE":
            amount = round(tier_cfg["avg"] * random.uniform(8, 20), 2)
            city = card["home_city"]
        elif fraud_type == "GEO_IMPOSSIBLE":
            amount = round(abs(random.gauss(tier_cfg["avg"], tier_cfg["std"])), 2) or 1.0
            city = random.choice([c for c in CITY_NAMES if c != card["home_city"]])
        elif fraud_type == "ACCOUNT_TAKEOVER":
            amount = round(abs(random.gauss(tier_cfg["avg"] * 2, tier_cfg["std"])), 2) or 1.0
            city = random.choice([c for c in CITY_NAMES if c != card["home_city"]])
        else:  # VELOCITY_ABUSE
            amount = round(abs(random.gauss(tier_cfg["avg"], tier_cfg["std"])), 2) or 1.0
            city = card["home_city"]
        category = random.choice(CATEGORIES)
    else:
        amount = round(max(0.01, random.gauss(tier_cfg["avg"], tier_cfg["std"])), 2)
        # Prefer home city 85% of the time
        city = card["home_city"] if random.random() < 0.85 else random.choice(CITY_NAMES)
        # Prefer preferred categories 70% of the time
        if random.random() < 0.70:
            category = random.choice([
                card["preferred_cat_1"],
                card["preferred_cat_2"],
                card["preferred_cat_3"],
            ])
        else:
            category = random.choice(CATEGORIES)

    lat, lon = CITIES[city]
    merchant_id = f"MERCH_{random.randint(1, 200):04d}"
    merchant_name = f"{category.replace(' ', '_')}_Store_{random.randint(1, 50)}"
    is_home_city = 1 if city == card["home_city"] else 0

    transactions.append({
        "transaction_id":  txn_id,
        "card_id":         card["card_id"],
        "timestamp":       ts.strftime("%Y-%m-%d %H:%M:%S"),
        "amount":          amount,
        "merchant_id":     merchant_id,
        "merchant_name":   merchant_name,
        "category":        category,
        "city":            city,
        "lat":             lat,
        "lon":             lon,
        "is_home_city":    is_home_city,
        "is_night":        is_night,
        "income_tier":     card["income_tier"],
        "is_fraud":        is_fraud,
        "fraud_type":      fraud_type,
    })

# Sort by timestamp (realistic ordering)
transactions.sort(key=lambda x: x["timestamp"])

# ── 3. Write CSVs ──────────────────────────────────────────────────────────────
import os
out_dir = os.path.dirname(os.path.abspath(__file__))

txn_path = os.path.join(out_dir, "transactions_sample.csv")
with open(txn_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=transactions[0].keys())
    writer.writeheader()
    writer.writerows(transactions)
print(f"Written {len(transactions)} transactions → {txn_path}")

card_path = os.path.join(out_dir, "cardholders_sample.csv")
with open(card_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cardholders[0].keys())
    writer.writeheader()
    writer.writerows(cardholders)
print(f"Written {len(cardholders)} cardholders → {card_path}")
