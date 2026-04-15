"""
Synthetic transaction data generator for FinCrime Transaction Monitor.

Design decisions:
- Fraud prevalence ~2%: realistic for real-world card fraud (industry benchmark
  is 1-3%). Higher rates would make the ML problem artificially easy.
- User population: 5,000 users with distinct behavioral profiles. Enables
  user-level feature engineering (velocity, historical averages).
- Time window: 90 days. Enough history for rolling features without
  unmanageable volume.
- Fraud patterns injected (not random): mimics real typologies like
  card-testing, account takeover, and high-value anomalies.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ---------- Config ----------
N_USERS = 5_000
N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.02
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 3, 31)

COUNTRIES = ['CO', 'US', 'MX', 'BR', 'ES', 'GB', 'AR', 'CL', 'PE', 'NG', 'RU', 'CN']
HIGH_RISK_COUNTRIES = ['NG', 'RU', 'CN']  # simplified proxy for FATF grey/high-risk
MERCHANT_CATEGORIES = [
    'grocery', 'restaurant', 'gas_station', 'online_retail',
    'electronics', 'travel', 'gambling', 'crypto_exchange',
    'money_transfer', 'atm_withdrawal'
]
HIGH_RISK_MCC = ['gambling', 'crypto_exchange', 'money_transfer']
CHANNELS = ['card_present', 'card_not_present', 'online', 'mobile_app']


def generate_users(n_users: int) -> pd.DataFrame:
    """Create a user base with behavioral baselines."""
    users = []
    for i in range(n_users):
        users.append({
            'user_id': f'U{i:05d}',
            'home_country': random.choices(
                COUNTRIES, weights=[40, 15, 10, 8, 5, 5, 5, 4, 3, 2, 2, 1]
            )[0],
            'avg_transaction_amount': np.random.lognormal(mean=3.5, sigma=0.8),
            'account_age_days': random.randint(30, 1500),
        })
    return pd.DataFrame(users)


def generate_legitimate_tx(user: pd.Series) -> dict:
    """Generate a transaction that matches the user's normal behavior."""
    amount = max(1, np.random.lognormal(
        mean=np.log(user['avg_transaction_amount']), sigma=0.5
    ))
    timestamp = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    return {
        'user_id': user['user_id'],
        'timestamp': timestamp,
        'amount_usd': round(amount, 2),
        'country': user['home_country'] if random.random() < 0.92 else random.choice(COUNTRIES),
        'merchant_category': random.choice(MERCHANT_CATEGORIES),
        'channel': random.choice(CHANNELS),
        'is_fraud': 0,
    }


def generate_fraud_tx(user: pd.Series, pattern: str) -> dict:
    """
    Generate a fraudulent transaction following a known typology.

    Patterns:
    - 'high_value_anomaly': amount 10-50x user's baseline
    - 'geo_anomaly': high-risk country, unusual for this user
    - 'velocity_burst': will be chained in post-processing
    - 'high_risk_mcc': gambling / crypto / money transfer
    """
    base = generate_legitimate_tx(user)
    base['is_fraud'] = 1

    if pattern == 'high_value_anomaly':
        base['amount_usd'] = round(user['avg_transaction_amount'] * random.uniform(10, 50), 2)
    elif pattern == 'geo_anomaly':
        base['country'] = random.choice(HIGH_RISK_COUNTRIES)
        base['channel'] = 'card_not_present'
    elif pattern == 'high_risk_mcc':
        base['merchant_category'] = random.choice(HIGH_RISK_MCC)
        base['amount_usd'] = round(base['amount_usd'] * random.uniform(2, 8), 2)
    elif pattern == 'velocity_burst':
        base['amount_usd'] = round(random.uniform(5, 50), 2)  # card testing = small amounts

    return base


def generate_transactions() -> pd.DataFrame:
    users_df = generate_users(N_USERS)
    n_fraud = int(N_TRANSACTIONS * FRAUD_RATE)
    n_legit = N_TRANSACTIONS - n_fraud

    print(f"Generating {n_legit:,} legitimate and {n_fraud:,} fraudulent transactions...")

    transactions = []

    # Legitimate
    for _ in range(n_legit):
        user = users_df.sample(1).iloc[0]
        transactions.append(generate_legitimate_tx(user))

    # Fraud with patterns
    patterns = ['high_value_anomaly', 'geo_anomaly', 'high_risk_mcc', 'velocity_burst']
    for _ in range(n_fraud):
        user = users_df.sample(1).iloc[0]
        pattern = random.choice(patterns)
        transactions.append(generate_fraud_tx(user, pattern))

    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['transaction_id'] = [f'T{i:07d}' for i in range(len(df))]

    # Reorder columns
    cols = ['transaction_id', 'user_id', 'timestamp', 'amount_usd',
            'country', 'merchant_category', 'channel', 'is_fraud']
    return df[cols], users_df


if __name__ == '__main__':
    tx_df, users_df = generate_transactions()
    tx_df.to_csv('data/transactions.csv', index=False)
    users_df.to_csv('data/users.csv', index=False)
    print(f"\n✅ Saved {len(tx_df):,} transactions to data/transactions.csv")
    print(f"✅ Saved {len(users_df):,} users to data/users.csv")
    print(f"\nFraud rate: {tx_df['is_fraud'].mean():.2%}")
    print(f"\nSample:\n{tx_df.head()}")