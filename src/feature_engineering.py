"""
Feature engineering for FinCrime Transaction Monitor.

Each feature is mapped to a known fraud typology:
- Amount features → high-value anomaly detection
- Velocity features → card testing, account takeover
- Geographic features → geo anomaly, high-risk jurisdiction
- Behavioral features → high-risk MCC, unusual timing/channel

Design principle: all features are computed using ONLY information available
BEFORE the transaction happens (no data leakage). Rolling windows exclude the
current transaction.
"""

import pandas as pd
import numpy as np

HIGH_RISK_COUNTRIES = ['NG', 'RU', 'CN']
HIGH_RISK_MCC = ['gambling', 'crypto_exchange', 'money_transfer']


def build_features(tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Build 10 risk features from raw transactions + user base.

    Returns a DataFrame with original columns + engineered features,
    sorted by timestamp (chronological order matters for rolling features).
    """
    df = tx.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    # Merge user baselines
    df = df.merge(users[['user_id', 'home_country', 'avg_transaction_amount']],
                  on='user_id', how='left')

    # -------------------------------------------------------------------
    # FAMILY 1: AMOUNT-BASED FEATURES (high-value anomaly typology)
    # -------------------------------------------------------------------

    # 1. Ratio of current amount to user's historical baseline.
    #    > 1 means the tx is larger than the user's typical amount.
    #    High-value fraud injects tx 10-50x the baseline → this should spike.
    df['amount_vs_user_avg'] = df['amount_usd'] / df['avg_transaction_amount']

    # 2. Z-score within the user's transaction history (expanding window,
    #    shifted by 1 to avoid leakage — the current tx doesn't inform its own stats).
    user_rolling = df.groupby('user_id')['amount_usd']
    df['user_mean_prior'] = user_rolling.transform(lambda x: x.shift(1).expanding().mean())
    df['user_std_prior']  = user_rolling.transform(lambda x: x.shift(1).expanding().std())
    df['amount_zscore'] = (df['amount_usd'] - df['user_mean_prior']) / df['user_std_prior']
    df['amount_zscore'] = df['amount_zscore'].fillna(0).replace([np.inf, -np.inf], 0)

    # 3. Log-transformed amount. Raw amounts are heavy-tailed; log stabilizes
    #    the scale so linear models and tree splits behave better.
    df['log_amount'] = np.log1p(df['amount_usd'])

    # -------------------------------------------------------------------
    # FAMILY 2: VELOCITY FEATURES (card testing, account takeover)
    # -------------------------------------------------------------------

    # 4. Transactions in the last hour by the same user. Card testing
    #    typology: attackers run many small tx in a short window to validate
    #    stolen card data. Uses rolling time window, excludes current tx.
    df = df.set_index('timestamp')
    df['tx_count_1h'] = (
        df.groupby('user_id')['transaction_id']
          .rolling('1h', closed='left').count()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    # 5. Transactions in the last 24h by the same user. Catches slower-paced
    #    account takeover where attacker spreads activity to evade 1h thresholds.
    df['tx_count_24h'] = (
        df.groupby('user_id')['transaction_id']
          .rolling('24h', closed='left').count()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    # 6. Total amount spent in last 24h (excluding current). Catches the
    #    cumulative drain pattern even when individual tx look normal.
    df['amount_sum_24h'] = (
        df.groupby('user_id')['amount_usd']
          .rolling('24h', closed='left').sum()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )
    df = df.reset_index()

    # -------------------------------------------------------------------
    # FAMILY 3: GEOGRAPHIC FEATURES (geo anomaly)
    # -------------------------------------------------------------------

    # 7. Transaction country differs from user's home country. Not a hard
    #    flag (travelers exist) but a useful weak signal.
    df['is_foreign_country'] = (df['country'] != df['home_country']).astype(int)

    # 8. Country is on the high-risk list. Simplified proxy for FATF
    #    grey/black list logic used in real compliance programs.
    df['is_high_risk_country'] = df['country'].isin(HIGH_RISK_COUNTRIES).astype(int)

    # -------------------------------------------------------------------
    # FAMILY 4: BEHAVIORAL / CATEGORICAL
    # -------------------------------------------------------------------

    # 9. Merchant category in high-risk list. Gambling, crypto, money
    #    transfer are over-represented in real fraud datasets and in AML
    #    typologies (layering, structuring).
    df['is_high_risk_mcc'] = df['merchant_category'].isin(HIGH_RISK_MCC).astype(int)

    # 10. Nighttime transaction (00:00-05:59). Fraud concentrates at hours
    #     when the legitimate user is likely asleep and slower to notice.
    df['hour'] = df['timestamp'].dt.hour
    df['is_night_tx'] = df['hour'].between(0, 5).astype(int)

    # Also one-hot for channel (card_not_present is the main fraud vector)
    df['is_cnp'] = (df['channel'] == 'card_not_present').astype(int)

    # Clean up intermediate columns we don't want as features
    df = df.drop(columns=['user_mean_prior', 'user_std_prior', 'hour',
                        'home_country', 'avg_transaction_amount'])

    return df

if __name__ == '__main__':
    tx = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
    users = pd.read_csv('data/users.csv')

    print(f"Building features for {len(tx):,} transactions...")
    features_df = build_features(tx, users)

    feature_cols = [
        'amount_vs_user_avg', 'amount_zscore', 'log_amount',
        'tx_count_1h', 'tx_count_24h', 'amount_sum_24h',
        'is_foreign_country', 'is_high_risk_country',
        'is_high_risk_mcc', 'is_night_tx', 'is_cnp',
    ]

    print(f"\n✅ Built {len(feature_cols)} features")
    print(f"\nFeature summary by class:")
    print(features_df.groupby('is_fraud')[feature_cols].mean().round(3).T)

    features_df.to_csv('data/transactions_featured.csv', index=False)
    print(f"\n✅ Saved enriched dataset to data/transactions_featured.csv")