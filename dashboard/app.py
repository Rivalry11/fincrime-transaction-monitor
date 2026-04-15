"""
FinCrime Transaction Monitor — Streamlit Dashboard

Single-page operational dashboard for compliance analysts:
- Top-line metrics (volume, alerts, fraud rate)
- Risk score distribution
- Alert queue with filtering
- Per-user risk drill-down
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------- Page config --------------
st.set_page_config(
    page_title="FinCrime Transaction Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------- Load assets --------------
FEATURES = [
    'amount_vs_user_avg', 'amount_zscore', 'log_amount',
    'tx_count_1h', 'tx_count_24h', 'amount_sum_24h',
    'is_foreign_country', 'is_high_risk_country',
    'is_high_risk_mcc', 'is_night_tx', 'is_cnp',
]

@st.cache_data
def load_data():
    df = pd.read_csv('data/transactions_featured.csv', parse_dates=['timestamp'])
    return df

@st.cache_resource
def load_model():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return model, threshold

df = load_data()
model, default_threshold = load_model()

# Score all transactions (cached)
@st.cache_data
def score_transactions(df):
    X = df[FEATURES].fillna(0)
    df = df.copy()
    df['risk_score'] = model.predict_proba(X)[:, 1]
    return df

df = score_transactions(df)

# -------------- Sidebar: filters --------------
st.sidebar.title("🛡️ FinCrime Monitor")
st.sidebar.markdown("**Compliance analyst console**")
st.sidebar.divider()

st.sidebar.subheader("Risk threshold")
threshold = st.sidebar.slider(
    "Alert threshold (risk score ≥)",
    min_value=0.0, max_value=1.0,
    value=float(default_threshold), step=0.01,
    help="Transactions scoring above this threshold are flagged as alerts. "
         "Lower = more recall, more false positives."
)

st.sidebar.subheader("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
)

countries = st.sidebar.multiselect(
    "Country",
    options=sorted(df['country'].unique()),
    default=None,
)

mcc_filter = st.sidebar.multiselect(
    "Merchant category",
    options=sorted(df['merchant_category'].unique()),
    default=None,
)

# Apply filters
filtered = df.copy()
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered['timestamp'] >= start) & (filtered['timestamp'] < end)]
if countries:
    filtered = filtered[filtered['country'].isin(countries)]
if mcc_filter:
    filtered = filtered[filtered['merchant_category'].isin(mcc_filter)]

filtered['is_alert'] = (filtered['risk_score'] >= threshold).astype(int)

# -------------- Header --------------
st.title("Transaction Monitoring Dashboard")
st.caption("Real-time fraud and AML signal monitoring across the transaction stream.")

# -------------- KPI row --------------
total_tx = len(filtered)
n_alerts = int(filtered['is_alert'].sum())
alert_rate = (n_alerts / total_tx * 100) if total_tx > 0 else 0
total_volume = filtered['amount_usd'].sum()
flagged_volume = filtered[filtered['is_alert'] == 1]['amount_usd'].sum()

# True fraud caught (only available in dev with labels — in prod you'd track this via SAR confirmations)
true_fraud = int(filtered['is_fraud'].sum())
caught = int(filtered[(filtered['is_alert'] == 1) & (filtered['is_fraud'] == 1)].shape[0])
recall = (caught / true_fraud * 100) if true_fraud > 0 else 0
precision = (caught / n_alerts * 100) if n_alerts > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total transactions", f"{total_tx:,}")
c2.metric("Alerts generated", f"{n_alerts:,}", f"{alert_rate:.2f}% of volume")
c3.metric("Volume flagged", f"${flagged_volume:,.0f}", f"of ${total_volume:,.0f}")
c4.metric("Recall (model eval)", f"{recall:.1f}%", help="% of true fraud caught at this threshold")
c5.metric("Precision (model eval)", f"{precision:.1f}%", help="% of alerts that are true fraud")

st.divider()

# -------------- Two-column: distribution + alerts by category --------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Risk score distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered[filtered['is_fraud'] == 0]['risk_score'],
        name='Legitimate', marker_color='#4C72B0', opacity=0.7, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=filtered[filtered['is_fraud'] == 1]['risk_score'],
        name='Fraud (ground truth)', marker_color='#C44E52', opacity=0.85, nbinsx=50,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold {threshold:.2f}", annotation_position="top right")
    fig.update_layout(
        barmode='overlay', height=400,
        xaxis_title="Risk score", yaxis_title="Transactions",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Alerts by merchant category")
    alerts_by_mcc = (filtered[filtered['is_alert'] == 1]
                     .groupby('merchant_category').size()
                     .sort_values(ascending=True)
                     .reset_index(name='alerts'))
    if len(alerts_by_mcc) > 0:
        fig = px.bar(alerts_by_mcc, x='alerts', y='merchant_category',
                     orientation='h', color='alerts',
                     color_continuous_scale='Reds')
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False,
                          xaxis_title="Alert count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts in current filter")

st.divider()

# -------------- Alert queue --------------
st.subheader(f"🚨 Alert queue ({n_alerts:,} transactions)")
st.caption("Transactions flagged by the model, sorted by risk score (highest first).")

alerts = (filtered[filtered['is_alert'] == 1]
          .sort_values('risk_score', ascending=False)
          .head(100))

if len(alerts) > 0:
    display_cols = ['transaction_id', 'timestamp', 'user_id', 'amount_usd',
                    'country', 'merchant_category', 'channel', 'risk_score', 'is_fraud']
    st.dataframe(
        alerts[display_cols].rename(columns={'is_fraud': 'true_label'}),
        use_container_width=True, hide_index=True,
        column_config={
            'risk_score': st.column_config.ProgressColumn(
                'Risk score', min_value=0, max_value=1, format="%.3f"),
            'amount_usd': st.column_config.NumberColumn('Amount (USD)', format="$%.2f"),
            'true_label': st.column_config.NumberColumn(
                'True label', help="1 = confirmed fraud (only in dev with labels)"),
        }
    )
else:
    st.info("No alerts generated at this threshold.")

st.divider()

# -------------- User drill-down --------------
st.subheader("🔍 User investigation")

user_options = sorted(filtered['user_id'].unique())
selected_user = st.selectbox(
    "Select user to investigate",
    options=user_options,
    index=0 if user_options else None,
    placeholder="Choose a user_id...",
)

if selected_user:
    user_tx = filtered[filtered['user_id'] == selected_user].sort_values('timestamp')

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Total tx", len(user_tx))
    u2.metric("Total spent", f"${user_tx['amount_usd'].sum():,.2f}")
    u3.metric("Avg risk score", f"{user_tx['risk_score'].mean():.3f}")
    u4.metric("Alerts", int(user_tx['is_alert'].sum()))

    fig = px.scatter(
        user_tx, x='timestamp', y='amount_usd',
        color='risk_score', size='amount_usd',
        color_continuous_scale='RdYlGn_r',
        hover_data=['country', 'merchant_category', 'channel', 'risk_score'],
        title=f"Transaction history — {selected_user}",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# -------------- Footer --------------
st.divider()
st.caption(
    "Built with synthetic data for portfolio purposes. "
    "Model: XGBoost trained on 11 engineered features mapped to AML typologies "
    "(high-value anomaly, velocity bursts, geo anomaly, high-risk MCC). "
    f"Default operating threshold: {default_threshold:.3f}"
)