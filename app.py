import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from data_utils import get_intraday_features, create_intraday_target
from intraday_model import train_intraday_model, FEATURES

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intraday Circuit Probability",
    layout="wide"
)

st.title("📈 Intraday 5-Minute Circuit Probability Dashboard")

# ---------------- STOCK LIST ----------------
NSE_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS"
}

# ---------------- USER INPUT ----------------
st.subheader("🔍 Select Stock")

selected_stock = st.selectbox(
    "Choose from popular NSE stocks",
    list(NSE_STOCKS.keys())
)

custom_symbol = st.text_input(
    "Or enter custom NSE symbol (example: ADANIENT.NS)",
    ""
)

symbol = (
    custom_symbol.strip().upper()
    if custom_symbol.strip() != ""
    else NSE_STOCKS[selected_stock]
)

st.caption(f"Using symbol: **{symbol}**")

# ---------------- LIVE PRICE ----------------
try:
    ticker = yf.Ticker(symbol)
    live_data = ticker.history(period="1d", interval="1m")

    if live_data.empty:
        st.error("Live price data unavailable.")
        st.stop()

    live_price = live_data['Close'].iloc[-1]
    st.metric("💰 Live Price", f"₹{round(live_price, 2)}")
except Exception:
    st.error("Failed to fetch live price.")
    st.stop()

# ---------------- INTRADAY DATA ----------------
df = get_intraday_features(symbol)

if df.empty or len(df) < 50:
    st.warning("Not enough intraday data to run prediction.")
    st.stop()

df = create_intraday_target(df)

# ---------------- TRAIN MODEL ----------------
model = train_intraday_model(df)

# ---------------- PREDICTION ----------------
latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
probs = model.predict_proba(latest)[0]
classes = model.classes_

# Safe probability mapping
prob_map = {-1: 0.0, 0: 0.0, 1: 0.0}
for cls, p in zip(classes, probs):
    prob_map[int(cls)] = float(p)

# ---------------- DISPLAY ----------------
st.subheader("🔮 Probability of Circuit (By End of Day)")

col1, col2, col3 = st.columns(3)
col1.metric("📉 Lower Circuit", f"{prob_map[-1]*100:.2f}%")
col2.metric("⚖️ No Circuit", f"{prob_map[0]*100:.2f}%")
col3.metric("📈 Upper Circuit", f"{prob_map[1]*100:.2f}%")

st.caption(f"Model trained on classes: {list(model.classes_)}")

# ---------------- CHART ----------------
st.subheader("📊 Intraday 5-Minute Price Chart")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['Close'], linewidth=2)
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.grid(True)
st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ Educational purpose only. Not financial advice.")
