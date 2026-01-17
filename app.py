import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from data_utils import get_intraday_features, create_intraday_target
from intraday_model import train_intraday_model, FEATURES

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Intraday Circuit Probability",
    layout="wide"
)

st.title("📈 Intraday 5-Minute Circuit Probability Dashboard")

# -----------------------------
# User Input
# -----------------------------
symbol = st.text_input(
    "Enter NSE Symbol",
    value="RELIANCE.NS"
).strip().upper()

# -----------------------------
# Live Price
# -----------------------------
try:
    ticker = yf.Ticker(symbol)
    live_hist = ticker.history(period="1d", interval="1m")

    if live_hist.empty:
        st.error("Live price data not available.")
        st.stop()

    live_price = live_hist['Close'].iloc[-1]
    st.metric("💰 Live Price", f"₹{round(live_price, 2)}")
except Exception as e:
    st.error("Failed to fetch live price.")
    st.stop()

# -----------------------------
# Intraday Feature Data
# -----------------------------
df = get_intraday_features(symbol)

if df.empty or len(df) < 50:
    st.warning("Not enough intraday data available to build model.")
    st.stop()

# -----------------------------
# Target Creation
# -----------------------------
df = create_intraday_target(df)

# -----------------------------
# Train Intraday Model
# -----------------------------
model = train_intraday_model(df)

# -----------------------------
# Prediction
# -----------------------------
latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
probs = model.predict_proba(latest)[0]
classes = model.classes_

# Safe probability mapping
prob_map = {
    -1: 0.0,  # Lower circuit
     0: 0.0,  # No circuit
     1: 0.0   # Upper circuit
}

for cls, p in zip(classes, probs):
    prob_map[int(cls)] = float(p)

# -----------------------------
# Display Probabilities
# -----------------------------
st.subheader("🔮 Probability of Circuit (By End of Day)")

col1, col2, col3 = st.columns(3)

col1.metric("📉 Lower Circuit", f"{prob_map[-1] * 100:.2f}%")
col2.metric("⚖️ No Circuit", f"{prob_map[0] * 100:.2f}%")
col3.metric("📈 Upper Circuit", f"{prob_map[1] * 100:.2f}%")

st.caption(f"Model trained on classes: {list(model.classes_)}")

# -----------------------------
# Intraday Price Chart
# -----------------------------
st.subheader("📊 Intraday 5-Minute Price Chart")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['Close'], linewidth=2)
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚠️ Educational use only. Not financial advice.")
