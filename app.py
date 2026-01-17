import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import get_intraday_features, create_intraday_target
from intraday_model import train_intraday_model, FEATURES

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intraday Circuit Probability", layout="wide")
st.title("📈 Intraday 5-Minute Circuit Probability Dashboard")

# ---------------- STOCK LIST ----------------
NSE_STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "Vedanta": "VEDL.NS",
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "NBCC": "NBCC.NS",
    "Suzlon": "SUZLON.NS",
    "JP Power": "JPPOWER.NS",
    "IFCI": "IFCI.NS",
    "South Indian Bank": "SOUTHBANK.NS",
    "Adani Green": "ADANIGREEN.NS",
    "National Aluminium": "NATIONALUM.NS",
    "Hindustan Copper": "HINDCOPPER.NS",
    "Silver ETF": "SILVERBEES.NS"
}

# ---------------- STOCK SELECT ----------------
st.subheader("🔍 Select Stock")

selected = st.selectbox("Choose stock", list(NSE_STOCKS.keys()))
custom = st.text_input("Or enter custom NSE symbol (e.g. ADANIENT.NS)", "")

symbol = custom.strip().upper() if custom.strip() else NSE_STOCKS[selected]
st.caption(f"Using symbol: **{symbol}**")

# ---------------- LIVE PRICE ----------------
ticker = yf.Ticker(symbol)
live = ticker.history(period="1d", interval="1m")

if live.empty:
    st.error("Live price not available.")
    st.stop()

price = live['Close'].iloc[-1]
st.metric("💰 Live Price", f"₹{round(price,2)}")

# ---------------- SINGLE STOCK PREDICTION ----------------
df = get_intraday_features(symbol)

if df.empty or len(df) < 50:
    st.warning("Not enough intraday data.")
    st.stop()

df = create_intraday_target(df)
model = train_intraday_model(df)

latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
probs = model.predict_proba(latest)[0]
classes = model.classes_

prob_map = {-1: 0.0, 0: 0.0, 1: 0.0}
for c, p in zip(classes, probs):
    prob_map[int(c)] = float(p)

st.subheader("🔮 Circuit Probability (Today)")

c1, c2, c3 = st.columns(3)
c1.metric("📉 Lower", f"{prob_map[-1]*100:.2f}%")
c2.metric("⚖️ No Circuit", f"{prob_map[0]*100:.2f}%")
c3.metric("📈 Upper", f"{prob_map[1]*100:.2f}%")

# ---------------- PRICE CHART ----------------
st.subheader("📊 Intraday Price (5-min)")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df.index, df['Close'])
ax.grid(True)
st.pyplot(fig)

# ================= TOP-10 SCANNER =================
st.markdown("---")
st.header("🚨 Top-10 Intraday Circuit Probability Scanner")

scanner_results = []

with st.spinner("Scanning stocks..."):
    for name, sym in NSE_STOCKS.items():
        try:
            d = get_intraday_features(sym)
            if d.empty or len(d) < 50:
                continue

            d = create_intraday_target(d)
            m = train_intraday_model(d)

            last = d[FEATURES].iloc[-1].values.reshape(1, -1)
            pr = m.predict_proba(last)[0]
            cl = m.classes_

            pm = {-1: 0.0, 0: 0.0, 1: 0.0}
            for c, p in zip(cl, pr):
                pm[int(c)] = float(p)

            scanner_results.append({
                "Stock": name,
                "Symbol": sym,
                "Upper %": round(pm[1]*100, 2),
                "Lower %": round(pm[-1]*100, 2),
                "Max Circuit %": round(max(pm[1], pm[-1])*100, 2)
            })
        except:
            continue

if scanner_results:
    scan_df = pd.DataFrame(scanner_results)
    scan_df.sort_values("Max Circuit %", ascending=False, inplace=True)
    st.dataframe(scan_df.head(10), use_container_width=True)
else:
    st.warning("No stocks with sufficient data.")

st.caption("⚠️ Educational use only. Not financial advice.")
