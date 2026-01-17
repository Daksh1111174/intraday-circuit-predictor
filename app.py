import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from data_utils import get_intraday_features, create_intraday_target
from intraday_model import train_intraday_model, FEATURES

st.set_page_config(layout="wide")
st.title("📈 Intraday 5-Minute Circuit Probability Dashboard")

symbol = st.text_input("Enter NSE Symbol", "RELIANCE.NS")

ticker = yf.Ticker(symbol)
live_price = ticker.history(period="1d")['Close'].iloc[-1]

st.metric("💰 Live Price", f"₹{round(live_price, 2)}")

df = get_intraday_features(symbol)
df = create_intraday_target(df)

model = train_intraday_model(df)

latest = df[FEATURES].iloc[-1].values.reshape(1, -1)
prob = model.predict_proba(latest)[0]

st.subheader("🔮 Probability of Circuit (By End of Day)")

col1, col2, col3 = st.columns(3)
col1.metric("📉 Lower Circuit", f"{prob[0]*100:.2f}%")
col2.metric("⚖️ No Circuit", f"{prob[1]*100:.2f}%")
col3.metric("📈 Upper Circuit", f"{prob[2]*100:.2f}%")

st.subheader("📊 Intraday 5-Minute Price Chart")
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'])
ax.set_xlabel("Time")
ax.set_ylabel("Price")
st.pyplot(fig)
