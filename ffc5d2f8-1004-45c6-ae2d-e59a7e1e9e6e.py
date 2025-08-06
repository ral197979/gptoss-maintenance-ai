
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import schedule, time, threading, smtplib, configparser
from sklearn.ensemble import RandomForestClassifier
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import plotly.graph_objs as go

# Load config
config = configparser.ConfigParser()
config.read("config.ini")

# CONFIG VALUES
API_KEY = config.get("ALPACA", "API_KEY", fallback="")
API_SECRET = config.get("ALPACA", "API_SECRET", fallback="")
SMTP_EMAIL = config.get("EMAIL", "SMTP_EMAIL", fallback="")
SMTP_PASSWORD = config.get("EMAIL", "SMTP_PASSWORD", fallback="")
RECIPIENT_EMAIL = config.get("EMAIL", "RECIPIENT_EMAIL", fallback="")
SLACK_URL = config.get("WEBHOOKS", "SLACK_URL", fallback="")
DISCORD_URL = config.get("WEBHOOKS", "DISCORD_URL", fallback="")
TELEGRAM_TOKEN = config.get("WEBHOOKS", "TELEGRAM_TOKEN", fallback="")
TELEGRAM_CHAT_ID = config.get("WEBHOOKS", "TELEGRAM_CHAT_ID", fallback="")

# USER TIERS
USER_TIERS = {
    "admin": {"pass": "adminpass", "tier": "Admin"},
    "pro": {"pass": "propass", "tier": "Pro"},
    "free": {"pass": "1234", "tier": "Free"},
}

st.set_page_config(page_title="Alpaca AI Bot", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_tier = "Free"

def login():
    st.title("🔐 Login to Alpaca Bot")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USER_TIERS and USER_TIERS[user]["pass"] == pw:
            st.session_state.logged_in = True
            st.session_state.user_tier = USER_TIERS[user]["tier"]
        else:
            st.error("Invalid login.")

if not st.session_state.logged_in:
    login()
    st.stop()

st.title("🚀 Alpaca AI Trading Bot")
st.markdown(f"**Plan Tier:** `{st.session_state.user_tier}`")

col1, col2 = st.columns(2)
with col1:
    stock = st.text_input("Stock Symbol", value="AAPL")
    strategy = st.selectbox("Strategy", ["RandomForest", "MarkovChain"])
with col2:
    days_back = st.slider("Backtest Days", 30, 365, 90)
    show_chart = st.checkbox("📊 Show Chart", True)
    interval = st.selectbox("Auto-Run", ["None", "5 min", "Hourly", "Daily"])

@st.cache_data(show_spinner=False)
def get_data(symbol, days):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    return yf.download(symbol, start=start)

df = get_data(stock, days_back)

def run_random_forest(df):
    df["Return"] = df["Close"].pct_change()
    df["Direction"] = (df["Return"] > 0).astype(int)
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df.dropna(inplace=True)
    X = df[["Lag1", "Lag2"]]
    y = df["Direction"]
    model = RandomForestClassifier()
    model.fit(X, y)
    df["Prediction"] = model.predict(X)
    df["BuySignal"] = df["Prediction"].shift(1)
    return df

def run_markov_chain(df):
    df["Signal"] = (df["Close"] > df["Close"].shift(1)).astype(int)
    df.dropna(inplace=True)
    mat = np.zeros((2, 2))
    for i in range(1, len(df)):
        mat[df["Signal"].iloc[i-1], df["Signal"].iloc[i]] += 1
    probs = mat / mat.sum(axis=1, keepdims=True)
    df["BuySignal"] = probs[1][1] > 0.6
    return df

def place_trade(symbol, qty=10):
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET
    }
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": "buy",
        "type": "market",
        "time_in_force": "gtc"
    }
    return requests.post("https://paper-api.alpaca.markets/v2/orders", json=payload, headers=headers).json()

def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, RECIPIENT_EMAIL, msg.as_string())

def send_webhooks(text):
    if SLACK_URL:
        requests.post(SLACK_URL, json={"text": text})
    if DISCORD_URL:
        requests.post(DISCORD_URL, json={"content": text})
    if TELEGRAM_TOKEN:
        requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", params={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        })

def execute_strategy():
    d = run_random_forest(df.copy()) if strategy == "RandomForest" else run_markov_chain(df.copy())
    if show_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close"))
        buys = d[d["BuySignal"] == 1]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", marker=dict(color="green", size=8), name="Buy"))
        st.plotly_chart(fig, use_container_width=True)
    return d

def gpt_summary(symbol):
    try:
        import openai
        key = config.get("LLM", "OPENAI_API_KEY")
        openai.api_key = key
        txt = f"Summarize the recent market trend for {symbol} in 50 words."
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": txt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"

# Background scheduler
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = True
    threading.Thread(target=run_scheduler, daemon=True).start()

if interval == "5 min":
    schedule.every(5).minutes.do(execute_strategy)
elif interval == "Hourly":
    schedule.every().hour.do(execute_strategy)
elif interval == "Daily":
    schedule.every().day.at("10:00").do(execute_strategy)

if st.button("🚀 Run Now"):
    d = execute_strategy()
    st.dataframe(d.tail(5)[["Close", "BuySignal"]])
    summary = gpt_summary(stock)
    st.markdown("#### 📘 Market Summary")
    st.info(summary)

if st.button("📈 Place Trade"):
    if st.session_state.user_tier == "Free":
        st.warning("Upgrade to Pro or Admin to trade.")
    else:
        r = place_trade(stock)
        st.success("✅ Trade Placed")
        st.json(r)
        send_email("Alpaca Trade Executed", f"Trade Info:
{r}")
        send_webhooks(f"Alpaca Trade Executed: {stock}")

if st.button("🔓 Logout"):
    st.session_state.logged_in = False
    st.rerun()
