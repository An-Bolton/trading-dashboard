# trading_dashboard.py – Streamlit med P&L-graf, porteføljeverdi og justerbar take profit/stop loss

import streamlit as st
import pandas as pd
import sqlite3
import os
import subprocess
import time
from datetime import datetime


DB_PATH = "orders.db"
ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


st.set_page_config(page_title="Smart Order Router Dashboard", layout="wide")
st.title("📊 Smart Order Router Dashboard")

# Automatisk oppdatering hvert 30. sekund
st_autorefresh(interval=30 * 1000, key="datarefresh")

# Justerbar TP/SL
tp_pct = st.sidebar.slider("Take Profit (%)", 1, 20, 5)
sl_pct = st.sidebar.slider("Stop Loss (%)", 1, 20, 5)

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM orders ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_open_positions():
    try:
        positions = api.list_positions()
        if not positions:
            return pd.DataFrame()
        rows = [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "take_profit_price": round(float(p.avg_entry_price) * (1 + tp_pct / 100), 2),
                "stop_loss_price": round(float(p.avg_entry_price) * (1 - sl_pct / 100), 2)
            }
            for p in positions
        ]
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Feil ved henting av posisjoner: {e}")
        return pd.DataFrame()

def get_closed_positions():
    try:
        activities = api.get_activities(activity_types="FILL")
        rows = []
        for a in activities:
            if a.side in ["sell", "buy"] and float(a.qty) > 0:
                rows.append({
                    "symbol": a.symbol,
                    "side": a.side,
                    "qty": float(a.qty),
                    "price": float(a.price),
                    "amount": float(a.price) * float(a.qty),
                    "timestamp": a.transaction_time.strftime("%Y-%m-%d %H:%M:%S")
                })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Feil ved henting av handler: {e}")
        return pd.DataFrame()

def export_tax_report(df):
    export_file = f"skatteklar_rapport_{datetime.today().date()}.xlsx"
    df.to_excel(export_file, index=False)
    st.success(f"Excel-rapport generert: {export_file}")
    with open(export_file, "rb") as file:
        st.download_button(
            label="📥 Last ned skatteklar rapport",
            data=file,
            file_name=export_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.sidebar.header("🛠 Kontrollpanel")

if "bot_running" not in st.session_state:
    st.session_state.bot_running = False

start = st.sidebar.button("▶️ Start tradingbot")
stop = st.sidebar.button("⏹️ Stopp tradingbot")

if start:
    if not st.session_state.bot_running:
        st.session_state.process = subprocess.Popen(["python", "smart_order_router.py"])
        st.session_state.bot_running = True
        st.success("Tradingbot startet!")
    else:
        st.info("Bot kjører allerede.")

if stop:
    if st.session_state.bot_running:
        st.session_state.process.terminate()
        st.session_state.bot_running = False
        st.warning("Tradingbot stoppet.")
    else:
        st.info("Bot er ikke i gang.")

# Statusindikator
status_msg = "🟢 Tradingbot kjører" if st.session_state.bot_running else "🔴 Tradingbot er stoppet"
st.subheader(f"Status: {status_msg}")

with st.expander("📋 Ordrehistorikk", expanded=True):
    data = load_data()
    if data.empty:
        st.warning("Ingen ordre funnet.")
    else:
        st.dataframe(data, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(data["symbol"].value_counts())
        with col2:
            if "avg_price" in data.columns:
                st.line_chart(data.groupby("symbol")["avg_price"].mean())

with st.expander("📈 Åpne posisjoner (m/ Take Profit / Stop Loss)", expanded=True):
    positions_df = get_open_positions()
    if positions_df.empty:
        st.info("Ingen åpne posisjoner")
    else:
        st.dataframe(positions_df, use_container_width=True)
        st.bar_chart(positions_df.set_index("symbol")["market_value"].astype(float))
        st.line_chart(positions_df.set_index("symbol")["unrealized_pl"].astype(float))

with st.expander("📉 Porteføljeutvikling og realisert P&L", expanded=True):
    closed_df = get_closed_positions()
    if not closed_df.empty:
        closed_df["timestamp"] = pd.to_datetime(closed_df["timestamp"])
        pnl_df = closed_df.copy()
        pnl_df["signed_amount"] = pnl_df.apply(lambda row: row["amount"] if row["side"] == "sell" else -row["amount"], axis=1)
        pnl_over_time = pnl_df.groupby(pd.Grouper(key="timestamp", freq="1H")).sum(numeric_only=True)[["signed_amount"]].cumsum()
        st.line_chart(pnl_over_time.rename(columns={"signed_amount": "Realisert P&L over tid"}))

with st.expander("💰 Reali­serte handler og skatteklar rapport", expanded=True):
    if closed_df.empty:
        st.info("Ingen realiserte handler funnet.")
    else:
        st.dataframe(closed_df, use_container_width=True)
        if st.button("📤 Generer skatteklar Excel-rapport"):
            export_tax_report(closed_df)

st.caption("Sist oppdatert: " + time.strftime("%H:%M:%S"))
