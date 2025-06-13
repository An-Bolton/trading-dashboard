
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess
import sqlite3
import alpaca_trade_api as tradeapi
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

import streamlit as st
import pandas as pd
import time
from datetime import datetime

st.set_page_config(
    page_title="Smart Order Router",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# 🎨 Ekstra styling med Tailwind-lignende CSS
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        background-color: #ffffff;
        color: #111827;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        padding: 0.4rem;
    }
    .stMetric {
        background-color: #dbdbdb;
        color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)
# Nå kan du skrive resten av appen
with open("trading_dashboard_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📈 Trading Dashboard med smart router ordre algortime-utførelse")

def ticker_tape_html():
    import yfinance as yf
    tickers = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^DJI": "Dow Jones",
        "^GDAXI": "DAX",
        "^FTSE": "FTSE 100"
        # "^OSEAX": "OSEAX",
        # "^N225": "Nikkei 225"
    }

    try:
        data = yf.download(tickers=list(tickers.keys()), period="1d", interval="1m", progress=False)
        prices = []
        warning = False

        for symbol, name in tickers.items():
            latest = data["Close"][symbol].dropna()[-1]
            prev = data["Close"][symbol].dropna()[-2]
            change = latest - prev
            pct = (change / prev) * 100
            arrow = "▲" if change > 0 else "▼"
            color = "#16a34a" if change > 0 else "#dc2626"
            prices.append(f"<span style='color:{color}'>{name}: {latest:.2f} {arrow} ({pct:+.2f}%)</span>")
            if pct <= -1:
                warning = True

        tape = " | ".join(prices)

        if warning:
            tape = f"<span style='color:#dc2626; font-weight:bold;'>🔔 En eller flere indekser har falt mer enn 1 %!</span> | {tape}"

        return tape

    except Exception:
        return "🛑 Klarte ikke hente indeksdata"


DB_PATH = "orders.db"
ALPACA_API_KEY = "PKQXL7LF4EAFA6R9VRDS"
ALPACA_SECRET_KEY = "vVnLT4aasb8wJgHPabAXpjxldudLpIdFahKemBDH"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL, api_version="v2")

st.markdown(f"""
<marquee behavior="scroll" direction="left" scrollamount="4" style="color: #0f172a; font-size: 16px; background-color: #f1f5f9; padding: 8px; border-radius: 8px;">
{ticker_tape_html()}
</marquee>
""", unsafe_allow_html=True)

st.markdown("""
### 📚 Valideringsinfo
**In-sample vs. Out-of-sample:**
- 🟢 *In-sample*: Treningsdata brukt til å utvikle og tilpasse strategien.
- 🔵 *Out-of-sample*: Nye data brukt til testing og validering for å vurdere robusthet.
""")

st_autorefresh(interval=30 * 1000, key="datarefresh")

st.markdown("## 📦 Nåverdi av porteføljen")

try:
    # Hent åpne posisjoner
    positions = api.list_positions()
    rows = []
    total_value = 0.0

    for p in positions:
        symbol = p.symbol
        qty = float(p.qty)
        current_price = api.get_latest_trade(symbol).price
        market_value = qty * current_price
        total_value += market_value

        rows.append({
            "Ticker": symbol,
            "Antall": qty,
            "Pris nå": f"${current_price:.2f}",
            "Verdi": f"${market_value:,.2f}"
        })

    if rows:
        st.dataframe(pd.DataFrame(rows))
        st.metric("💼 Total porteføljeverdi", f"${total_value:,.2f}")
    else:
        st.info("Ingen åpne posisjoner.")

except Exception as e:
    st.error(f"Kunne ikke hente porteføljeverdi: {e}")

st.sidebar.header("🛒 MENY")

strategy = st.sidebar.selectbox("📊 Velg forhåndsinnstilt strategi",
                                ["🎯 Normal (TP 8%, SL 5%)", "🛡️ Trygg (TP 4%, SL 2%)", "🔥 Aggressiv (TP 15%, SL 10%)"])


if strategy == "🎯 Normal (TP 8%, SL 5%)":
    tp_pct, sl_pct = 8, 5
elif strategy == "🛡️ Trygg (TP 4%, SL 2%)":
    tp_pct, sl_pct = 4, 2
elif strategy == "🔥 Aggressiv (TP 15%, SL 10%)":
    tp_pct, sl_pct = 15, 10

st.sidebar.markdown("---")
tp_pct = st.sidebar.slider("🎯 Juster Take Profit (%)", 1, 20, tp_pct)
sl_pct = st.sidebar.slider("🛑 Juster Stop Loss (%)", 1, 20, sl_pct)


# --- Funksjoner ---

def log_trade(symbol, side, qty, price, log_source="bot"):
    try:
        with open("realiserte_handler.log", "r") as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line or "timestamp" in line:
                continue
            parts = line.split(",")
            if len(parts) == 7:
                # Reorganiser hvis første element ikke er timestamp
                try:
                    datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                    cleaned_lines.append(line)
                except ValueError:
                    # Timestamp er sannsynligvis sist – reorganiser
                    new_line = ",".join([parts[5], parts[0], parts[1], parts[2], parts[3], parts[4], "manual"])
                    cleaned_lines.append(new_line)

        with open("realiserte_handler.log", "w") as f:
            f.write("timestamp,symbol,side,qty,price,amount,source\n")
            for line in cleaned_lines:
                f.write(line + "\n")

        st.success(f"✅ Renset {len(cleaned_lines)} linjer – alt i korrekt format!")

    except Exception as e:
        st.error(f"Kunne ikke rense loggfil: {e}")


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
    from pandas import ExcelWriter

    # Beregninger
    df["signed_amount"] = df.apply(
        lambda row: row["amount"] if row["side"] == "sell" else -row["amount"], axis=1
    )
    pnl_per_ticker = df.groupby("symbol")["signed_amount"].sum().reset_index().rename(columns={"signed_amount": "P&L"})
    top5 = df.groupby("symbol")["qty"].sum().nlargest(5).reset_index().rename(columns={"qty": "Totalt volum"})

    # Filnavn
    dato = datetime.today().date()
    export_file = f"skatteklar_rapport_{dato}.xlsx"

    # Eksporter til Excel med flere ark
    with ExcelWriter(export_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Handler", index=False)
        pnl_per_ticker.to_excel(writer, sheet_name="P&L per ticker", index=False)
        top5.to_excel(writer, sheet_name="Topp 5 tickere", index=False)

    # Streamlit UI
    st.success(f"Excel-rapport generert: {export_file}")
    with open(export_file, "rb") as file:
        st.download_button(
            label="📥 Last ned skatteklar rapport",
            data=file,
            file_name=export_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
def calculate_sharpe(df):
    if df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["signed_amount"] = df.apply(
        lambda row: row["amount"] if row.get("side") == "sell" else -row["amount"],
        axis=1
    )
    df = df.dropna(subset=["signed_amount"])
    df["return"] = df["signed_amount"].pct_change().fillna(0)
    mean_return = df["return"].mean()
    std_return = df["return"].std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    return sharpe_ratio


# Kontrollpanel

# 📥 Handle direkte fra dashboardet
st.sidebar.header("🛒 Handle manuelt")
symbol = st.sidebar.text_input("Ticker (f.eks. AAPL)")
qty = st.sidebar.number_input("Antall aksjer", min_value=1, step=1, value=1)
limit_price_input = st.sidebar.text_input("Limitpris (valgfritt)", placeholder="Fyll inn for limit-ordre")

col1, col2 = st.sidebar.columns(2)

valid_limit = True  # Til visning før ordren sendes
order_message = ""
color = "gray"

if symbol and limit_price_input:
    try:
        current_price = api.get_latest_trade(symbol.upper()).price
        limit_price = float(limit_price_input)
        diff_pct = ((limit_price - current_price) / current_price) * 100

        if diff_pct > 20:
            valid_limit = False
            order_message = f"⚠️ Limit-pris er **{diff_pct:.1f}% over** markedspris (${current_price:.2f})"
            color = "red"
        elif diff_pct < -20:
            valid_limit = False
            order_message = f"⚠️ Limit-pris er **{abs(diff_pct):.1f}% under** markedspris (${current_price:.2f})"
            color = "orange"
        else:
            order_message = f"✅ Limitpris er **{abs(diff_pct):.1f}% innenfor** markedspris (${current_price:.2f})"
            color = "green"

        st.sidebar.markdown(f"<div style='color:{color}; font-weight:bold;'>{order_message}</div>",
                            unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"❌ Klarte ikke hente markedspris: {e}")

# Ordrebok
with st.expander("📈 Ordrebok (Best Bid/Ask)", expanded=False):
    ordrebok_ticker = st.text_input("Velg ticker for ordrebok", value=symbol.upper() if symbol else "")
    if ordrebok_ticker:
        try:
            snapshot = api.get_snapshot(ordrebok_ticker.upper())
            bid = snapshot.latest_quote.bid_price
            ask = snapshot.latest_quote.ask_price
            st.metric("Best BID (kjøpspris)", f"${bid:.2f}")
            st.metric("Best ASK (salgspris)", f"${ask:.2f}")
            spread = ask - bid
            st.caption(f"🔍 Spread: ${spread:.2f}")
        except Exception as e:
            st.warning(f"Kunne ikke hente ordrebok for {ordrebok_ticker.upper()}: {e}")

# Kjøpsordre
if col1.button("📈 Kjøp"):
    try:
        order_type = "limit" if limit_price_input else "market"
        price = float(limit_price_input) if limit_price_input else None
        current_price = api.get_latest_trade(symbol.upper()).price

        # Avvis for høy limitpris (> +20 %)
        if order_type == "limit" and price > current_price * 1.2:
            st.warning(
                f"⛔ Limitpris (${price:.2f}) overstiger 20 % over markedspris (${current_price:.2f}) – kjøpsordre avvist.")
        else:
            api.submit_order(
                symbol=symbol.upper(),
                qty=qty,
                side="buy",
                type=order_type,
                time_in_force="gtc",
                limit_price=price if order_type == "limit" else None
            )
            log_trade(symbol.upper(), "buy", qty, price or current_price, log_source="manual")
            st.success(f"✅ Sendte KJØP av {qty} {symbol.upper()} til {'limitpris' if price else 'markedspris'}")
    except Exception as e:
        st.error(f"Feil ved kjøpsordre: {e}")

# Salgsordre
if col2.button("📉 Selg"):
    try:
        order_type = "limit" if limit_price_input else "market"
        price = float(limit_price_input) if limit_price_input else None
        current_price = api.get_latest_trade(symbol.upper()).price

        # Avvis for lav limitpris (< -20 %)
        if order_type == "limit" and price < current_price * 0.8:
            st.warning(
                f"⛔ Limitpris (${price:.2f}) er mer enn 20 % under markedspris (${current_price:.2f}) – salgsordre avvist.")
        else:
            api.submit_order(
                symbol=symbol.upper(),
                qty=qty,
                side="sell",
                type=order_type,
                time_in_force="gtc",
                limit_price=price if order_type == "limit" else None
            )
            log_trade(symbol.upper(), "sell", qty, price or current_price, log_source="manual")
            st.success(f"✅ Sendte SALG av {qty} {symbol.upper()} til {'limitpris' if price else 'markedspris'}")
    except Exception as e:
        st.error(f"Feil ved salgsordre: {e}")

# 📐 Risikojustert posisjonsstørrelse
st.sidebar.header("🧮 Risikojustert posisjonsstørrelse")
account_equity = st.sidebar.number_input("Kontoverdi ($)", min_value=100.0, value=10000.0)
risk_pct = st.sidebar.slider("Risiko per trade (%)", 0.1, 5.0, 1.0)
stop_pct = st.sidebar.slider("Stop-Loss (%)", 0.5, 20.0, 2.0)

risk_amount = account_equity * (risk_pct / 100)
stop_loss_frac = stop_pct / 100

if symbol:
    try:
        latest_price = api.get_latest_trade(symbol.upper()).price
        per_share_risk = latest_price * stop_loss_frac
        max_shares = int(risk_amount / per_share_risk) if per_share_risk > 0 else 0
        st.sidebar.metric("📏 Anbefalt maks antall aksjer", f"{max_shares} stk")
        st.sidebar.caption(f"(Basert på ${latest_price:.2f} pris og {stop_pct}% SL)")
    except Exception as e:
        st.sidebar.warning(f"Klarte ikke hente pris for {symbol.upper()}: {e}")

if "bot_running" not in st.session_state:
    st.session_state.bot_running = False

st.sidebar.header("🛠 Kontrollpanel")

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
...

if st.button("🔄 Oppdater portefølje manuelt"):
    st.rerun()

# Status
status_msg = "🟢 Tradingbot kjører" if st.session_state.bot_running else "🔴 Tradingbot er stoppet"
status_color = "#22c55e" if st.session_state.bot_running else "#ef4444"
st.markdown(f"""
<div style='display: flex; align-items: center; margin-top: 1rem;'>
    <div style='width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 10px; animation: pulse 2s infinite;'></div>
    <h3 style='margin: 0'>{status_msg}</h3>
</div>
<style>
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 " + "REPLACEME" + "; }}
  70% {{ box-shadow: 0 0 0 10px transparent; }}
  100% {{ box-shadow: 0 0 0 0 transparent; }}
}}
</style>
""", unsafe_allow_html=True)

# Ordrehistorikk-seksjon
st.markdown("""
<div style='background-color: #ffffff; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
""", unsafe_allow_html=True)
with st.expander("📋 Ordrehistorikk", expanded=True):
    data = load_data()
    if data.empty:
        st.warning("Ingen ordre funnet.")
    else:
        st.dataframe(data, use_container_width=True, key=int(time.time()))
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(data["symbol"].value_counts())
        with col2:
            if "avg_price" in data.columns:
                st.line_chart(data.groupby("symbol")["avg_price"].mean())
st.markdown("</div>", unsafe_allow_html=True)

# Åpne posisjoner
st.markdown("""<div style='background-color: #ffffff; padding: 1rem; margin-top: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
""", unsafe_allow_html=True)
with st.expander("📈 Åpne posisjoner (TP/SL)", expanded=True):
    positions_df = get_open_positions()
    if positions_df.empty:
        st.info("Ingen åpne posisjoner")
    else:
        st.dataframe(positions_df, use_container_width=True, key=int(time.time()))
        st.bar_chart(positions_df.set_index("symbol")["market_value"].astype(float))
        st.line_chart(positions_df.set_index("symbol")["unrealized_pl"].astype(float))
st.markdown("</div>", unsafe_allow_html=True)

# P&L og Sharpe
import logging

LOGFILE = "realiserte_handler.log"
if not os.path.exists(LOGFILE):
    with open(LOGFILE, "w") as f:
        f.write("timestamp,symbol,side,qty,price,amount\n")

with st.expander("📉 Portefølje og Sharpe", expanded=True):
    closed_df = get_closed_positions()
    if not closed_df.empty:
        closed_df["timestamp"] = pd.to_datetime(closed_df["timestamp"])
        pnl_df = closed_df.copy()
        # Logg nye handler
        logged = pd.read_csv(LOGFILE, on_bad_lines='skip')
        new = pnl_df[~pnl_df["timestamp"].isin(logged["timestamp"])]
        if not new.empty:
            new.to_csv(LOGFILE, mode="a", index=False, header=False)
            st.success(f"🎉 {len(new)} ny(e) handler logget!")

        st.markdown("## 📦 Nåverdi av porteføljen")

        try:
            # Hent åpne posisjoner
            positions = api.list_positions()
            rows = []
            total_value = 0.0

            for p in positions:
                symbol = p.symbol
                qty = float(p.qty)
                current_price = api.get_latest_trade(symbol).price
                market_value = qty * current_price
                total_value += market_value

                rows.append({
                    "Ticker": symbol,
                    "Antall": qty,
                    "Pris nå": f"${current_price:.2f}",
                    "Verdi": f"${market_value:,.2f}"
                })

            if rows:
                st.dataframe(pd.DataFrame(rows))
                st.metric("💼 Total porteføljeverdi", f"${total_value:,.2f}")
            else:
                st.info("Ingen åpne posisjoner.")

        except Exception as e:
            st.error(f"Kunne ikke hente porteføljeverdi: {e}")

        # Daglig P&L
        pnl_df["signed_amount"] = pnl_df.apply(lambda row: row["amount"] if row["side"] == "sell" else -row["amount"],
                                               axis=1)
        daily_pnl = pnl_df.groupby(pnl_df["timestamp"].dt.date)["signed_amount"].sum().reset_index()
        st.line_chart(daily_pnl.rename(columns={"timestamp": "Dato", "signed_amount": "Daglig P&L"}).set_index("Dato"))
        pnl_df["signed_amount"] = pnl_df.apply(lambda row: row["amount"] if row["side"] == "sell" else -row["amount"],
                                               axis=1)
        pnl_over_time = pnl_df.groupby(pd.Grouper(key="timestamp", freq="1H")).sum(numeric_only=True)[
            ["signed_amount"]].cumsum()
        st.line_chart(pnl_over_time.rename(columns={"signed_amount": "Realisert P&L over tid"}))
        sharpe = calculate_sharpe(pnl_df)
        if sharpe is not None:
            st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}")
        total_pnl = pnl_df["signed_amount"].sum()
        pnl_label = "💰 Total gevinst" if total_pnl >= 0 else "💸 Totalt tap"
        pnl_color = "green" if total_pnl >= 0 else "red"
        st.metric(pnl_label, f"${total_pnl:,.2f}")
    else:
        st.info("Ingen realiserte handler ennå – P&L og Sharpe vises når salg er gjennomført.")
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("📆 Historisk P&L per uke og måned", expanded=False):
    try:
        df = pd.read_csv("realiserte_handler.log", on_bad_lines="skip")

        # Filtrer bort linjer som mangler nødvendig struktur
        df = df[df["timestamp"].str.contains(r"\d{4}-\d{2}-\d{2}", na=False)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["timestamp", "amount", "side"])

        # Beregn signed_amount
        df["signed_amount"] = df.apply(
            lambda row: row["amount"] if str(row["side"]).strip().lower() == "sell" else -row["amount"],
            axis=1
        )
        # P&L per uke og måned
        weekly = df.groupby(pd.Grouper(key="timestamp", freq="W"))["signed_amount"].sum()
        monthly = df.groupby(pd.Grouper(key="timestamp", freq="M"))["signed_amount"].sum()

        st.subheader("📅 Ukentlig P&L")
        st.bar_chart(weekly)

        st.subheader("🗓️ Månedlig P&L")
        st.line_chart(monthly)

    except Exception as e:
        st.warning(f"Kunne ikke laste historisk P&L: {e}")

# Realiserte handler og eksport

# 📄 Tidligere sluttsedler
with st.expander("📄 Tidligere sluttsedler", expanded=False):
    try:
        df = pd.read_csv("realiserte_handler.log")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")  # ← viktig linje!
        df = df.dropna(subset=["timestamp", "symbol", "side", "qty", "price", "amount"])

        df = df.sort_values(by="timestamp", ascending=False)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📌 Summer:")
        col1, col2 = st.columns(2)
        col1.metric("Antall handler", len(df))
        col2.metric("Total verdi", f"${df['amount'].sum():,.2f}")

    except Exception as e:
        st.warning(f"Kunne ikke laste sluttsedler: {e}")

st.markdown("""
<div style='background-color: #ffffff; padding: 1rem; margin-top: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
""", unsafe_allow_html=True)

with st.expander("💰 Skatteklar eksport", expanded=True):
    st.markdown("**Filtrer handler:**")
    if not closed_df.empty:
        # Unngå navnekonflikt
        filtered_export_df = closed_df.copy()

        # Filtrering
        unique_sources = list(filtered_export_df["source"].unique()) if "source" in filtered_export_df.columns else []
        selected_sources = st.multiselect("Velg kilde(r)", unique_sources, default=unique_sources)
        tickers = filtered_export_df["symbol"].unique().tolist()
        selected_ticker = st.selectbox("Velg ticker (eller Alle)", ["Alle"] + tickers, key="ticker_selectbox")

        if "source" in filtered_export_df.columns:
            filtered_export_df = filtered_export_df[filtered_export_df["source"].isin(selected_sources)]
        if selected_ticker != "Alle":
            filtered_export_df = filtered_export_df[filtered_export_df["symbol"] == selected_ticker]

        st.dataframe(filtered_export_df, use_container_width=True, key=int(time.time()))

        # 📊 Fordeling av handler per kilde (kakediagram)
        st.write("Antall rader etter filtrering:", len(filtered_export_df))

        # Sikre at 'source' finnes og fyll eventuelle hull
        if "source" not in filtered_export_df.columns:
            filtered_export_df["source"] = "manual"
        else:
            filtered_export_df["source"] = filtered_export_df["source"].replace("ukjent", "manual").fillna("manual")

        if not filtered_export_df.empty:
            valid_sources = filtered_export_df["source"].value_counts()

            if not valid_sources.empty:
                import matplotlib.pyplot as plt

                st.markdown("### 📊 Fordeling av handler per kilde")
                fig, ax = plt.subplots()
                ax.pie(valid_sources, labels=valid_sources.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.info("Ingen gyldige 'source'-verdier å vise i kakediagrammet.")
        else:
            st.info("Ingen handler å vise som kakediagram ennå. Prøv å velge en annen kilde eller ticker.")

        # Sum per kilde
        st.markdown("**Kilde**: 'manual' = jeg handlet selv, 'bot' = automatisk handler")
        st.dataframe(
            filtered_export_df.groupby("source")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "Sum per kilde"})
        )

        # Rapport og nedlasting
        if st.button("📤 Generer skatteklar Excel-rapport"):
            export_tax_report(filtered_export_df)

        with open(LOGFILE, "rb") as log_file:
            st.download_button(
                label="📑 Last ned loggfil for realiserte handler",
                data=log_file,
                file_name="realiserte_handler.log",
                mime="text/csv"
            )

st.caption("⏱ Sist oppdatert: " + time.strftime("%H:%M:%S"))