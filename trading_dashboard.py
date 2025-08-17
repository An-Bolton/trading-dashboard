import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import plotly.express as px
import streamlit as st
import sqlite3
import fpdf
import openpyxl
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from fpdf import FPDF
import fpdf
import os
import io
import sqlite3
import uuid

from derivater import AlpacaAdapter, occ_symbol, bs_price_greeks, OptionInput

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        min-width: 100vw;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        padding: 10px 28px;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------- LOGO ----------------
def show_logo():
    if os.path.exists("Farmand1.png"):
        st.image("Farmand1.png", width=230)
    else:
        st.markdown(
            "<h1 style='color:#1e375a;font-weight:900;'>Farmand & Morse Securities</h1>"
            ,
            unsafe_allow_html=True,
        )

# ---------------- LOGIN ----------------
def login():
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        show_logo()
        with st.form("login_form"):
            st.write("🔒 **Logg inn for tilgang**")
            username = st.text_input("Brukernavn")
            password = st.text_input("Passord", type="password")
            submit = st.form_submit_button("Logg inn")
            if submit:
                if username == "admin" and password == "jaujaujau12":
                    st.session_state['logged_in'] = True
                    st.success("Innlogging vellykket!")
                    st.rerun()
                else:
                    st.error("Feil brukernavn/passord")
        st.stop()

        import sqlite3
        from datetime import datetime
        from datetime import datetime

        filled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Eksakt tidspunkt, tekstformat

        def log_trade(symbol, qty, side, price):
            conn = sqlite3.connect("orders.db")
            c = conn.cursor()
            c.execute("INSERT INTO trades (symbol, qty, side, price, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (symbol, qty, side, price, datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
def get_trade_log():
    conn = sqlite3.connect("orders.db")
    df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
    conn.close()
    return df

TICKERS = [
    "^GSPC", "^DJI", "^NDX", "^IXIC", "^OSEBX", "^GDAXI", "AAPL", "TSLA", "MSFT"
]
TICKER_LABELS = {
    "^GSPC": "S&P500",
    "^DJI": "Dow",
    "^NDX": "Nasdaq100",
    "^IXIC": "Nasdaq Comp.",
    "^OSEBX": "OSEBX",
    "^GDAXI": "DAX"
}

def fetch_ticker_data(tickers):
    last, pct = {}, {}
    for t in tickers:
        try:
            data = yf.download(t, period="1d", interval="1m", progress=False, threads=False)
            close = data["Close"].dropna()
            open_ = data["Open"].dropna()
            if len(close) and len(open_):
                last[t] = float(close.iloc[-1])
                pct[t] = float((close.iloc[-1] - open_.iloc[0]) / open_.iloc[0] * 100)
        except Exception as e:
            pass
    return last, pct

last, pct = fetch_ticker_data(TICKERS)

tape = ""
for t in TICKERS:
    l = last.get(t, None)
    p = pct.get(t, None)
    if l is not None and p is not None:
        color = "#32CD32" if p >= 0 else "#FF4B4B"
        sign = "+" if p >= 0 else ""
        label = TICKER_LABELS.get(t, t)
        tape += f"""<span style="display:inline-block; margin-right:18px; vertical-align:middle;">
            <b>{label}</b>
            <span style="color:{color}; font-weight:600;">{l:.2f} ({sign}{p:.2f}%)</span>
        </span>"""

ticker_html = f"""
<div style="
    background: #111927;
    border-radius: 12px;
    padding: 6px 0 6px 0;
    margin-bottom: 14px;
    overflow: hidden;
    white-space: nowrap;
">
    <div style="
        display: inline-block;
        white-space: nowrap;
        animation: ticker 18s linear infinite;
        font-family: 'Menlo', 'Consolas', monospace;
        font-size: 18px;
        height: 26px;
        line-height: 26px;
    " id="ticker-tape">
        {tape}
    </div>
</div>
<style>
@keyframes ticker {{
    0%   {{ transform: translateX(100%); }}
    100% {{ transform: translateX(-100%); }}
}}
#ticker-tape span {{
    display: inline-block;
    margin-right: 10px;
    vertical-align: middle;
    white-space: nowrap;
}}
</style>
"""
st.markdown(ticker_html, unsafe_allow_html=True)
# ---------------- ALPACA ----------------
def get_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        API_KEY = st.secrets["ALPACA_API_KEY"]
        API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        return trading_client
    except Exception as e:
        st.error(f"❌ Alpaca-klient ikke tilgjengelig: {e}")
        return None

# ---------------- HANDLE AKSJER ----------------
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

def handle_order(trading_client):
    st.subheader("Handle aksjer")
    symbol = st.text_input("Ticker-symbol (f.eks. AAPL)", key="order_symbol")
    qty = st.number_input("Antall aksjer", min_value=1, value=1, step=1, key="order_qty")
    side_str = st.selectbox("Kjøp eller selg?", ["buy", "sell"], key="order_side")
    order_type_str = st.selectbox("Ordretype", ["market", "limit", "stop", "trailing_stop"], key="order_type")

    limit_price = stop_price = trail_price = trail_percent = None
    if order_type_str == "limit":
        limit_price = st.number_input("Limitpris", min_value=0.0, value=0.0, step=0.01, key="order_limit")
    elif order_type_str == "stop":
        stop_price = st.number_input("Stoppris", min_value=0.0, value=0.0, step=0.01, key="order_stop")
    elif order_type_str == "trailing_stop":
        use_trail_price = st.checkbox("Bruk trail-price (ellers prosent)", key="order_trail_type")
        if use_trail_price:
            trail_price = st.number_input("Trail amount (USD)", min_value=0.0, value=0.0, step=0.01, key="order_trail_price")
        else:
            trail_percent = st.number_input("Trail percent (%)", min_value=0.0, value=1.0, step=0.01, key="order_trail_percent")

    tif_str = st.selectbox("Time in force", ["gtc", "day", "opg", "cls", "ioc", "fok"], index=0, key="order_tif")

    # Konverter til enums:
    side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
    order_type = OrderType(order_type_str)
    tif = TimeInForce(tif_str)

    if st.button("Send ordre", key="order_submit2"):
        try:
            order_kwargs = dict(
                symbol=symbol.upper(),
                qty=int(qty),
                side=side,
                type=order_type,
                time_in_force=tif,
            )
            if order_type == OrderType.LIMIT and limit_price:
                order_kwargs["limit_price"] = float(limit_price)
            if order_type == OrderType.STOP and stop_price:
                order_kwargs["stop_price"] = float(stop_price)
            if order_type == OrderType.TRAILING_STOP:
                if trail_price and trail_price > 0:
                    order_kwargs["trail_price"] = float(trail_price)
                elif trail_percent and trail_percent > 0:
                    order_kwargs["trail_percent"] = float(trail_percent)

            order_request = OrderRequest(**order_kwargs)
            order = trading_client.submit_order(order_request)
            st.success(f"✅ Ordre sendt! ID: {order.id}")
        except Exception as e:
            st.error(f"❌ Feil ved ordre: {e}")


# --------------- HJELPEFUNKSJONER EKSEMPEL -----------------
def get_account(trading_client):
    try:
        account = trading_client.get_account()
        return account
    except Exception:
        return {"portfolio_value": 100000, "cash": 50000}

def get_positions(trading_client):
    try:
        positions = trading_client.get_all_positions()
        # Gjør om til dicts hvis ikke allerede
        return [pos.__dict__ for pos in positions]
    except Exception:
        return []

def get_orders(trading_client):
    try:
        orders = trading_client.get_orders(status="all")
        return [o.__dict__ for o in orders]
    except Exception:
        return []

# ---------------- MAIN ----------------
def main():
    login()
    show_logo()
    trading_client = get_trading_client()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "Portefølje", "Handler", "Teknisk Analyse", "Tradingbot",
        "Ordrebok", "Backtest", "Nyheter", "Rapporter", "Rebalanser", "Om plattformen", "Markedsoversikt", "Makroutsikter", "Derivater"
    ])

    from macro_forecast import macro_compare_tab

    with tab12:

        macro_compare_tab()

    # ---------- TAB 1: PORTFØLJE ----------

    order_id = str(uuid.uuid4())

    with tab1:
        st.header("Porteføljeoversikt")
        if trading_client:
            account = get_account(trading_client)
            portfolio_value = float(getattr(account, "portfolio_value", 0))
            st.metric("Porteføljeverdi", f"${portfolio_value:,.2f}")
            cash = float(getattr(account, "cash", 0))
            st.metric("Ledig kontantbeholdning", f"${cash:,.2f}")

            st.subheader("Aktive posisjoner")
            positions = get_positions(trading_client)
            if positions:
                df_pos = pd.DataFrame(positions)
                for col in ["market_value", "qty", "cost_basis"]:
                    if col in df_pos.columns:
                        df_pos[col] = pd.to_numeric(df_pos[col], errors="coerce")
                st.dataframe(df_pos)
            else:
                df_pos = pd.DataFrame()
                st.info("Ingen åpne posisjoner.")

                # ----------- AVANSERT HANDELSVINDU -----------
            st.markdown("---")
            st.subheader("Utfør handel (avansert)")

            with st.form("advanced_trade_form"):
                    symbol = st.text_input("Ticker (f.eks. AAPL)").upper()
                    qty = st.number_input("Antall aksjer", min_value=1, step=1)
                    side = st.selectbox("Kjøp eller selg?", ["KJØP", "SELG"])
                    order_type = st.selectbox("Ordretype", ["MARKET", "LIMIT", "STOP"])
                    price = None
                    stop_price = None

                    if order_type == "LIMIT":
                        price = st.number_input("Limitpris", min_value=0.0, step=0.01, format="%.2f")
                    if order_type == "STOP":
                        stop_price = st.number_input("Stoppkurs", min_value=0.0, step=0.01, format="%.2f")

                    submitted = st.form_submit_button("Gjennomfør handel")

                    if submitted:
                        if not symbol.isalpha():
                            st.error("Ugyldig ticker-symbol.")
                        else:
                            try:
                                from alpaca.trading.requests import (
                                    MarketOrderRequest,
                                    LimitOrderRequest,
                                    StopOrderRequest
                                )
                                from alpaca.trading.enums import OrderSide, TimeInForce

                                order_req = None
                                if order_type == "MARKET":
                                    order_req = MarketOrderRequest(
                                        symbol=symbol,
                                        qty=int(qty),
                                        side=OrderSide.BUY if side == "KJØP" else OrderSide.SELL,
                                        time_in_force=TimeInForce.DAY,
                                    )
                                elif order_type == "LIMIT":
                                    if price is None or price <= 0:
                                        st.error("Du må angi en gyldig limitpris.")
                                        st.stop()
                                    order_req = LimitOrderRequest(
                                        symbol=symbol,
                                        qty=int(qty),
                                        side=OrderSide.BUY if side == "KJØP" else OrderSide.SELL,
                                        limit_price=float(price),
                                        time_in_force=TimeInForce.DAY,
                                    )
                                elif order_type == "STOP":
                                    if stop_price is None or stop_price <= 0:
                                        st.error("Du må angi en gyldig stoppkurs.")
                                        st.stop()
                                    order_req = StopOrderRequest(
                                        symbol=symbol,
                                        qty=int(qty),
                                        side=OrderSide.BUY if side == "KJØP" else OrderSide.SELL,
                                        stop_price=float(stop_price),
                                        time_in_force=TimeInForce.DAY,
                                    )
                                else:
                                    st.error("Ugyldig ordretype.")
                                    st.stop()

                                order = trading_client.submit_order(order_req)
                                st.success(f"✅ Ordre sendt! ID: {order.id}")

                                import time as _time
                                _time.sleep(1)
                                order_filled = trading_client.get_order_by_id(order.id)
                                filled_time = str(order_filled.filled_at)
                                filled_price = order_filled.filled_avg_price
                                venue = getattr(order_filled, "filled_venue", None)

                                st.info(f'''
                                **Handel utført!**
                                - Status: {order_filled.status}
                                - Utført på: {venue if venue else 'Ukjent'}
                                - Tidspunkt: {filled_time}
                                - Pris: {filled_price}
                                ''')

                                # Logg i egen tabell
                                conn = sqlite3.connect("orders.db")
                                c = conn.cursor()
                                c.execute("""
                                    CREATE TABLE IF NOT EXISTS order_fills (
                                        order_id TEXT, 
                                        filled_at TEXT, 
                                        filled_price REAL, 
                                        venue TEXT
                                    )
                                """)
                                c.execute(
                                    "INSERT INTO order_fills (order_id, filled_at, filled_price, venue) VALUES (?, ?, ?, ?)",
                                    (str(order.id), str(filled_time), filled_price, venue)
                                )
                                conn.commit()
                                conn.close()

                            except Exception as e:
                                st.error(f"Feil ved handel: {e}")

            # Porteføljegraf (verdiutvikling siste 30 dager)
            import plotly.graph_objects as go
            from datetime import datetime

            try:
                history = trading_client.get_portfolio_history()
                df = pd.DataFrame({
                    "date": [datetime.fromtimestamp(ts) for ts in history.timestamp],
                    "equity": history.equity
                })
                st.subheader("Porteføljeverdi siste 30 dager")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=df["equity"],
                    mode="lines+markers",
                    name="Porteføljeverdi"
                ))
                fig.update_layout(
                    xaxis_title="Dato",
                    yaxis_title="USD",
                    template="plotly_white",
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Kunne ikke hente porteføljehistorikk: {e}")

            # Donut-chart for porteføljefordeling (valgfritt)
            import plotly.express as px
            if not df_pos.empty and "market_value" in df_pos.columns:
                st.subheader("Fordeling av portefølje")
                fig = px.pie(df_pos[df_pos["market_value"] > 0], values="market_value", names="symbol", hole=0.4,
                             title="Porteføljefordeling")
                st.plotly_chart(fig, use_container_width=True)

                st.header("P&L-utvikling (siste 30 dager)")

                try:
                    # For Alpaca, porteføljehistorikk: daglig equity/porteføljeverdi
                    history = trading_client.get_portfolio_history()  # ingen 'period'/'timeframe'!
                    # Lag DataFrame
                    df_pnl = pd.DataFrame({
                        "date": [datetime.fromtimestamp(ts) for ts in history.timestamp],
                        "equity": history.equity
                    })
                    df_pnl["P&L"] = df_pnl["equity"].diff().fillna(0)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_pnl["date"], y=df_pnl["P&L"],
                        mode="lines+markers", name="Daglig P&L"
                    ))
                    fig.update_layout(
                        title="Daglig P&L (porteføljeutvikling)",
                        xaxis_title="Dato", yaxis_title="P&L ($)",
                        template="plotly_white", showlegend=False, height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Kunne ikke hente porteføljehistorikk: {e}")

    # ---------- TAB 2: HANDLER ----------
    with tab2:
        st.header("Handler (utfylte handler fra Alpaca)")
        if trading_client:
            # Oppdater-knapp
            if st.button("Oppdater handler", key="refresh_trades"):
                st.rerun()
            try:
                orders = trading_client.get_orders()
                df_orders = pd.DataFrame([o.__dict__ for o in orders])

                # Robust håndtering av kolonner
                if not df_orders.empty:
                    if "status" not in df_orders.columns:
                        df_orders["status"] = "unknown"

                    # Fyll evt. NaN
                    df_orders["status"] = df_orders["status"].fillna("unknown")
                    filled_orders = df_orders[df_orders["status"] == "filled"]

                    # Vis alle/utfylte: Velg selv!
                    mode = st.radio("Vis", ["Kun utfylte handler", "Alle handler"], index=0, key="trade_mode")
                    show_cols = ["id", "symbol", "side", "qty", "type", "filled_at", "status"]
                    show_cols = [c for c in show_cols if c in df_orders.columns]

                    # Søkefelt
                    search = st.text_input("Søk (ticker, status ...)", key="handler_search")
                    table = filled_orders if mode == "Kun utfylte handler" else df_orders
                    if search:
                        mask = table.apply(
                            lambda row: row.astype(str).str.contains(search, case=False, regex=False).any(), axis=1)
                        table = table[mask]

                    if not table.empty:
                        st.dataframe(table[show_cols], use_container_width=True)
                    else:
                        st.info("Fant ingen handler i dette utvalget.")
                else:
                    st.info("Fant ingen handler.")

            except Exception as e:
                st.error(f"Kunne ikke hente handler: {e}")
        else:
            st.warning("Handler er ikke tilgjengelig uten Alpaca-tilkobling.")

    # ---------- TAB 3: TEKNISK ANALYSE ----------

    with tab3:

        st.header("Teknisk Analyse")

        tickerlist = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY", "GOOGL", "AMZN", "NOK.OL"]
        symbol = st.selectbox("Velg ticker", tickerlist, index=0, key="ta_symbol_full")
        period = st.selectbox("Periode", ["6mo", "1y", "3mo", "3y"], index=1, key="ta_period_full")

        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if not df.empty:
            # Indikatorer
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()

            # Bollinger Bands
            df["BB_upper"] = df["MA20"] + 2 * df["Close"].rolling(20).std()
            df["BB_lower"] = df["MA20"] - 2 * df["Close"].rolling(20).std()

            # RSI
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            exp12 = df["Close"].ewm(span=12, adjust=False).mean()
            exp26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp12 - exp26
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

            # Støtte/motstand (lokal min/max siste 90 dager)
            lookback = 90
            local_min = df["Low"].rolling(window=lookback, center=True).min().iloc[-lookback:]
            local_max = df["High"].rolling(window=lookback, center=True).max().iloc[-lookback:]
            support = sorted(local_min.unique())[:2]
            resistance = sorted(local_max.unique())[-2:]

            # Fibonacci retracement (siste 180 dager)
            fib_df = df[-180:] if len(df) >= 180 else df.copy()
            fib_min = fib_df["Low"].min()
            fib_max = fib_df["High"].max()
            fib_levels = [fib_max - (fib_max - fib_min) * ratio for ratio in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]]

            # ---- Matplotlib + mplfinance CANDLESTICK ----
            df_mpf = df.copy()
            df_mpf.index.name = 'Date'
            apds = [
                mpf.make_addplot(df_mpf["MA20"], color="blue", width=1.1),
                mpf.make_addplot(df_mpf["MA50"], color="orange", width=1.1),
                mpf.make_addplot(df_mpf["MA200"], color="green", width=1.1),
                mpf.make_addplot(df_mpf["BB_upper"], color="#888", linestyle="dotted"),
                mpf.make_addplot(df_mpf["BB_lower"], color="#888", linestyle="dotted"),
            ]

            # Plott Fibonacci og støtte/motstand manuelt
            def custom_fibo(ax):
                for fl in fib_levels:
                    ax.axhline(fl, color="#2196f3", linestyle="dashdot", alpha=0.18)
                for i, fl in enumerate(fib_levels):
                    if i in [1, 2, 3, 4, 5]:
                        ax.text(df.index[-10], fl, f"{[0, 23.6, 38.2, 50, 61.8, 78.6, 100][i]}%", color="#1976d2",
                                alpha=0.7, fontsize=9)
                for lvl in support:
                    ax.axhline(lvl, color="#4caf50", linestyle="dotted", linewidth=1.2, alpha=0.8)
                for lvl in resistance:
                    ax.axhline(lvl, color="#f44336", linestyle="dotted", linewidth=1.2, alpha=0.8)

            st.subheader("Candlestick med MA/Bollinger/Fib/støtte/motstand")
            fig, axes = mpf.plot(
                df_mpf,
                type='candle',
                style='yahoo',
                addplot=apds,
                volume=True,
                returnfig=True,
                figscale=1.25,
                figratio=(18, 10),
                panel_ratios=(6, 2)
            )
            custom_fibo(axes[0])
            st.pyplot(fig)

            # ---- MACD og RSI egne grafer ----
            st.subheader("MACD og RSI")
            fig2, (ax_macd, ax_rsi) = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                                                   gridspec_kw={"height_ratios": [1, 1]})
            ax_macd.plot(df.index, df["MACD"], label="MACD", color="#2ca02c")
            ax_macd.plot(df.index, df["MACD_signal"], label="Signal", color="#ff9800", linestyle="--")
            ax_macd.bar(df.index, df["MACD_hist"], label="Histogram", color="#90caf9", alpha=0.6)
            ax_macd.set_title("MACD")
            ax_macd.legend()
            ax_macd.grid(alpha=0.22)

            ax_rsi.plot(df.index, df["RSI"], color="#607d8b", label="RSI (14)")
            ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.7)
            ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.7)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_title("RSI (14)")
            ax_rsi.legend()
            ax_rsi.grid(alpha=0.22)
            st.pyplot(fig2)

            # Data
            st.markdown(
                f"**Nivåer støtte:** {['%.2f' % lvl for lvl in support]} &nbsp;&nbsp;&nbsp; **Motstand:** {['%.2f' % lvl for lvl in resistance]}")
            st.markdown(f"**Fibonacci retracement-nivåer:** {[f'{x:.2f}' for x in fib_levels]}")
            st.dataframe(df[["Close", "MA20", "MA50", "MA200", "BB_upper", "BB_lower", "RSI", "MACD", "MACD_signal",
                             "Volume"]].tail(30))

        else:
            st.warning("Fant ingen data for valgt ticker.")

        with tab3:
            st.header("Aksjegraf – Candlestick")
            ticker = st.text_input("Ticker for graf (f.eks. DNB.OL, EQNR.OL)")
            if ticker:
                data = yf.Ticker(ticker).history(period="1mo")
                if not data.empty:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=data.index,
                            open=data['Open'], high=data['High'],
                            low=data['Low'], close=data['Close']
                        )
                    ])
                    fig.update_layout(
                        template="plotly_white",
                        xaxis_rangeslider_visible=False,
                        title=f"{ticker} - Candlestick siste måned"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Fant ingen data for valgt ticker.")

    # ---------- TAB 4: Backtest-utsnitt ----------
    def backtest_rsi_df(df, rsi_len=14, buy_th=30.0, sell_th=70.0,
                        initial_capital=10_000.0, use_wilder=True):
        """
        Backtester RSI-strategi på en DataFrame med 'Close'.
        Returnerer: df_bt, trades_df, metrics, buys, sells
        """
        df_bt = df.copy()

        # 1) Rydd duplikate kolonner (kan gi Series i stedet for skalar i row["RSI"])
        if df_bt.columns.duplicated().any():
            df_bt = df_bt.loc[:, ~df_bt.columns.duplicated()].copy()

        # 2) RSI
        delta = df_bt["Close"].diff()

        if use_wilder:
            # Wilder’s RSI (EMA-basert)
            gain = delta.clip(lower=0).ewm(alpha=1 / rsi_len, min_periods=rsi_len, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_len, min_periods=rsi_len, adjust=False).mean()
        else:
            # SMA-variant
            gain = delta.clip(lower=0).rolling(rsi_len, min_periods=rsi_len).mean()
            loss = (-delta.clip(upper=0)).rolling(rsi_len, min_periods=rsi_len).mean()

        # Beskytt mot deling på 0
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df_bt["RSI"] = rsi.astype(float)

        # 3) Signaler og posisjon (bruk skalar-floats i if-sjekker)
        df_bt["position"] = 0
        buys, sells = [], []
        in_pos = False

        # bruk arrays for ytelse, men trekk skalar for sammenligning
        rsi_vals = df_bt["RSI"].to_numpy(copy=False)
        close_vals = df_bt["Close"].to_numpy(copy=False)
        idx_list = df_bt.index.to_list()

        for i, idx in enumerate(idx_list):
            rsi_val = rsi_vals[i]
            # sørg for skalar float
            rsi_val = float(rsi_val) if not (pd.isna(rsi_val)) else np.nan

            if np.isnan(rsi_val):
                df_bt.at[idx, "position"] = int(in_pos)
                continue

            if (not in_pos) and (rsi_val < float(buy_th)):
                in_pos = True
                df_bt.at[idx, "position"] = 1
                buys.append((idx, float(close_vals[i])))
            elif in_pos and (rsi_val > float(sell_th)):
                in_pos = False
                df_bt.at[idx, "position"] = 0
                sells.append((idx, float(close_vals[i])))
            else:
                df_bt.at[idx, "position"] = int(in_pos)

        # 4) Avkastning og equity
        df_bt["ret"] = df_bt["Close"].pct_change().fillna(0.0)
        df_bt["strategy_ret"] = (df_bt["ret"] * df_bt["position"].shift(1).fillna(0.0)).astype(float)
        df_bt["equity"] = initial_capital * (1.0 + df_bt["strategy_ret"]).cumprod()

        # 5) Tradelogg (match buy->sell i rekkefølge)
        trades = []
        for i, (b_time, b_price) in enumerate(buys):
            if i < len(sells):
                s_time, s_price = sells[i]
            else:
                s_time, s_price = np.nan, np.nan
            pnl = 0.0 if np.isnan(s_price) else (float(s_price) - float(b_price))
            pnl_pct = 0.0 if np.isnan(s_price) else (float(s_price) / float(b_price) - 1.0)
            trades.append({
                "buy_time": b_time, "buy_price": float(b_price),
                "sell_time": s_time, "sell_price": (None if np.isnan(s_price) else float(s_price)),
                "pnl_$": pnl, "pnl_%": pnl_pct
            })
        trades_df = pd.DataFrame(trades)

        # 6) Nøkkeltall
        total_trades = int(len(trades_df))
        closed_mask = trades_df["sell_price"].notna()
        closed_trades = int(closed_mask.sum())
        win_rate = float((trades_df.loc[closed_mask, "pnl_$"] > 0).mean()) if closed_trades else 0.0
        total_pnl = float(trades_df.loc[closed_mask, "pnl_$"].sum()) if closed_trades else 0.0
        equity_end = float(df_bt["equity"].iloc[-1])
        sr_std = df_bt["strategy_ret"].std()
        daily_sharpe = float((df_bt["strategy_ret"].mean() / sr_std) * np.sqrt(252)) if sr_std and sr_std > 0 else 0.0
        roll_max = df_bt["equity"].cummax()
        drawdown = df_bt["equity"] / roll_max - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        metrics = {
            "total_trades": total_trades,
            "closed_trades": closed_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "equity_end": equity_end,
            "daily_sharpe": daily_sharpe,
            "max_drawdown": max_drawdown,
        }

        return df_bt, trades_df, metrics, buys, sells

    def cum_trade_pnl(trades_df: pd.DataFrame) -> pd.Series:
        """Kumulativ P&L for lukkede handler (sortert på sell_time)."""
        if trades_df.empty:
            return pd.Series(dtype=float)
        t = trades_df.copy()
        t = t[t["sell_price"].notna()].sort_values("sell_time")
        return t["pnl_$"].cumsum()

    # ---------- TAB 4: TRADINGBOT ----------

    import datetime as dt

    with tab4:
        st.header("Tradingbot – Automatisk handel med Alpaca")

        # Start/Stop
        if "bot_running" not in st.session_state:
            st.session_state["bot_running"] = False

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Tradingbot", key="bot_start"):
                st.session_state["bot_running"] = True
                st.success("Tradingbot startet!")
        with c2:
            if st.button("Stopp Tradingbot", key="bot_stop"):
                st.session_state["bot_running"] = False
                st.warning("Tradingbot stoppet.")

        st.markdown("**RSI-strategi: Kjøp under terskel, selg over terskel**")

        # Parametre
        left, right = st.columns(2)
        with left:
            symbols = st.multiselect("Velg tickere (multi-backtest)", ["AAPL", "TSLA", "NVDA", "MSFT"],
                                     default=["AAPL"])
            qty = st.number_input("Antall aksjer per trade (live)", min_value=1, value=1, key="bot_qty")
            start_date = st.date_input("Startdato", value=dt.date.today() - dt.timedelta(days=180))
        with right:
            end_date = st.date_input("Sluttdato", value=dt.date.today())
            rsi_len = st.number_input("RSI-lengde", min_value=2, value=14)
            buy_th = st.number_input("Kjøpsgrense (RSI < x)", min_value=1.0, max_value=99.0, value=30.0)
            sell_th = st.number_input("Salgsgrense (RSI > x)", min_value=1.0, max_value=99.0, value=70.0)

        use_wilder = st.checkbox("Bruk Wilder’s RSI (anbefalt)", value=True)
        mode = st.radio("Modus", ["Kun backtest (anbefalt)", "Live handel (Alpaca)"], index=0, horizontal=True)

        # Hent data og backtest for hver ticker
        results = {}
        for symbol in symbols:
            df = yf.download(symbol, start=start_date, end=end_date + dt.timedelta(days=1),
                             interval="1d", progress=False)
            if df.empty:
                st.warning(f"Ingen data for {symbol} i valgt periode.")
                continue

            df_bt, trades_df, metrics, buys, sells = backtest_rsi_df(
                df.copy(), rsi_len=rsi_len, buy_th=buy_th, sell_th=sell_th,
                initial_capital=10_000.0, use_wilder=use_wilder
            )
            results[symbol] = (df_bt, trades_df, metrics, buys, sells)

        if not results:
            st.stop()

        # Sammendragstabell for alle symbols
        summary_rows = []
        for sym, (_, _, m, _, _) in results.items():
            summary_rows.append({
                "Symbol": sym,
                "Handler (lukkede)": m["closed_trades"],
                "Win rate %": round(m["win_rate"] * 100, 1),
                "Sum P&L $": round(m["total_pnl"], 2),
                "Sluttkapital": round(m["equity_end"], 2),
                "Sharpe": round(m["daily_sharpe"], 2),
                "Max DD %": round(m["max_drawdown"] * 100, 1),
            })
        st.subheader("Oppsummering (multi-symbol)")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Plot: equity-kurver for alle
        st.subheader("Equity-kurver")
        fig_eq, ax_eq = plt.subplots(figsize=(10, 3.2))
        for sym, (df_bt, _, _, _, _) in results.items():
            ax_eq.plot(df_bt.index, df_bt["equity"], label=sym)
        ax_eq.set_title("Equity-kurver (backtest)")
        ax_eq.legend()
        st.pyplot(fig_eq)

        # Detaljvisning per symbol
        for symbol, (df_bt, trades_df, metrics, buys, sells) in results.items():
            st.markdown("---")
            st.markdown(f"### {symbol}")

            # Nøkkeltall
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Handler (lukkede)", f"{metrics['closed_trades']}/{metrics['total_trades']}")
            k2.metric("Win rate", f"{metrics['win_rate'] * 100:.1f}%")
            k3.metric("Sum P&L ($)", f"{metrics['total_pnl']:.2f}")
            k4.metric("Sluttkapital", f"{metrics['equity_end']:,.2f}")
            k5.metric("Sharpe", f"{metrics['daily_sharpe']:.2f}")
            k6.metric("Max DD", f"{metrics['max_drawdown'] * 100:.1f}%")

            # Pris + kjøp/selg-markører
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(df_bt.index, df_bt["Close"], label="Close")
            b_list = [(t, p) for t, p in buys]
            s_list = [(t, p) for t, p in sells]
            if b_list:
                ax1.scatter([t for t, _ in b_list], [p for _, p in b_list], marker="^", s=80, label="Kjøp", zorder=3)
            if s_list:
                ax1.scatter([t for t, _ in s_list], [p for _, p in s_list], marker="v", s=80, label="Selg", zorder=3)
            ax1.set_title(f"{symbol} – pris med signaler (RSI {rsi_len})")
            ax1.legend()
            st.pyplot(fig1)

            # RSI
            fig2, ax2 = plt.subplots(figsize=(10, 2.8))
            ax2.plot(df_bt.index, df_bt["RSI"], label=f"RSI({rsi_len}) – {'Wilder' if use_wilder else 'SMA'}")
            ax2.axhline(buy_th, linestyle="--")
            ax2.axhline(sell_th, linestyle="--")
            ax2.set_ylim(0, 100)
            ax2.set_title("RSI og terskler")
            ax2.legend()
            st.pyplot(fig2)

            # Equity
            fig3, ax3 = plt.subplots(figsize=(10, 3.2))
            ax3.plot(df_bt.index, df_bt["equity"], label="Equity")
            ax3.set_title("Equity-kurve")
            ax3.legend()
            st.pyplot(fig3)

            # Kumulativ P&L pr trade
            st.subheader("Kumulativ P&L per trade (lukkede)")
            cpnls = cum_trade_pnl(trades_df)
            if not cpnls.empty:
                fig4, ax4 = plt.subplots(figsize=(10, 3.0))
                ax4.step(range(1, len(cpnls) + 1), cpnls.values, where="post")
                ax4.set_xlabel("Trade # (lukkede)")
                ax4.set_ylabel("Kumulativ P&L ($)")
                ax4.set_title("Kumulativ P&L per trade")
                st.pyplot(fig4)
            else:
                st.info("Ingen lukkede handler ennå.")

            # Tradelogg
            with st.expander(f"Se tradeloggen ({symbol})"):
                st.dataframe(trades_df, use_container_width=True)

        # Live handel (valgfritt) – bruker siste symbol i listen som “aktivt”
        if symbols:
            last_symbol = symbols[-1]
            df_last = results[last_symbol][0]
            current_rsi = float(df_last["RSI"].iloc[-1])
            st.write(f"**Siste RSI ({last_symbol}):** {current_rsi:.2f}")
            if mode == "Live handel (Alpaca)":
                st.info("Live-modus: sender faktisk ordre via Alpaca når du trykker knappen under.")
                can_buy = current_rsi < buy_th
                can_sell = current_rsi > sell_th
                cL, cR = st.columns(2)
                with cL:
                    if st.button(f"KJØP {qty} {last_symbol} (RSI<{buy_th})", disabled=not can_buy, key="live_buy_btn"):
                        try:
                            order = trading_client.submit_order(
                                symbol=last_symbol, qty=qty, side="buy", type="market",
                                time_in_force="day", routing="smart"
                            )
                            st.success(f"Live kjøp sendt! Ordre-id: {getattr(order, 'id', '?')}")
                        except Exception as e:
                            st.error(f"Feil ved kjøp: {e}")
                with cR:
                    if st.button(f"SELG {qty} {last_symbol} (RSI>{sell_th})", disabled=not can_sell,
                                 key="live_sell_btn"):
                        try:
                            order = trading_client.submit_order(
                                symbol=last_symbol, qty=qty, side="sell", type="market",
                                time_in_force="day", routing="smart"
                            )
                            st.success(f"Live salg sendt! Ordre-id: {getattr(order, 'id', '?')}")
                        except Exception as e:
                            st.error(f"Feil ved salg: {e}")


    # ---------- TAB 5: ORDREBOK ----------
    from alpaca.trading.enums import OrderStatus

    with tab5:
        st.header("Ordrebok")
        if trading_client:
            # Oppdater-knapp
            if st.button("Oppdater ordrebok", key="refresh_orders"):
                st.rerun()

            try:
                # Hent alle ordrer (inkl. lukkede)
                orders = trading_client.get_orders()
                df_orders = pd.DataFrame([o.__dict__ for o in orders])

                # ---- Velg kolonner ----
                alle_cols = df_orders.columns.tolist()
                default_cols = ["id", "symbol", "side", "qty", "type", "limit_price", "stop_price", "trail_price",
                                "status", "submitted_at", "filled_at"]
                show_cols = st.multiselect(
                    "Velg hvilke kolonner som skal vises:",
                    options=alle_cols,
                    default=[col for col in default_cols if col in alle_cols],
                    key="order_show_cols"
                )

                # ---- Søk og filtrering ----
                search = st.text_input("Søk (ticker, side, status, id ...)", key="order_search")
                search_df = df_orders
                if search:
                    mask = search_df.apply(
                        lambda row: row.astype(str).str.contains(search, case=False, regex=False).any(), axis=1)
                    search_df = search_df[mask]

                # ---- Åpne ordrer ----
                open_orders = search_df[search_df["status"] == "new"] if "status" in search_df else pd.DataFrame()
                if not open_orders.empty:
                    st.subheader("Åpne ordrer")
                    st.dataframe(open_orders[show_cols], use_container_width=True)

                    # Kansellering
                    st.write("Kanseller en ordre:")
                    if "id" in open_orders:
                        ordre_id = st.selectbox("Velg ordre-id", open_orders["id"], key="cancel_order")
                        if st.button("Kanseller ordre", key="cancel_order_btn"):
                            try:
                                trading_client.cancel_order(ordre_id)
                                st.success(f"Ordre {ordre_id} kansellert!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Feil ved kansellering: {e}")
                else:
                    st.info("Ingen åpne ordrer.")

                # ---- Historiske ordrer ----
                closed_orders = search_df[search_df["status"] != "new"] if "status" in search_df else pd.DataFrame()
                if not closed_orders.empty:
                    st.subheader("Historiske ordrer")
                    st.dataframe(closed_orders[show_cols], use_container_width=True)

            except Exception as e:
                st.error(f"Kunne ikke hente ordrer: {e}")
        else:
            st.warning("Ordrebok er ikke tilgjengelig uten Alpaca-tilkobling.")

    # ---------- TAB 6: BACKTEST ----------
    with tab6:
        st.header("Backtest strategi")
        symbol = st.text_input("Ticker for backtest", value="AAPL", key="bt_symbol")
        period = st.selectbox("Periode", ["1y", "6mo", "3mo"], key="bt_period")
        strategi = st.selectbox("Strategi", ["MA-kryss (20/50)", "Kjøp og hold"], key="bt_strategy")

        if symbol:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            # --- Sjekk MultiIndex! ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            if not df.empty:
                df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()
                df["Signal"] = 0
                if strategi == "MA-kryss (20/50)":
                    df.loc[df["MA20"] > df["MA50"], "Signal"] = 1
                else:
                    df["Signal"] = 1
                df["Posisjon"] = df["Signal"].shift(1).fillna(0)
                df["Avkastning"] = df["Close"].pct_change().fillna(0)
                df["StrategiAvkastning"] = df["Avkastning"] * df["Posisjon"]
                df["Kumulativ"] = (1 + df["StrategiAvkastning"]).cumprod()
                df["KumulativBuyHold"] = (1 + df["Avkastning"]).cumprod()
                st.line_chart(df[["Kumulativ", "KumulativBuyHold"]])
                sharpe = np.nan
                if df["StrategiAvkastning"].std() > 0:
                    sharpe = (df["StrategiAvkastning"].mean() / df["StrategiAvkastning"].std()) * np.sqrt(252)
                st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
                st.write(df.tail(15))
            else:
                st.warning("Fant ingen data for valgt ticker.")

    # ---------- TAB 7: NYHETER ----------
    import requests
    from datetime import datetime, timedelta

    with tab7:
        st.header("Aksjenyheter")
        st.info("Velg selskap og datointervall for å vise relevante nyheter fra Finnhub.")

        tickerlist = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY", "GOOGL", "AMZN", "NOK.OL"]
        symbol = st.selectbox("Velg ticker for nyheter", tickerlist, index=0, key="news_symbol")

        # Dato-velger
        default_end = datetime.now().date()
        default_start = default_end - timedelta(days=14)
        from_date = st.date_input("Fra dato", default_start, key="nyheter_fra")
        to_date = st.date_input("Til dato", default_end, key="nyheter_til")
        limit = st.slider("Antall nyheter", 1, 20, 10, key="nyheter_limit")

        api_key = st.secrets.get("FINNHUB_API_KEY", None)
        if not api_key:
            st.error("FINNHUB_API_KEY mangler i secrets.toml!")
        else:
            # Finnhub: https://finnhub.io/docs/api/company-news
            def get_finnhub_news(symbol, api_key, from_date, to_date, limit=10):
                url = f"https://finnhub.io/api/v1/company-news"
                params = {
                    "symbol": symbol,
                    "from": from_date.strftime("%Y-%m-%d"),
                    "to": to_date.strftime("%Y-%m-%d"),
                    "token": api_key
                }
                resp = requests.get(url, params=params)
                if resp.status_code == 200:
                    news = resp.json()
                    # Sorter nyeste øverst
                    news = sorted(news, key=lambda x: x.get("datetime", 0), reverse=True)
                    return news[:limit]
                else:
                    st.error(f"Feil fra Finnhub: {resp.status_code}, {resp.text}")
                    return []

            news_items = get_finnhub_news(symbol, api_key, from_date, to_date, limit)
            if news_items:
                for n in news_items:
                    dt = datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"""
                    <div style="margin-bottom: 10px; border-bottom:1px solid #ddd;">
                        <b>{n.get("headline")}</b><br>
                        <span style="color: #666; font-size: 13px;">{dt} &nbsp;|&nbsp; {n.get("source")}</span><br>
                        {n.get("summary", '')[:150] + ("..." if n.get("summary") and len(n.get("summary")) > 150 else "")}<br>
                        <a href="{n.get("url")}" target="_blank">Les mer</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Ingen nyheter funnet for valgt selskap og periode.")

    # ---------- TAB 8: RAPPORTER ----------

    with tab8:
        st.header("Rapporter & Skatteklar PDF")

        # Sjekk/lag tabellen først
        conn = sqlite3.connect("orders.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS order_fills (
                order_id TEXT, 
                filled_at TEXT, 
                filled_price REAL, 
                venue TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Hent ALLE handler fra order_fills
        conn = sqlite3.connect("orders.db")
        fills_df = pd.read_sql_query("SELECT * FROM order_fills ORDER BY filled_at DESC", conn)
        conn.close()

        # 1. **Slett alle handler – vises alltid**
        if st.button("Slett alle handler (reset)", key="slett_handler"):
            conn = sqlite3.connect("orders.db")
            c = conn.cursor()
            c.execute("DELETE FROM order_fills")
            conn.commit()
            conn.close()
            st.success("Alle handler slettet! Last inn siden på nytt for å oppdatere tabellen.")

        # 2. **Hvis handler finnes: vis tabell og eksport**
        if not fills_df.empty:
            st.subheader("Alle handler (for rapportering)")
            st.dataframe(fills_df, use_container_width=True)

            # Lag PDF-rapport
            if st.button("Lag skatteklar PDF", key="pdf_knapp_handler"):
                pdf = fpdf.FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, "Rapport: Handler for Skatt", ln=1, align='C')
                pdf.ln(5)
                for i, row in fills_df.iterrows():
                    pdf.cell(0, 10,
                             f"ID: {row['order_id']} - Pris: {row['filled_price']} - Venue: {row['venue']} - Tid: {row['filled_at']}",
                             ln=1)
                pdf.output("handlerapport.pdf")
                st.success("PDF klar til nedlasting! Klikk knappen under:")

            # Nedlastingsknapp for PDF (egen key)
            if os.path.exists("handlerapport.pdf"):
                with open("handlerapport.pdf", "rb") as f:
                    st.download_button("Last ned rapport (PDF)", f, file_name="handlerapport.pdf",
                                       mime="application/pdf", key="pdf_download_handler")

            # CSV-eksport (egen key)
            csv = fills_df.to_csv(index=False).encode("utf-8")
            st.download_button("Last ned som CSV", data=csv, file_name="handlerapport.csv", mime="text/csv",
                               key="csv_download_handler")

            # Excel-eksport (egen key)
            excel_buffer = io.BytesIO()
            fills_df.to_excel(excel_buffer, index=False)
            st.download_button("Last ned som Excel", data=excel_buffer.getvalue(),
                               file_name="handlerapport.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="excel_download_handler")
        else:
            st.info("Ingen handler logget enda.")

            # Nedlastingsknapp for PDF selv om det ikke finnes handler (bruk egen key)
            if os.path.exists("handlerapport.pdf"):
                with open("handlerapport.pdf", "rb") as f:
                    st.download_button("Last ned rapport (PDF)", f, file_name="handlerapport.pdf",
                                       mime="application/pdf", key="pdf_download_handler_empty")

        # ---------- TAB 9: Rebalansering ----------

        from rebalansering import kjør_rebalansering

        with tab9:
            kjør_rebalansering()

        # ---------- TAB 10: KART OG GEOGRAFI ----------

    with tab11:


        st.header("Indekskart – live oppdatering")

        # --- INDEKSKART ---
        indekser = [
            {"navn": "S&P 500", "symbol": "^GSPC", "land": "USA", "lat": 38.9, "lon": -77.03, "flagg": "🇺🇸"},
            {"navn": "NASDAQ", "symbol": "^IXIC", "land": "USA", "lat": 37.3, "lon": -121.9, "flagg": "🇺🇸"},
            {"navn": "Dow Jones", "symbol": "^DJI", "land": "USA", "lat": 40.7, "lon": -74.0, "flagg": "🇺🇸"},
            {"navn": "FTSE 100", "symbol": "^FTSE", "land": "UK", "lat": 51.5, "lon": -0.12, "flagg": "🇬🇧"},
            {"navn": "DAX", "symbol": "^GDAXI", "land": "Tyskland", "lat": 52.52, "lon": 13.4, "flagg": "🇩🇪"},
            {"navn": "CAC 40", "symbol": "^FCHI", "land": "Frankrike", "lat": 48.85, "lon": 2.35, "flagg": "🇫🇷"},
            {"navn": "OMXSPI", "symbol": "^OMXSPI", "land": "Sverige", "lat": 59.3, "lon": 18.0, "flagg": "🇸🇪"},
            {"navn": "OSEBX", "symbol": "OSEBX.OL", "land": "Norge", "lat": 59.9, "lon": 10.7, "flagg": "🇳🇴"},
            {"navn": "Nikkei 225", "symbol": "^N225", "land": "Japan", "lat": 35.7, "lon": 139.7, "flagg": "🇯🇵"},
            {"navn": "Hang Seng", "symbol": "^HSI", "land": "Hong Kong", "lat": 22.3, "lon": 114.2, "flagg": "🇭🇰"},
            {"navn": "Shanghai", "symbol": "000001.SS", "land": "Kina", "lat": 31.2, "lon": 121.5, "flagg": "🇨🇳"},
            {"navn": "S&P/ASX 200", "symbol": "^AXJO", "land": "Australia", "lat": -33.9, "lon": 151.2, "flagg": "🇦🇺"},
            {"navn": "Bovespa", "symbol": "^BVSP", "land": "Brasil", "lat": -23.5, "lon": -46.6, "flagg": "🇧🇷"},
            {"navn": "S&P/TSX", "symbol": "^GSPTSE", "land": "Canada", "lat": 45.4, "lon": -75.7, "flagg": "🇨🇦"},
        ]

        symbols = [i["symbol"] for i in indekser]
        live_data = []
        for item in indekser:
            symbol = item["symbol"]
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.history(period="2d")
                if info.shape[0] > 1:
                    latest = info.iloc[-1]
                    prev = info.iloc[-2]
                    kurs = latest["Close"]
                    endring = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
                else:
                    kurs = info["Close"].iloc[-1]
                    endring = 0.0
            except Exception as e:
                kurs = float('nan')
                endring = float('nan')
            live_data.append({**item, "kurs": kurs, "endring": endring})

        df = pd.DataFrame(live_data)
        df["label"] = (
                df["flagg"] + " " + df["navn"] + "<br>" +
                df["kurs"].map(lambda x: f"{x:,.2f}") + "<br>" +
                df["endring"].map(lambda x: f"{x:+.2f}%")
        )
        df["farge"] = df["endring"].map(lambda x: "green" if x > 0 else "red")

        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            text="label",
            color="farge",
            hover_name="navn",
            size_max=30,
            projection="natural earth",
            template="plotly_dark",
        )

        fig.update_traces(marker=dict(size=18, line=dict(width=2, color="white")))
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="white",
                bgcolor="#a7c7e7",
                showcoastlines=False,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.header("Indeks, valuta & råvarer – live tabell")
        # --- INDEKS & RÅVARER TABELL ---
        assets = [
            # Indekser
            {"Navn": "OSEBX", "Symbol": "OSEBX.OL", "Land": "🇳🇴"},
            {"Navn": "OMX Stockholm PI", "Symbol": "^OMXSPI", "Land": "🇸🇪"},
            {"Navn": "OMX København PI", "Symbol": "OMXC20.CO", "Land": "🇩🇰"},
            {"Navn": "Nasdaq Composite", "Symbol": "^IXIC", "Land": "🇺🇸"},
            {"Navn": "Nasdaq 100", "Symbol": "^NDX", "Land": "🇺🇸"},
            {"Navn": "DJ Industrial Average", "Symbol": "^DJI", "Land": "🇺🇸"},
            {"Navn": "DAX (Tyskland)", "Symbol": "^GDAXI", "Land": "🇩🇪"},
            {"Navn": "CAC 40 (Frankrike)", "Symbol": "^FCHI", "Land": "🇫🇷"},
            {"Navn": "Euronext 100 (Europa)", "Symbol": "^N100", "Land": "🇪🇺"},
            {"Navn": "Nikkei 225", "Symbol": "^N225", "Land": "🇯🇵"},
            # Valuta
            {"Navn": "EUR/NOK", "Symbol": "EURNOK=X", "Land": "🇪🇺🇳🇴"},
            {"Navn": "USD/NOK", "Symbol": "USDNOK=X", "Land": "🇺🇸🇳🇴"},
            {"Navn": "GBP/NOK", "Symbol": "GBPNOK=X", "Land": "🇬🇧🇳🇴"},
            {"Navn": "SEK/NOK", "Symbol": "SEKNOK=X", "Land": "🇸🇪🇳🇴"},
            # Råvarer
            {"Navn": "Olje (Brent)", "Symbol": "BZ=F", "Land": "🛢️"},
            {"Navn": "Gull (spot)", "Symbol": "GC=F", "Land": "🥇"},
        ]

        rows = []
        for asset in assets:
            sym = asset["Symbol"]
            navn = asset["Navn"]
            land = asset["Land"]
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="2d")
                if len(hist) > 1:
                    ny = hist.iloc[-1]
                    forrige = hist.iloc[-2]
                    siste = ny["Close"]
                    endring_pct = ((ny["Close"] - forrige["Close"]) / forrige["Close"]) * 100
                    diff = ny["Close"] - forrige["Close"]
                    åpning = ny["Open"]
                    høy = ny["High"]
                    lav = ny["Low"]
                else:
                    siste = hist["Close"].iloc[-1]
                    endring_pct = 0
                    diff = 0
                    åpning = hist["Open"].iloc[-1]
                    høy = hist["High"].iloc[-1]
                    lav = hist["Low"].iloc[-1]
            except:
                siste = endring_pct = diff = åpning = høy = lav = None

            rows.append({
                "Navn": navn,
                "Land": land,
                "Siste": f"{siste:,.2f}" if siste is not None else "-",
                "I dag %": endring_pct / 100 if endring_pct is not None else None,
                "+/-": f"{diff:,.2f}" if diff is not None else "-",
                "Åpning": f"{åpning:,.2f}" if åpning is not None else "-",
                "Høy": f"{høy:,.2f}" if høy is not None else "-",
                "Lav": f"{lav:,.2f}" if lav is not None else "-",
            })

        df = pd.DataFrame(rows)

        def highlight_pct(val):
            if pd.isnull(val):
                return ""
            if val > 0:
                return "color: green"
            elif val < 0:
                return "color: red"
            else:
                return ""

        def highlight_diff(val):
            try:
                if float(val.replace(',', '')) > 0:
                    return "color: green"
                elif float(val.replace(',', '')) < 0:
                    return "color: red"
            except:
                return ""
            return ""

        styled_df = df.style.format({"I dag %": "{:.2%}"}).applymap(highlight_pct, subset=["I dag %"]).applymap(
            highlight_diff, subset=["+/-"])
        st.dataframe(styled_df, use_container_width=True, key="indeks_valuta_ravare")

        # Last ned som Excel
        st.markdown("---")
        excel = df.copy()
        excel["I dag %"] = excel["I dag %"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        st.download_button(
            "Last ned tabell som Excel",
            data=excel.to_csv(index=False, sep=";").encode("utf-8"),
            file_name="indekser_valuta_råvarer.csv",
            mime="text/csv",
            key="dl_indeks_valuta_ravare"
        )

        # --- Alpaca-aksjer: Live kurstabell med Yahoo-fallback ---
        st.header("Aksjer (Alpaca/Yahoo, live)")

        API_KEY = st.secrets["ALPACA_API_KEY"]
        API_SECRET = st.secrets["ALPACA_SECRET_KEY"]
        client = StockHistoricalDataClient(API_KEY, API_SECRET)

        obx_default = [
            "AKRBP.OL", "BWLG.OL", "DNB.OL", "EQNR.OL", "FRO.OL", "GJF.OL", "GOGL.OL", "HAFNI.OL",
            "HYARD.OL", "KOG.OL", "MOWI.OL", "MPCC.OL", "NHY.OL", "NOD.OL", "NAS.OL", "ORK.OL",
            "SALM.OL", "STB.OL", "SUBC.OL", "TEL.OL", "TOM.OL", "ULTI.OL", "VAR.OL", "YAR.OL", "PGS.OL"
        ]
        tickers = st.text_input(
            "Skriv inn tickere (kommaseparert, f.eks. AKRBP.OL, DNB.OL, EQNR.OL)",
            ", ".join(obx_default),
            key="alpaca_tickere"
        )
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

        latest = pd.DataFrame()
        if ticker_list:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=ticker_list,
                    timeframe=TimeFrame.Day,
                    start=datetime.now() - timedelta(days=5),
                    end=datetime.now()
                )
                bars = client.get_stock_bars(request)
                data = bars.df.reset_index()
                data["symbol"] = data["symbol"].str.upper()
                latest = data.groupby("symbol").last().reset_index()
                if latest.empty:
                    raise Exception("Alpaca returnerte ingen data – prøver Yahoo.")
            except Exception as e:
                st.info(f"Kunne ikke hente data fra Alpaca: {e} – prøver Yahoo Finance...")
                rows = []
                for ticker in ticker_list:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        hist = ticker_obj.history(period="5d")
                        if hist.empty:
                            continue
                        ny = hist.iloc[-1]
                        forrige = hist.iloc[-2] if len(hist) > 1 else ny
                        rows.append({
                            "symbol": ticker,
                            "close": ny["Close"],
                            "open": ny["Open"],
                            "high": ny["High"],
                            "low": ny["Low"],
                            "volume": ny["Volume"],
                            "timestamp": ny.name,
                        })
                    except Exception as err:
                        continue
                latest = pd.DataFrame(rows)

        if not latest.empty:
            latest["diff"] = latest["close"] - latest["open"]
            latest["pct"] = 100 * (latest["close"] - latest["open"]) / latest["open"]
            tabell = pd.DataFrame({
                "Navn": latest["symbol"],
                "Siste": latest["close"].round(2),
                "+/-": latest["diff"].round(2),
                "I dag %": latest["pct"].round(2),
                "Åpning": latest["open"].round(2),
                "Høy": latest["high"].round(2),
                "Lav": latest["low"].round(2),
                "Volum": latest["volume"].apply(lambda v: f"{int(v):,}"),
                "Tid": pd.to_datetime(latest["timestamp"]).dt.strftime("%H:%M"),
            })

            def highlight_pct(val):
                if pd.isnull(val):
                    return ""
                return "color: green" if val > 0 else "color: red" if val < 0 else ""

            def highlight_diff(val):
                try:
                    if float(val) > 0:
                        return "color: green"
                    elif float(val) < 0:
                        return "color: red"
                except:
                    return ""
                return ""

            styled = tabell.style.format({"I dag %": "{:.2f}%"}).applymap(
                highlight_pct, subset=["I dag %"]).applymap(
                highlight_diff, subset=["+/-"])
            st.dataframe(styled, use_container_width=True, key="aksjetabell_live")
        else:
            st.info("Ingen aksjer å vise. Skriv inn gyldige tickere for å se live kurser.")

        # --- LIVE ESG-søk via Finnhub ---
        import finnhub
        import plotly.graph_objects as go

        # --- ESG-søk, Finnhub/Yahoo fallback ---
        st.header("ESG-søk – Finnhub og Yahoo fallback")

        finnhub_api_key = st.secrets.get("FINNHUB_API_KEY", None)
        ticker = st.text_input("Skriv inn ticker (f.eks. AAPL, MSFT, TSLA):", value="AAPL").upper()

        esg = None
        esg_source = None

        # Prøv Finnhub først (hvis nøkkel finnes)
        if finnhub_api_key:
            finnhub_client = finnhub.Client(api_key=finnhub_api_key)
            try:
                esg = finnhub_client.company_esg_score(ticker)
                if esg and esg.get("esgScore") is not None:
                    esg_source = "Finnhub"
            except Exception as e:
                # Sjekk om det er 403-feil fra Finnhub
                if "403" in str(e):
                    st.info("Finnhub ESG krever premium – prøver Yahoo i stedet.")
                else:
                    st.error(f"Feil fra Finnhub: {e}")

        # Hvis Finnhub ikke funker – prøv Yahoo
        if not esg_source:
            t = yf.Ticker(ticker)
            sust = t.sustainability
            if sust is not None and not sust.empty:
                # Yahoo ESG-data er i DataFrame, vi henter ut tall
                esg = {}
                for row in sust.itertuples():
                    key = row[0].replace('Score', '').replace('score', '').capitalize()
                    value = row[1]
                    if key.lower() == "esg":
                        esg["esgScore"] = value
                    elif key == "Environment":
                        esg["environmentScore"] = value
                    elif key == "Social":
                        esg["socialScore"] = value
                    elif key == "Governance":
                        esg["governanceScore"] = value
                esg_source = "Yahoo"
            else:
                esg = None

        if ticker:
            if esg and esg.get("esgScore") is not None:
                st.success(f"ESG-score for {ticker} ({esg_source})")
                cols = st.columns(4)
                gauges = [
                    ("ESG", esg.get("esgScore")),
                    ("Environment", esg.get("environmentScore")),
                    ("Social", esg.get("socialScore")),
                    ("Governance", esg.get("governanceScore")),
                ]
                for i, (label, value) in enumerate(gauges):
                    with cols[i]:
                        if value is not None:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=value,
                                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "seagreen"}},
                                number={"valueformat": ".2f"},
                            ))
                            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=160)
                            st.plotly_chart(fig, use_container_width=True, key=f"esg_{i}")
                        st.caption(label)
                with st.expander(f"Rå ESG-data fra {esg_source}"):
                    st.write(esg)
            else:
                st.warning("Fant ingen ESG-data for denne tickeren hos Finnhub eller Yahoo.")

                # ---------- TAB 10: OM PLATTFORMEN ----------
                with tab10:
                    st.header("Om plattformen")
                    st.markdown("""
                        **Farmand & Morse Securities**  
                        Demo-plattform for trading, analyse og rapportering.  
                        Laget av Andreas Bolton Seielstad! 
                        Kontakt meg for tips, ros eller feilmeldinger.
                        """)
    import datetime as dt

    with tab13:
        st.subheader("📈 Derivater (Alpaca Options)")
        alp = AlpacaAdapter()

        # -------------------- Options chain --------------------
        with st.expander("➕ Hent options chain", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                symbol = st.text_input("Ticker", "AAPL", key="opt_symbol").upper().strip()
            with c2:
                # Valgfritt filter: slå på for å se dato-velgeren
                use_expiry_filter = st.checkbox("Filtrer på bestemt utløp", value=False, key="opt_chain_use_expiry")
                if use_expiry_filter:
                    expiry_date = st.date_input(
                        "Utløp",
                        value=dt.date.today() + dt.timedelta(days=30),
                        key="opt_chain_expiry_date"
                    )
                    expiry = expiry_date.strftime("%Y-%m-%d")
                else:
                    expiry_date = None
                    expiry = None  # ingen filtrering på utløpsdato

            if st.button("Hent kjede", key="btn_fetch_chain"):
                with st.spinner("Henter kontrakter…"):
                    try:
                        st.session_state["opt_chain_df"] = alp.get_options_chain(symbol, expiry)
                    except Exception as e:
                        st.error(f"Kunne ikke hente chain: {e}")

            df = st.session_state.get("opt_chain_df")
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("Tips: huk av «Filtrer på bestemt utløp» for å velge dato fra kalenderen.")

            # --- Velg kontrakt fra chain (sikrer gyldig symbol og unngår 422) ---
            selected_symbol = None
            sel_expiry = None
            sel_right = None
            sel_strike = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.markdown("**Velg kontrakt fra kjeden (anbefalt):**")
                expiries = sorted(df["expiry"].dropna().unique())
                sel_expiry = st.selectbox("Utløp (fra chain)", expiries, key="sel_expiry_chain")

                rights = ["CALL", "PUT"]
                sel_right = st.selectbox("Right (fra chain)", rights, key="sel_right_chain")

                df_f = df[(df["expiry"] == sel_expiry) & (df["right"] == sel_right)]
                strikes = sorted(df_f["strike"].dropna().unique())
                sel_strike = st.selectbox("Strike (fra chain)", strikes, key="sel_strike_chain")

                row = df_f[df_f["strike"] == sel_strike].head(1)
                if not row.empty:
                    selected_symbol = row.iloc[0]["symbol"]
                    st.caption(f"Valgt kontrakt fra Alpaca: `{selected_symbol}`")
                else:
                    st.warning("Fant ingen kontrakt for valgt kombinasjon.")

        st.markdown("---")

        colA, colB = st.columns(2)

        # -------------------- Ordrebillett (single-leg) --------------------
        with colA:
            st.subheader("Ordrebillett (single-leg)")
            right = st.selectbox("Right", ["CALL", "PUT"], key="order_right")
            strike = st.number_input("Strike", min_value=0.0, step=0.5,
                                     value=float(sel_strike) if sel_strike else 200.0, key="order_strike")

            # Kalender for ordre-utløp (påkrevd) – safe default selv om chain-filter er av
            default_order_date = (
                expiry_date if use_expiry_filter
                else (dt.datetime.strptime(sel_expiry,
                                           "%Y-%m-%d").date() if sel_expiry else dt.date.today() + dt.timedelta(
                    days=30))
            )
            order_expiry_date = st.date_input("Utløp", value=default_order_date, key="order_expiry_date")
            exp = order_expiry_date.strftime("%Y-%m-%d")  # streng til OCC/Alpaca

            qty = st.number_input("Antall kontrakter", min_value=1, step=1, value=1, key="order_qty")
            side = st.selectbox("Side", ["buy", "sell"], index=0, key="order_side")
            tif = st.selectbox("TIF", ["day", "gtc", "ioc", "fok"], index=0, key="order_tif")
            otype = st.selectbox("Ordertype", ["limit", "market"], index=0, key="order_type")
            limit_price = st.number_input("Limitpris", min_value=0.0, step=0.01, value=1.00,
                                          key="order_limit") if otype == "limit" else None

            # Bygg OCC som fallback (brukes bare hvis ingen chain-symbol er valgt)
            occ = occ_symbol(symbol, exp, right, strike)
            st.caption(f"OCC (fallback): {occ}")

            # --- Mini-risikomotor ---
            st.markdown("**Risikokontroller**")
            cRa, cRb, cRc = st.columns(3)
            with cRa:
                max_notional = st.number_input("Max notional", min_value=0.0, step=100.0, value=10_000.0,
                                               key="risk_max_notional")
            with cRb:
                max_contracts = st.number_input("Max kontrakter/ordre", min_value=1, step=1, value=10,
                                                key="risk_max_contracts")
            with cRc:
                strict_checks = st.checkbox("Streng validering", value=True, key="risk_strict")

            notional = (limit_price or 0.0) * qty * 100
            if side == "buy" and otype == "limit" and notional > max_notional:
                st.error("Over max notional – juster størrelse eller limit.")
            if qty > max_contracts:
                st.error("For mange kontrakter i én ordre.")

            # --- Send ordre ---
            if st.button("Send ordre", key="btn_send_order"):
                # Hvis bruker har valgt en kontrakt fra chain, bruk den – ellers bruk OCC
                symbol_to_trade = selected_symbol or occ
                if not symbol_to_trade:
                    st.error("Velg kontrakt fra kjeden eller fyll inn gyldig OCC-symbol.")
                elif strict_checks and (
                        (side == "buy" and otype == "limit" and notional > max_notional) or qty > max_contracts):
                    st.error("Ordre blokkert av risikokontroll.")
                else:
                    with st.spinner("Sender ordre…"):
                        resp = alp.place_option_order(symbol_to_trade, int(qty), side, otype, tif, limit_price)
                    if isinstance(resp, dict) and resp.get("error"):
                        st.error(f"Feil {resp['status_code']}: {resp['response']}")
                    else:
                        st.success("Ordre sendt!")
                        st.json(resp)

        # -------------------- Greker / Teoretisk pris --------------------
        with colB:
            st.subheader("Greker / Teo (Black–Scholes)")
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1:
                S = st.number_input("Spot", min_value=0.0, step=0.01, value=200.00, key="greek_spot")
            with g2:
                K = st.number_input("Strike K", min_value=0.0, step=0.5,
                                    value=float(sel_strike) if sel_strike else 200.0, key="greek_K")
            with g3:
                days = st.number_input("Dager", min_value=1, step=1, value=30, key="greek_days")
            with g4:
                iv = st.number_input("IV %", min_value=1.0, step=0.5, value=25.0, key="greek_iv") / 100
            with g5:
                r = st.number_input("Rente %", min_value=0.0, step=0.25, value=4.0, key="greek_r") / 100
            right2 = st.selectbox("Right (teo)", ["CALL", "PUT"], index=0, key="greek_right")

            if st.button("Beregn greker", key="btn_calc_greeks"):
                res = bs_price_greeks(OptionInput(S=S, K=K, T=days / 365, r=r, sigma=iv), right2)
                theta_day = res["theta"] / 365.0  # vis mer intuitivt per dag
                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.metric("Teo", f"{res['theo']:.2f}")
                m2.metric("Delta", f"{res['delta']:.3f}")
                m3.metric("Gamma", f"{res['gamma']:.4f}")
                m4.metric("Vega", f"{res['vega']:.2f}")
                m5.metric("Theta (år)", f"{res['theta']:.2f}")
                m6.metric("Theta/dag", f"{theta_day:.4f}")
                m7.metric("Rho", f"{res['rho']:.2f}")

        st.markdown("---")

        # -------------------- Posisjoner & Konto --------------------
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Posisjoner")
            try:
                pos = alp.get_positions()
                if not pos.empty:
                    st.dataframe(pos, use_container_width=True, hide_index=True, key="df_positions")
                else:
                    st.info("Ingen åpne posisjoner.")
            except Exception as e:
                st.error(f"Kunne ikke hente posisjoner: {e}")

        with c2:
            st.subheader("Konto")
            try:
                acct = alp.get_account()
                view = {k: acct.get(k) for k in [
                    "status", "buying_power", "portfolio_value", "cash", "equity", "last_equity", "pattern_day_trader"
                ]}
                st.json(view, expanded=False)
            except Exception as e:
                st.error(f"Kunne ikke hente konto: {e}")

if __name__ == "__main__":
    main()

