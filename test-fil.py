import streamlit as st
import os
from alpaca.trading.client import TradingClient

# ------------------ LOGO ------------------
def show_logo():
    if os.path.exists("Farmand1.png"):
        st.image("Farmand1.png", width=230)
    else:
        st.markdown("<h1 style='color:#1e375a;font-weight:900;'>Farmand & Morse Securities</h1>", unsafe_allow_html=True)

# ------------------ INNLOGGING ------------------
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

# ------------------ ORDREVINDU ------------------
def handle_order(trading_client):
    st.subheader("Handle aksjer")
    symbol = st.text_input("Ticker-symbol (f.eks. AAPL)", key="order_symbol")
    qty = st.number_input("Antall aksjer", min_value=1, value=1, step=1, key="order_qty")
    side = st.selectbox("Kjøp eller selg?", ["buy", "sell"], key="order_side")
    order_type = st.selectbox("Ordretype", ["market", "limit", "stop", "trailing_stop"], key="order_type")

    limit_price = stop_price = trail_price = trail_percent = None

    if order_type == "limit":
        limit_price = st.number_input("Limitpris", min_value=0.0, value=0.0, step=0.01, key="order_limit")
    elif order_type == "stop":
        stop_price = st.number_input("Stoppris", min_value=0.0, value=0.0, step=0.01, key="order_stop")
    elif order_type == "trailing_stop":
        use_trail_price = st.checkbox("Bruk trail-price (ellers prosent)", key="order_trail_type")
        if use_trail_price:
            trail_price = st.number_input("Trail amount (USD)", min_value=0.0, value=0.0, step=0.01, key="order_trail_price")
        else:
            trail_percent = st.number_input("Trail percent (%)", min_value=0.0, value=1.0, step=0.01, key="order_trail_percent")

    tif = st.selectbox("Time in force", ["gtc", "day", "opg", "cls", "ioc", "fok"], index=0, key="order_tif")

    if st.button("Send ordre", key="order_submit2"):
        try:
            order_args = dict(
                symbol=symbol.upper(),
                qty=int(qty),
                side=side,
                type=order_type,
                time_in_force=tif
            )
            if order_type == "limit" and limit_price:
                order_args['limit_price'] = float(limit_price)
            if order_type == "stop" and stop_price:
                order_args['stop_price'] = float(stop_price)
            if order_type == "trailing_stop":
                if trail_price and trail_price > 0:
                    order_args['trail_price'] = float(trail_price)
                elif trail_percent and trail_percent > 0:
                    order_args['trail_percent'] = float(trail_percent)
            order = trading_client.submit_order(**order_args)
            st.success(f"✅ Ordre sendt! ID: {order.id}")
        except Exception as e:
            st.error(f"❌ Feil ved ordre: {e}")

# ------------------ MAIN ------------------
def main():
    login()
    show_logo()

    # Hent Alpaca-nøkler og lag klient (kun etter login!)
    API_KEY = st.secrets["PKQXL7LF4EAFA6R9VRDS"]
    API_SECRET = st.secrets["vVnLT4aasb8wJgHPabAXpjxldudLpIdFahKemBDH"]
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Portefølje",
        "Handler",
        "Teknisk Analyse",
        "Tradingbot",
        "Ordrebok",
        "Backtest",
        "Nyheter",
        "Rapporter",
        "Rebalanser",
        "Om plattformen"
    ])

    with tab1:
        st.header("Porteføljeoversikt")
            account = get_account()
            if "portfolio_value" in account:
                st.metric("Porteføljeverdi", f"${float(account['portfolio_value']):,.2f}")
            if "cash" in account:
                st.metric("Ledig kontantbeholdning", f"${float(account['cash']):,.2f}")

            st.subheader("Aktive posisjoner")
            positions = get_positions()
            if positions and isinstance(positions, list) and len(positions) > 0:
                df_pos = pd.DataFrame(positions)
                # Tving til numerisk
                df_pos["market_value"] = pd.to_numeric(df_pos["market_value"], errors="coerce")
                df_pos["unrealized_pl"] = pd.to_numeric(df_pos["unrealized_pl"], errors="coerce")
                df_pos["qty"] = pd.to_numeric(df_pos["qty"], errors="coerce")

                # VIS TOTALT SUM:
                total_market_value = df_pos["market_value"].sum()
                st.markdown(f"### Totalt markedsverdi for porteføljen: **${total_market_value:,.2f}**")

                # Short-varsling og summering
                df_pos["⚠️"] = df_pos["side"].apply(lambda x: "🚨" if x == "short" else "")

                show_cols = ["symbol", "qty", "market_value", "unrealized_pl", "side", "⚠️"]

                # Summering nederst
                totals = {
                    "symbol": "SUM:",
                    "qty": df_pos["qty"].sum(),
                    "market_value": df_pos["market_value"].sum(),
                    "unrealized_pl": df_pos["unrealized_pl"].sum(),
                    "side": "",
                    "⚠️": ""
                }
                df_sum = pd.concat([df_pos[show_cols], pd.DataFrame([totals])], ignore_index=True)

                # Fargelegging
                def color_rows(row):
                    if row["side"] == "short":
                        return ['background-color: #fff3cd; color: #e67e22' for _ in row]
                    else:
                        return ['' for _ in row]

                def color_pl(val):
                    try:
                        v = float(val)
                        if v > 0:
                            return "background-color: #c8f7c5; color: green;"
                        elif v < 0:
                            return "background-color: #f7c5c5; color: red;"
                    except:
                        return ""
                    return ""

                styled_df = (
                    df_sum
                    .style
                    .apply(lambda x: color_rows(x), axis=1)
                    .applymap(color_pl, subset=["unrealized_pl"])
                    .format({"market_value": "{:,.2f}", "unrealized_pl": "{:,.2f}", "qty": "{:,.0f}"})
                )

                st.write("**Posisjoner med short-varsling og summering:**")
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("Ingen åpne posisjoner.")

        import streamlit as st

    with tab2:
        st.header("Handler / ordrehistorikk")
        # Lim inn handler/ordrebok-tab her

    with tab3:
        st.header("Teknisk Analyse")
        # Lim inn teknisk analyse-tab her

    with tab4:
        st.header("Tradingbot")
        # Lim inn tradingbot-tab her

    with tab5:
        st.header("Ordrebok")
        # Lim inn ordrebok-tab her

    with tab6:
        st.header("Backtest")
        # Lim inn backtest-tab her

    with tab7:
        st.header("Nyheter")
        # Lim inn nyhets-feed/tab her

    with tab8:
        st.header("Rapporter & Skatteklar")
        # Lim inn rapport-tab her

    with tab9:
        st.header("Rebalansering")
        # Lim inn rebalanseringsfunksjon/tab her

    with tab10:
        st.header("Om plattformen")
        # Lim inn info/om-app/tab her

    # Eksempel: vil du vise "Handle aksjer" kun i en tab?
    # with tab2:
    #     handle_order(trading_client)

if __name__ == "__main__":
    main()