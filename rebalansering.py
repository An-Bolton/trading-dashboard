import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- TAB 9: REBALANSERING ----------

def kjør_rebalansering():
    st.header("Rebalansering")
    st.info("Simuler rebalansering (velg allokering, beholdning og kontanter)")

    # ---------------- Parametre ----------------
    colA, colB = st.columns([2, 1])
    with colA:
        tickers_input = st.text_input("Tickere (kommaseparert)", "AAPL,MSFT,SPY", key="reb_tickers")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    with colB:
        lot_size = st.number_input("Lot-størrelse (aksjer)", min_value=1, value=1, step=1, key="reb_lot")

    # --- Målvekter ---
    st.subheader("Målvekter (%)")
    target = {}
    cols = st.columns(min(4, max(1, len(tickers))))
    for i, t in enumerate(tickers or ["—"]):
        if tickers:
            target[t] = cols[i % len(cols)].slider(
                f"{t} ønsket %", 0, 100, 100 // max(1, len(tickers)), 1, key=f"reb_w_{t}"
            )
    tot = sum(target.values()) if tickers else 0
    st.caption(f"Sum målvekter: **{tot}%**")

    ready_weights = bool(tickers) and (tot == 100)
    if not tickers:
        st.warning("Legg inn minst én ticker.")
    elif tot != 100:
        st.error("Total allokering må være 100%")

    # --- Nåværende beholdning & kontanter ---
    st.subheader("Nåværende portefølje")
    hold_cols = st.columns(4)
    current_shares = {t: 0 for t in tickers}
    if tickers:
        for i, t in enumerate(tickers):
            current_shares[t] = hold_cols[i % 4].number_input(
                f"{t} antall", min_value=0, value=0, step=1, key=f"reb_sh_{t}"
            )
    cash = st.number_input("Kontanter (USD)", min_value=0.0, value=0.0, step=100.0, key="reb_cash")
    fee_per_trade = st.number_input(
        "Estimert gebyr pr handel (USD)", min_value=0.0, value=0.0, step=0.5, key="reb_fee"
    )

    # ---------------- Priser ----------------
    st.subheader("Priser")
    if st.button("Hent siste close-priser", key="reb_fetch") and tickers:
        try:
            px = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"].ffill().iloc[-1]
            prices = {t: float(px[t]) for t in tickers} if isinstance(px, pd.Series) else {tickers[0]: float(px)}
            st.session_state["reb_prices"] = prices
            st.success("Priser oppdatert.")
        except Exception as e:
            st.error(f"Kunne ikke hente priser: {e}")

    prices = st.session_state.get("reb_prices", {t: np.nan for t in tickers})
    price_cols = st.columns(min(4, max(1, len(tickers))))
    for i, t in enumerate(tickers):
        prices[t] = price_cols[i % len(price_cols)].number_input(
            f"{t} pris ($)", min_value=0.0, value=float(prices.get(t, 0.0) or 0.0), step=0.01, key=f"reb_px_{t}"
        )

    # ---------------- Simulering ----------------
    st.subheader("Simulering")

    can_simulate = ready_weights and bool(tickers)
    if not can_simulate:
        st.info("Fyll inn tickere og sett vekter til 100 % for å simulere.")
        return

    # Porteføljeverdi nå
    mkt_val = {t: current_shares[t] * prices[t] for t in tickers}
    port_value_now = float(cash + sum(mkt_val.values()))
    if port_value_now <= 0:
        st.info("Sett inn priser, beholdning eller kontanter for å simulere.")
        return

    # Mål i USD og mål-aksjer
    target_usd = {t: (target[t] / 100.0) * port_value_now for t in tickers}
    desired_shares_raw = {t: (target_usd[t] / prices[t] if prices[t] > 0 else 0.0) for t in tickers}
    desired_shares = {t: int(np.round(desired_shares_raw[t] / lot_size) * lot_size) for t in tickers}

    # Foreslå handler
    delta_shares = {t: desired_shares[t] - int(current_shares[t]) for t in tickers}
    trade_value = {t: delta_shares[t] * prices[t] for t in tickers}

    # Kontantsjekk
    est_fees = sum(fee_per_trade for t in tickers if delta_shares[t] != 0)
    cash_after = cash - sum(v for v in trade_value.values() if v > 0) + \
                 sum(-v for v in trade_value.values() if v < 0) - est_fees

    # Skaler ned kjøp ved behov
    if cash_after < 0:
        need = -cash_after
        buys = sorted([t for t in tickers if delta_shares[t] > 0], key=lambda x: prices[x], reverse=True)
        for t in buys:
            while delta_shares[t] > 0 and need > 0:
                delta_shares[t] -= 1  # lot_size er 1 i de fleste aksjer; vil du, bytt til lot_size
                need -= prices[t] * 1
        trade_value = {t: delta_shares[t] * prices[t] for t in tickers}
        est_fees = sum(fee_per_trade for t in tickers if delta_shares[t] != 0)
        cash_after = cash - sum(v for v in trade_value.values() if v > 0) + \
                     sum(-v for v in trade_value.values() if v < 0) - est_fees

    # Ikke negative aksjer
    for t in tickers:
        if current_shares[t] + delta_shares[t] < 0:
            delta_shares[t] = -current_shares[t]

    # Recompute
    trade_value = {t: delta_shares[t] * prices[t] for t in tickers}
    est_fees = sum(fee_per_trade for t in tickers if delta_shares[t] != 0)
    cash_after = cash - sum(v for v in trade_value.values() if v > 0) + \
                 sum(-v for v in trade_value.values() if v < 0) - est_fees

    # Tabeller og nøkkeltall
    rows = []
    for t in tickers:
        act = "KJØP" if delta_shares[t] > 0 else ("SELG" if delta_shares[t] < 0 else "HOLD")
        rows.append({
            "Ticker": t, "Pris": round(prices[t], 4),
            "Nå (stk)": int(current_shares[t]), "Mål (stk)": int(desired_shares[t]),
            "Delta (stk)": int(delta_shares[t]), "Handling": act,
            "Handelsverdi ($)": round(trade_value[t], 2),
        })
    df_plan = pd.DataFrame(rows)

    new_positions = {t: int(current_shares[t] + delta_shares[t]) for t in tickers}
    new_mkt_val = {t: new_positions[t] * prices[t] for t in tickers}
    port_value_after = float(cash_after + sum(new_mkt_val.values()))
    new_weights = {t: (new_mkt_val[t] / port_value_after * 100.0 if port_value_after > 0 else 0.0)
                   for t in tickers}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Porteføljeverdi (før)", f"${port_value_now:,.2f}")
    c2.metric("Porteføljeverdi (etter)", f"${port_value_after:,.2f}")
    c3.metric("Gebyr-estimat", f"${est_fees:,.2f}")
    c4.metric("Kontant etter", f"${cash_after:,.2f}")

    st.subheader("Rebalanseringsforslag")
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    st.subheader("Nye vekter (etter forslag)")
    wrows = [{"Ticker": t, "Vekt %": round(new_weights[t], 2)} for t in tickers]
    st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)

    if cash_after < 0:
        st.error("Foreslåtte handler gir negative kontanter. Reduser kjøp eller øk salg/kontanter.")
    elif abs(sum(new_weights.values()) - 100.0) > 0.1:
        st.warning("Vektsum avviker litt fra 100 % pga. avrunding til hele aksjer.")
    else:
        st.success("Rebalanseringsforslaget er innenfor kontantgrense og lot-avrunding.")