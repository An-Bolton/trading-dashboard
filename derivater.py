import math, requests, pandas as pd, streamlit as st
from dataclasses import dataclass

# ----- OCC-symbolbygger (AAPL250920C00200000) -----
def occ_symbol(underlying: str, expiry_yyyy_mm_dd: str, right: str, strike: float) -> str:
    y, m, d = expiry_yyyy_mm_dd.split("-")
    yy, mm, dd = y[2:], f"{int(m):02d}", f"{int(d):02d}"
    cp = "C" if right.upper().startswith("C") else "P"
    strike_int = int(round(float(strike) * 1000))
    return f"{underlying.upper()}{yy}{mm}{dd}{cp}{strike_int:08d}"

# ----- Minimal Alpaca adapter for opsjoner -----
class AlpacaAdapter:
    def __init__(self):
        self.api_key = str(st.secrets["ALPACA_API_KEY"]).strip()
        self.api_secret = str(st.secrets["ALPACA_SECRET_KEY"]).strip()
        self.trading_base_url = str(st.secrets.get("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets")).strip().rstrip("/")
        self.data_base_url    = str(st.secrets.get("ALPACA_DATA_URL",    "https://data.alpaca.markets")).strip().rstrip("/")
        self._hdr = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def ping_account(self):
        import requests
        url = f"{self.trading_base_url}/v2/account"
        r = requests.get(url, headers=self._hdr, timeout=30)
        return r.status_code, r.text

    def ping_options_contracts(self, symbol="AAPL"):
        import requests
        url = f"{self.data_base_url}/v2/options/contracts"
        params = {"underlying_symbols": symbol, "status": "active", "limit": 1000}
        r = requests.get(url, headers=self._hdr, params=params, timeout=30)
        return r.status_code, r.url, r.text

    def get_options_chain(self, underlying: str, expiry: str | None = None) -> pd.DataFrame:
        url = f"{self.trading_base_url}/v2/options/contracts"  # <— byttet fra data_base_url
        params = {"underlying_symbols": underlying.upper(), "status": "active", "limit": 1000}
        if expiry:
            params["expiration_date"] = expiry
        out, page = [], None
        while True:
            if page:
                params["page_token"] = page
            r = requests.get(url, headers=self._hdr, params=params, timeout=30)
            if r.status_code == 404:
                raise RuntimeError(f"Ingen opsjonsdata funnet (404) for {underlying}. URL: {r.url}")
            r.raise_for_status()
            data = r.json()
            items = data.get("data", []) or data.get("contracts", [])
            out.extend([{
                "symbol": c.get("symbol"),
                "underlying": c.get("underlying_symbol") or underlying.upper(),
                "expiry": c.get("expiration_date"),
                "right": (c.get("type") or "").upper(),
                "strike": c.get("strike_price"),
                "multiplier": c.get("multiplier", 100),
                "style": c.get("style", ""),
                "status": c.get("status", ""),
            } for c in items])
            page = data.get("next_page_token")
            if not page:
                break
        df = pd.DataFrame(out)
        return df.sort_values(["expiry", "strike", "right"]).reset_index(drop=True) if not df.empty else df

    def place_option_order(self, option_symbol: str, qty: int, side: str, order_type: str,
                           time_in_force: str = "day", limit_price: float | None = None):
        url = f"{self.trading_base_url}/v2/orders"
        payload = {
            "symbol": option_symbol,
            "qty": qty,
            "side": side.lower(),           # "buy"/"sell"
            "type": order_type.lower(),     # "market"/"limit"
            "time_in_force": time_in_force  # "day","gtc","ioc","fok"
        }
        if order_type.lower() == "limit":
            if limit_price is None: raise ValueError("limit_price kreves for limit-ordre")
            payload["limit_price"] = str(limit_price)
        r = requests.post(url, headers=self._hdr, json=payload, timeout=30)
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"error": True, "status_code": r.status_code, "response": r.text}

    def get_positions(self) -> pd.DataFrame:
        url = f"{self.trading_base_url}/v2/positions"
        r = requests.get(url, headers=self._hdr, timeout=30); r.raise_for_status()
        df = pd.DataFrame(r.json())
        keep = ["symbol","qty","side","avg_entry_price","market_value","unrealized_pl","unrealized_plpc"]
        return df[[c for c in keep if c in df.columns]] if not df.empty else df

    def get_account(self) -> dict:
        url = f"{self.trading_base_url}/v2/account"
        r = requests.get(url, headers=self._hdr, timeout=30); r.raise_for_status()
        return r.json()

# ----- Black–Scholes for greker/teo (frivillig sanity check) -----
@dataclass
class OptionInput:
    S: float; K: float; T: float; r: float; sigma: float; q: float = 0.0
def _N(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def _n(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
def bs_price_greeks(o: OptionInput, right: str):
    if o.T<=0 or o.sigma<=0 or o.S<=0 or o.K<=0:
        return {k: float("nan") for k in ["theo","delta","gamma","vega","theta","rho"]}
    d1 = (math.log(o.S/o.K)+(o.r-o.q+0.5*o.sigma**2)*o.T)/(o.sigma*math.sqrt(o.T))
    d2 = d1 - o.sigma*math.sqrt(o.T); disc_r = math.exp(-o.r*o.T); disc_q = math.exp(-o.q*o.T)
    if right.upper()=="CALL":
        price = disc_q*o.S*_N(d1) - disc_r*o.K*_N(d2); delta = disc_q*_N(d1); rho = o.K*o.T*disc_r*_N(d2)/100
    else:
        price = disc_r*o.K*_N(-d2) - disc_q*o.S*_N(-d1); delta = -disc_q*_N(-d1); rho = -o.K*o.T*disc_r*_N(-d2)/100
    gamma = disc_q*_n(d1)/(o.S*o.sigma*math.sqrt(o.T)); vega = o.S*disc_q*_n(d1)*math.sqrt(o.T)/100
    theta = -(o.S*disc_q*_n(d1)*o.sigma)/(2*math.sqrt(o.T))
    return {"theo":price,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}
