# smart_order_router.py med intervallbasert live-trading, Finnhub, strategi, Slack-varsling, e-post, trailing stop-loss og SQLite-logg

from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import time
import random
import requests
import datetime
import sqlite3
import smtplib
from email.message import EmailMessage

FINNHUB_API_KEY = "d12nqahr01qmhi3jqje0d12nqahr01qmhi3jqjeg"
SLACK_WEBHOOK_URL = None
INTERVAL_SECONDS = 60
TRAILING_STOP_PERCENT = 5  # Eks: 5 % ned fra topp
EMAIL_ALERTS = False
EMAIL_SENDER = "youremail@example.com"
EMAIL_PASSWORD = "yourpassword"
EMAIL_RECEIVER = "receiver@example.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

DB_PATH = "orders.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            volume INTEGER,
            avg_price REAL,
            total_cost REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_database(symbol: str, side: str, volume: int, avg_price: float, total_cost: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO orders (timestamp, symbol, side, volume, avg_price, total_cost)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.now(), symbol, side, volume, avg_price, total_cost))
    conn.commit()
    conn.close()

def send_email(subject: str, body: str):
    if not EMAIL_ALERTS:
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"E-postfeil: {e}")

def get_realtime_price(symbol: str, token: str = FINNHUB_API_KEY) -> float:
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={token}"
    r = requests.get(url)
    data = r.json()
    return data.get("c", 0.0)

def get_realtime_stock_orderbook(symbol="AAPL", side="buy", token=FINNHUB_API_KEY) -> List[Dict]:
    price = get_realtime_price(symbol, token)
    price_levels = [round(price + i * (0.1 if side == "sell" else -0.1), 2) for i in range(5)]
    volumes = [random.randint(100, 500) for _ in range(5)]
    return [{
        "name": f"{symbol}_{side}_lvl_{i+1}",
        "available": volumes[i],
        "price": price_levels[i],
        "fee": 0.001
    } for i in range(5)]

def passes_strategy_filter(symbol: str, token: str = FINNHUB_API_KEY) -> bool:
    url = f"https://finnhub.io/api/v1/indicator?symbol={symbol}&resolution=D&indicator=sma&timeperiod=10&token={token}"
    data = requests.get(url).json()
    sma = data.get("sma", {}).get("values", [-1])[-1]
    current = get_realtime_price(symbol, token)
    return current < sma if sma > 0 else True

def smart_order_routing(markets: List[Dict], total_volume: int, stop_price: float = None, side="buy") -> List[Dict]:
    for market in markets:
        market["effective_price"] = market["price"] * (1 + market["fee"])
    sorted_markets = sorted(markets, key=lambda x: x["effective_price"], reverse=(side == "sell"))
    order_distribution = []
    remaining = total_volume
    for market in sorted_markets:
        if stop_price:
            if side == "buy" and market["price"] > stop_price:
                continue
            if side == "sell" and market["price"] < stop_price:
                continue
        if remaining <= 0:
            break
        volume_to_trade = min(market["available"], remaining)
        order_distribution.append({
            "market": market["name"],
            "price": market["price"],
            "fee": market["fee"],
            "effective_price": market["effective_price"],
            "volume": volume_to_trade,
            "cost": volume_to_trade * market["effective_price"]
        })
        remaining -= volume_to_trade
    return order_distribution

def send_slack_notification(message: str):
    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        except:
            print("Feil ved sending til Slack.")

def track_trailing_stop(symbol: str, top_price: float, threshold_percent: float = TRAILING_STOP_PERCENT) -> bool:
    current = get_realtime_price(symbol)
    if current > top_price:
        top_price = current
    drawdown = (top_price - current) / top_price * 100
    print(f"🔻 {symbol} nå: {current:.2f} | topp: {top_price:.2f} | drawdown: {drawdown:.2f}%")
    return drawdown >= threshold_percent

def plot_order_distribution(df: pd.DataFrame, side: str):
    plt.figure(figsize=(8, 5))
    plt.bar(df['market'], df['volume'], color='green' if side == "buy" else 'red')
    plt.xlabel("Markedsplass")
    plt.ylabel("Antall")
    plt.title(f"{'Kjøps' if side == 'buy' else 'Salgs'}ordrefordeling")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_order_to_csv(df: pd.DataFrame, filename: str):
    os.makedirs("logs", exist_ok=True)
    df.to_csv(os.path.join("logs", filename), index=False)

def handle_symbol(symbol: str, side="buy", volume=1000, stop_price=None):
    print(f"\n🔎 {datetime.datetime.now().strftime('%H:%M:%S')} - Behandler {symbol}...")
    if not passes_strategy_filter(symbol):
        print(f"⛔ {symbol} filtrert bort av strategi.")
        return
    markets = get_realtime_stock_orderbook(symbol, side)
    order = smart_order_routing(markets, volume, stop_price, side)
    df = pd.DataFrame(order)
    total_cost = df['cost'].sum()
    avg_price = total_cost / df['volume'].sum()
    print(df.to_string(index=False))
    print(f"→ Total: {total_cost:.2f} USD | Snittpris: {avg_price:.2f} USD")
    save_order_to_csv(df, f"order_{symbol}_{side}_{datetime.date.today()}.csv")
    send_slack_notification(f"✅ Ordre for {symbol}: {volume} {side.upper()} @ {avg_price:.2f} USD")
    send_email(f"Ordre for {symbol}", f"{volume} {side.upper()} @ {avg_price:.2f} USD")
    log_to_database(symbol, side, volume, avg_price, total_cost)
    plot_order_distribution(df, side)

def main():
    init_database()
    watchlist = [
        "AAPL",  # Apple – NASDAQ
        "MSFT",  # Microsoft – NASDAQ
        "TSLA",  # Tesla – NASDAQ
        "GOOGL",  # Alphabet – NASDAQ
        "AMZN",  # Amazon – NASDAQ
        "NVDA",  # Nvidia – NASDAQ
        "SPY",  # ETF for S&P 500 – NYSE Arca
        "QQQ",  # Nasdaq-100 ETF – NASDAQ
        "BABA",  # Alibaba – NYSE
        "SHOP",  # Shopify – NYSE
        "NIO",  # NIO – NYSE
    ]

    print("▶️ Starter kontinuerlig overvåking...")
    top_prices = {symbol: 0 for symbol in watchlist}
    try:
        while True:
            for symbol in watchlist:
                current_price = get_realtime_price(symbol)
                if track_trailing_stop(symbol, top_prices[symbol]):
                    handle_symbol(symbol, side="sell", volume=1000)
                else:
                    top_prices[symbol] = max(top_prices[symbol], current_price)
                    handle_symbol(symbol, side="buy", volume=1000)
            print(f"⏳ Venter {INTERVAL_SECONDS} sekunder...")
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("⏹️ Avbrutt av bruker.")

if __name__ == "__main__":
    main()
