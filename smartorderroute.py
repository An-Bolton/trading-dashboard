import pandas as pd

markets = [
    {"name": "A", "available": 300, "price": 100.0},
    {"name": "B", "available": 500, "price": 100.5},
    {"name": "C", "available": 700, "price": 101.0}
]

def smart_order_routing(markets, total_volume):
    sorted_markets = sorted(markets, key=lambda x: x["price"])
    order_distribution = []
    remaining = total_volume

    for market in sorted_markets:
        if remaining <= 0:
            break
        volume_to_buy = min(market["available"], remaining)
        order_distribution.append({
            "market": market["name"],
            "price": market["price"],
            "volume": volume_to_buy,
            "cost": volume_to_buy * market["price"]
        })
        remaining -= volume_to_buy

    return order_distribution

# Kjør algoritmen
order = smart_order_routing(markets, total_volume=1000)

# Oppsummering
total_cost = sum(o["cost"] for o in order)
total_volume = sum(o["volume"] for o in order)
average_price = total_cost / total_volume

# Print resultat
df = pd.DataFrame(order)
print(df)
print("\nTotal volum:", total_volume)
print("Total kostnad:", total_cost)
print("Gjennomsnittspris:", round(average_price, 2))