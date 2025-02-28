from coinbase.rest import RESTClient as cb
import time
import logging
from random import choice
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Coinbase API credentials
COINBASE_API_KEY = 'organizations/fe2f5935-ba3c-47a1-a9c3-1b5d403f05f3/apiKeys/cd7d0d3f-7602-4e4c-8698-ab3284046bc0'
COINBASE_API_SECRET = '''-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIIzYQUX3LkcZ7/9XbWDAlhNz2YTsPjVBa6tyBvaN1WRaoAoGCCqGSM49
AwEHoUQDQgAE7GLB65kdtrmwhKB9lJ2/+I59Woogmv8lFdliGuiJ8VAAL/QTOdVD
PU7Jqeq2nAUYOz1puPkkwCBnC5G8ngiu0A==
-----END EC PRIVATE KEY-----'''
client = cb(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)

# Trading parameters
PRICE_RISE_THRESHOLD = 0.001  # 0.1% price increase to buy
TRAILING_STOP = 0.015        # Sell if price drops 1.5% from peak
MIN_SELL_PROFIT = 0.01       # Ensure at least 1% profit before selling
INTERVAL = 60                # Check every 60 seconds
TREND_WINDOW = 180           # 3-minute window for trend detection
LOW_WINDOW = 600             # 10-minute window for tracking lows
FALLBACK_TREND_THRESHOLD = 0.0003  # 0.03% fallback threshold for owned coins
FALLBACK_CYCLE_LIMIT = 5     # Fallback after 5 cycles (5 minutes) without trades
BOLLINGER_PERIOD = 20        # 20 periods (20 minutes) for Bollinger Bands
BOLLINGER_K = 2              # 2 standard deviations for Bollinger Bands

# State tracking
last_buy_prices = {}  # Tracks buy price per coin
peak_prices = {}      # Tracks peak price since buying per coin
low_prices = {}       # Tracks lowest price per coin in LOW_WINDOW
price_history = {}    # Tracks prices for trend detection
bollinger_history = {}  # Tracks prices for Bollinger Bands (longer window)
cycles_without_trend = 0  # Track cycles without a trend

# Meme quotes
MEME_QUOTES = {
    'buy': ["To the moon! HODL activated!", "Diamond hands, much wow!", "Bought the dip—such crypto, very gains!"],
    'sell': ["Profit secured! Lambos incoming!", "Sold the top—doge approves!", "Moon trip complete, wow!"]
}

def get_trending_coins(owned_coins):
    """Identify trending coins, new lows, or Bollinger Band oversold conditions."""
    candidates = [
        "BONK-USD", "PNUT-USD", "KAITO-USD", "TURBO-USD",  # Prioritized coins
        "MOODENG-USD", "DOGE-USD", "XRP-USD", "BTC-USD",
        "ETH-USD", "SOL-USD", "ADA-USD", "SHIB-USD", "PEPE-USD",
        "WIF-USD", "FARTCOIN-USD", "BRETT-USD",
        "SHELL-USD", "FLOKI-USD", "MOG-USD", "BOME-USD"
    ]
    trending = {}
    global cycles_without_trend
    
    try:
        products = client.get_products()['products']
        coinbase_pairs = {p['product_id'] for p in products if p['quote_currency_id'] == 'USD'}
    except Exception as e:
        logging.error(f"Coinbase products fetch failed: {e}")
        return trending, ""

    current_time = time.time()
    reason = ""
    for pair in candidates:
        if pair not in coinbase_pairs:
            continue
        try:
            ticker = client.get_product(product_id=pair)
            current_price = float(ticker['price'])
            
            # Log current price on every check
            logging.info(f"{pair}: Current = {current_price:.8f}")
            
            # Update price history for trend detection
            if pair not in price_history:
                price_history[pair] = []
            price_history[pair].append((current_time, current_price))
            price_history[pair] = [(t, p) for t, p in price_history[pair] if t > current_time - TREND_WINDOW]
            
            # Update low price history for new low detection
            if pair not in low_prices:
                low_prices[pair] = []
            low_prices[pair].append((current_time, current_price))
            low_prices[pair] = [(t, p) for t, p in low_prices[pair] if t > current_time - LOW_WINDOW]
            
            # Update Bollinger Band history (longer window)
            if pair not in bollinger_history:
                bollinger_history[pair] = []
            bollinger_history[pair].append((current_time, current_price))
            bollinger_history[pair] = [(t, p) for t, p in bollinger_history[pair] if t > current_time - (BOLLINGER_PERIOD * INTERVAL)]
            
            # Calculate trend (upward movement)
            trend_detected = False
            if len(price_history[pair]) > 1:
                old_price = price_history[pair][0][1]
                price_change = (current_price - old_price) / old_price
                logging.info(f"{pair}: Change = {price_change:.2%} over {len(price_history[pair])} checks")
                if price_change >= PRICE_RISE_THRESHOLD:
                    size = "0.001" if pair == "BTC-USD" else "0.01" if pair in ["ETH-USD", "SOL-USD"] else "10" if pair in ["DOGE-USD", "SHIB-USD"] else "1000"
                    trending[pair] = size
                    reason = f"trend: {price_change:.2%} increase in {TREND_WINDOW//60} min"
                    logging.info(f"{pair} trending: {price_change:.2%} increase in {TREND_WINDOW//60} min")
                    trend_detected = True
            
            # Check for new low (alternative buying condition)
            if not trend_detected and len(low_prices[pair]) > 1:
                lowest_price = min(p for t, p in low_prices[pair][:-1])  # Exclude current price
                if current_price < lowest_price:
                    size = "0.001" if pair == "BTC-USD" else "0.01" if pair in ["ETH-USD", "SOL-USD"] else "10" if pair in ["DOGE-USD", "SHIB-USD"] else "1000"
                    trending[pair] = size
                    reason = f"new low: {current_price:.8f} (previous low: {lowest_price:.8f}) in {(LOW_WINDOW//60)} min"
                    logging.info(f"{pair} hit new low: {current_price:.8f} (previous low: {lowest_price:.8f}) in {(LOW_WINDOW//60)} min")
                    trend_detected = True
            
            # Check Bollinger Bands (alternative buying condition: oversold)
            if not trend_detected and len(bollinger_history[pair]) >= BOLLINGER_PERIOD:
                prices = [p for t, p in bollinger_history[pair][-BOLLINGER_PERIOD:]]
                sma = sum(prices) / len(prices)
                std = np.std(prices)
                upper_band = sma + (BOLLINGER_K * std)
                lower_band = sma - (BOLLINGER_K * std)
                logging.info(f"{pair}: SMA = {sma:.8f}, Lower Band = {lower_band:.8f}, Upper Band = {upper_band:.8f}")
                if current_price <= lower_band:
                    size = "0.001" if pair == "BTC-USD" else "0.01" if pair in ["ETH-USD", "SOL-USD"] else "10" if pair in ["DOGE-USD", "SHIB-USD"] else "1000"
                    trending[pair] = size
                    reason = f"Bollinger Band oversold: {current_price:.8f} <= {lower_band:.8f}"
                    logging.info(f"{pair} hit lower Bollinger Band (oversold): {current_price:.8f} <= {lower_band:.8f}")
        except Exception as e:
            logging.error(f"{pair}: Price fetch failed: {e}")
    
    # Fallback: If no trends, lows, or Bollinger signals after FALLBACK_CYCLE_LIMIT cycles, trade owned coins with smaller upward movement
    if not trending:
        cycles_without_trend += 1
        if cycles_without_trend >= FALLBACK_CYCLE_LIMIT:
            logging.info(f"No trends, new lows, or Bollinger signals detected after {FALLBACK_CYCLE_LIMIT} cycles, checking owned coins for smaller movements.")
            for pair in candidates:
                if pair not in coinbase_pairs:
                    continue
                if pair.split('-')[0] not in owned_coins:
                    continue
                try:
                    if len(price_history.get(pair, [])) > 1:
                        old_price = price_history[pair][0][1]
                        current_price = price_history[pair][-1][1]
                        price_change = (current_price - old_price) / old_price
                        if price_change >= FALLBACK_TREND_THRESHOLD:
                            size = "0.001" if pair == "BTC-USD" else "0.01" if pair in ["ETH-USD", "SOL-USD"] else "10" if pair in ["DOGE-USD", "SHIB-USD"] else "1000"
                            trending[pair] = size
                            reason = f"fallback: {price_change:.2%} increase in {TREND_WINDOW//60} min"
                            logging.info(f"Fallback: {pair} trending: {price_change:.2%} increase in {TREND_WINDOW//60} min")
                except Exception as e:
                    logging.error(f"Fallback check for {pair} failed: {e}")
            cycles_without_trend = 0  # Reset after fallback
    else:
        cycles_without_trend = 0  # Reset if a trend, low, or Bollinger signal is found

    if not trending:
        logging.warning("No trending coins, new lows, or Bollinger signals detected.")
    else:
        logging.info("Trending pairs: " + ", ".join(trending.keys()))
    return trending, reason

def get_price_history(pair):
    try:
        candles = client.get_market_trades(pair, limit=5)['trades']
        prices = [float(trade['price']) for trade in candles]
        return sum(prices) / len(prices) if prices else None
    except Exception as e:
        logging.error(f"{pair}: Price history failed: {e}")
        return None

def place_order(pair, side, price, size, reason):
    try:
        # Generate a unique client_order_id using timestamp and pair
        client_order_id = f"{pair}-{side}-{int(time.time())}"
        # Use base_size instead of size for market_order
        order = client.market_order(
            product_id=pair,
            side=side,  # Already uppercase (BUY or SELL)
            base_size=size,  # Quantity of the base currency (e.g., 1000 KAITO)
            client_order_id=client_order_id
        )
        quote = choice(MEME_QUOTES[side])
        logging.info(f"{pair}: {side} at {price:.8f} ({reason}) - {quote}")
    except Exception as e:
        logging.error(f"{pair}: Order failed: {e}")

def check_funds():
    try:
        accounts = client.get_accounts()['accounts']
        # Log all accounts using available_balance['value']
        account_balances = []
        owned_coins = []
        for acc in accounts:
            balance = "N/A"
            if hasattr(acc, 'available_balance') and isinstance(acc.available_balance, dict) and 'value' in acc.available_balance:
                balance = acc.available_balance['value']
                if float(balance) > 0:  # Track owned coins with non-zero balance
                    owned_coins.append(acc.currency)
            account_balances.append(f"{acc.currency}: {balance}")
        logging.info("All accounts: " + ", ".join(account_balances))
        
        # Check USD balance
        usd_variants = ['USD', 'CBI spot USD', 'CBI_SPOT_USD']
        usd_account = next((acc for acc in accounts if acc.currency in usd_variants), None)
        if usd_account:
            # Log the full USD account object to inspect its attributes
            usd_attrs = vars(usd_account)
            logging.info(f"USD account details: {usd_attrs}")
            # Access available_balance.value
            if hasattr(usd_account, 'available_balance') and isinstance(usd_account.available_balance, dict) and 'value' in usd_account.available_balance:
                usd_balance = float(usd_account.available_balance['value'])
                logging.info(f"USD balance: ${usd_balance:.2f}")
                return usd_balance, owned_coins
            else:
                logging.warning("USD wallet found but no accessible balance field.")
                return 0, owned_coins
        else:
            logging.warning("No USD wallet found in Coinbase account.")
            return 0, owned_coins
    except Exception as e:
        logging.error(f"Funds check failed: {e}")
        return 0, []

def update_trading_pairs(owned_coins):
    global PAIRS, last_buy_prices
    new_pairs, reason = get_trending_coins(owned_coins)
    PAIRS = new_pairs
    last_buy_prices = {pair: last_buy_prices.get(pair) for pair in PAIRS if pair in last_buy_prices}
    return reason

def main():
    logging.info("Coinbase Meme Bot: Riding Price Vibes, To the Moon! 🚀")
    last_trend_update = 0
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_trend_update >= TREND_WINDOW:
                usd_balance, owned_coins = check_funds()
                if usd_balance < 1:
                    logging.warning("Low USD balance! Pausing trades...")
                    time.sleep(INTERVAL)
                    continue
                reason = update_trading_pairs(owned_coins)
                last_trend_update = current_time

            for pair, size in PAIRS.items():
                try:
                    ticker = client.get_product(product_id=pair)
                    current_price = float(ticker['price'])
                except Exception as e:
                    logging.error(f"{pair}: Price fetch failed: {e}")
                    continue

                moving_avg = get_price_history(pair)
                if moving_avg is None:
                    continue
                logging.info(f"{pair}: Price = {current_price:.8f}, 5-min Avg = {moving_avg:.8f}")

                if last_buy_prices.get(pair) is None and current_price >= moving_avg:
                    place_order(pair, 'BUY', current_price, size, reason)
                    last_buy_prices[pair] = current_price
                    peak_prices[pair] = current_price  # Initialize peak at buy price
                    # Reset low price history after buying to avoid immediate re-buy
                    if pair in low_prices:
                        del low_prices[pair]
                elif last_buy_prices.get(pair) is not None:
                    # Update peak price
                    peak_prices[pair] = max(peak_prices.get(pair, current_price), current_price)
                    # Trailing stop: sell if price drops TRAILING_STOP % from peak
                    sell_trigger = peak_prices[pair] * (1 - TRAILING_STOP)
                    min_sell_price = last_buy_prices[pair] * (1 + MIN_SELL_PROFIT)
                    if current_price <= sell_trigger and current_price >= min_sell_price:
                        place_order(pair, 'SELL', current_price, size, "trailing stop")
                        last_buy_prices[pair] = None
                        peak_prices[pair] = None  # Reset peak after selling

            time.sleep(INTERVAL)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()