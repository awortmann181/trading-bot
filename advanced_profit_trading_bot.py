from coinbase_advanced_api import RESTClient as cb
import time
import logging
import logging.handlers
import json
import numpy as np
import os
from flask import Flask
import threading
from datetime import datetime
import sqlite3
import backoff
from random import choice
import requests  # For fetching remote config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
audit_handler = logging.handlers.RotatingFileHandler('audit.log', maxBytes=1000000, backupCount=5)
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)
audit_logger.addHandler(audit_handler)

# Load local configuration
CONFIG_FILE = 'trading_config.json'
COOLDOWN_PERIOD = 10
COOLDOWN_AFTER_SELL = 30
REMOTE_CONFIG_URL = "https://raw.githubusercontent.com/awortmann181/trading-bot-config/main/trading_config.json"
DEFAULT_CONFIG = {
    "min_sell_profit": 0.01,
    "btc_sell_profit": 0.01,
    "stop_loss_percent": 0.05,
    "interval": 60,
    "trend_window": 600,
    "one_hour_seconds": 3600,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_k": 2,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "position_size_percent": 0.03,
    "max_positions_per_coin": 1,
    "max_open_positions": 6,
    "low_funds_threshold": 10.0,
    "critical_low_threshold": 1.0,
    "minimum_reserve": 5.0,
    "pause_duration": 600,
    "trade_size_usd": 30.0,
    "force_sell_timeout": 1800,
    "rsi_buy_threshold": 75,
    "trading_pairs": [
        "BTC-USD",
        "BONK-USD", "PNUT-USD", "KAITO-USD", "TURBO-USD",
        "MOODENG-USD", "DOGE-USD", "XRP-USD",
        "ETH-USD", "SOL-USD", "ADA-USD", "SHIB-USD", "PEPE-USD",
        "WIF-USD", "FLOKI-USD", "MOG-USD"
    ],
    "paper_trading": False
}

# Initialize local config file if it doesn't exist
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)

# Load local config
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Function to fetch remote configuration
def fetch_remote_config():
    try:
        response = requests.get(REMOTE_CONFIG_URL, timeout=10)
        response.raise_for_status()
        remote_config = response.json()
        logging.info("Successfully fetched remote configuration.")
        logging.info(f"Remote config content: {json.dumps(remote_config, indent=2)}")  # Log the fetched config
        return remote_config
    except Exception as e:
        logging.warning(f"Failed to fetch remote config: {e}. Using local config.")
        return None

# Update local config with remote config if available
def update_config():
    global config
    remote_config = fetch_remote_config()
    if remote_config:
        config.update(remote_config)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info("Updated local configuration with remote settings.")
    else:
        logging.info("No remote config updates applied. Using existing settings.")

# Trading parameters from config with defaults
MIN_SELL_PROFIT = config.get('min_sell_profit', 0.01)
BTC_SELL_PROFIT = config.get('btc_sell_profit', 0.01)
STOP_LOSS_PERCENT = config.get('stop_loss_percent', 0.05)
INTERVAL = config.get('interval', 60)
TREND_WINDOW = config.get('trend_window', 600)
ONE_HOUR_SECONDS = config.get('one_hour_seconds', 3600)
RSI_PERIOD = config.get('rsi_period', 14)
MACD_FAST = config.get('macd_fast', 12)
MACD_SLOW = config.get('macd_slow', 26)
MACD_SIGNAL = config.get('macd_signal', 9)
BOLLINGER_PERIOD = config.get('bollinger_period', 20)
BOLLINGER_K = config.get('bollinger_k', 2)
ATR_PERIOD = config.get('atr_period', 14)
ATR_MULTIPLIER = config.get('atr_multiplier', 2.0)
POSITION_SIZE_PERCENT = config.get('position_size_percent', 0.03)
MAX_POSITIONS_PER_COIN = config.get('max_positions_per_coin', 1)
MAX_OPEN_POSITIONS = config.get('max_open_positions', 6)
LOW_FUNDS_THRESHOLD = config.get('low_funds_threshold', 10.0)
CRITICAL_LOW_THRESHOLD = config.get('critical_low_threshold', 1.0)
MINIMUM_RESERVE = config.get('minimum_reserve', 5.0)
PAUSE_DURATION = config.get('pause_duration', 600)
TRADE_SIZE_USD = config.get('trade_size_usd', 30.0)
FORCE_SELL_TIMEOUT = config.get('force_sell_timeout', 1800)
RSI_BUY_THRESHOLD = config.get('rsi_buy_threshold', 75)
TRADING_PAIRS = config.get('trading_pairs', [
    "BTC-USD",
    "BONK-USD", "PNUT-USD", "KAITO-USD", "TURBO-USD",
    "MOODENG-USD", "DOGE-USD", "XRP-USD",
    "ETH-USD", "SOL-USD", "ADA-USD", "SHIB-USD", "PEPE-USD",
    "WIF-USD", "FLOKI-USD", "MOG-USD"
])
PAPER_TRADING = config.get('paper_trading', False)

# State tracking
positions = {}
buy_prices = {}
last_buy_times = {}
last_sell_times = {}
price_history = {}
rsi_history = {}
macd_history = {}
bollinger_history = {}
atr_history = {}
ema_history = {}
total_profit = 0.0
product_details = {}
PAIRS = {}
pause_until = 0
total_positions = 0
one_hour_lows = {}
last_config_check = 0
config_check_interval = 3600  # Check for config updates every hour

MEME_QUOTES = {
    'buy': ["To the moon! HODL activated!", "Diamond hands, much wow!", "Bought the dip—such crypto, very gains!"],
    'sell': ["Profit secured! Lambos incoming!", "Sold the top—doge approves!", "Moon trip complete, wow!"]
}

def init_db():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS positions')
    c.execute('DROP TABLE IF EXISTS profits')
    c.execute('''CREATE TABLE IF NOT EXISTS positions
                 (pair TEXT, buy_price REAL, size REAL, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS profits
                 (total_profit REAL, timestamp TEXT)''')
    conn.commit()
    return conn

conn = init_db()

def save_state():
    c = conn.cursor()
    c.execute("DELETE FROM positions")
    for pair, buys in buy_prices.items():
        for buy_price, size in buys:
            c.execute("INSERT INTO positions VALUES (?, ?, ?, ?)", (pair, buy_price, size, datetime.now().isoformat()))
    c.execute("INSERT INTO profits VALUES (?, ?)", (total_profit, datetime.now().isoformat()))
    conn.commit()

def load_state():
    global positions, buy_prices, total_profit, total_positions
    c = conn.cursor()
    positions = {}
    buy_prices = {}
    c.execute("SELECT * FROM positions")
    for row in c.fetchall():
        pair, buy_price, size, _ = row
        if pair not in positions:
            positions[pair] = 0
            buy_prices[pair] = []
        if positions[pair] < MAX_POSITIONS_PER_COIN:
            positions[pair] += 1
            buy_prices[pair].append((buy_price, size))
        else:
            logging.warning(f"Skipping position for {pair} during load: exceeds MAX_POSITIONS_PER_COIN ({MAX_POSITIONS_PER_COIN})")
    total_positions = sum(positions.values())
    c.execute("SELECT total_profit FROM profits ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    if row:
        total_profit = row[0]
    else:
        total_profit = 0.0
    logging.info(f"Loaded state: {total_positions} positions, Total Profit: ${total_profit:.2f}")

client = cb(api_key=os.getenv('COINBASE_API_KEY'), api_secret=os.getenv('COINBASE_API_SECRET'))

app = Flask(__name__)

@app.route('/health')
def health():
    return "Bot is running", 200

@app.route('/dashboard')
def dashboard():
    return f"Total Profit: ${total_profit:.2f}<br>Positions: {positions}", 200

def run_flask():
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

def calculate_rsi(prices, period=RSI_PERIOD):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(prices) < slow + signal:
        return None, None, None
    exp1 = np.convolve(prices, np.ones(fast)/fast, mode='valid')[-1]
    exp2 = np.convolve(prices, np.ones(slow)/slow, mode='valid')[-1]
    macd = exp1 - exp2
    signal_line = np.convolve(np.pad([macd] * (len(prices) - slow + 1), (slow-1, 0), mode='constant'), np.ones(signal)/signal, mode='valid')[-1]
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_atr(highs, lows, closes, period=ATR_PERIOD):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr = max(tr1, tr2, tr3)
        trs.append(tr)
    if len(trs) < period:
        return None
    return np.mean(trs[-period:])

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def get_one_hour_low(pair, current_time):
    global one_hour_lows
    if pair in one_hour_lows:
        return one_hour_lows[pair]
    
    end_time = current_time
    start_time = end_time - ONE_HOUR_SECONDS
    candles = client.get_candles(
        product_id=pair,
        start=int(start_time),
        end=int(end_time),
        granularity="ONE_HOUR"
    )['candles']
    if not candles:
        logging.warning(f"No historical data for {pair} over the past 1 hour.")
        return None
    
    lows = [float(candle['low']) for candle in candles]
    one_hour_low = min(lows)
    one_hour_lows[pair] = one_hour_low
    logging.info(f"1-hour low for {pair}: ${one_hour_low:.8f}")
    return one_hour_low

def initialize_positions():
    global positions, buy_prices, total_positions
    try:
        accounts = client.get_accounts()['accounts']
        stablecoins = ['USD', 'USDC']
        for acc in accounts:
            if hasattr(acc, 'available_balance') and isinstance(acc.available_balance, dict) and 'value' in acc.available_balance:
                balance = float(acc.available_balance['value'])
                if balance > 0 and acc.currency not in stablecoins:
                    pair = f"{acc.currency}-USD"
                    try:
                        ticker = client.get_product(product_id=pair)
                        current_price = float(ticker['price'])
                        if pair not in positions:
                            positions[pair] = 0
                            buy_prices[pair] = []
                        if positions[pair] < MAX_POSITIONS_PER_COIN:
                            positions[pair] = 1
                            buy_prices[pair] = [(current_price, balance)]
                            logging.info(f"Initialized position for {pair}: {balance} units at ${current_price:.8f}")
                        else:
                            logging.warning(f"Skipping initialization for {pair}: already at MAX_POSITIONS_PER_COIN ({MAX_POSITIONS_PER_COIN})")
                    except Exception as e:
                        logging.error(f"Failed to fetch price for {pair} during initialization: {e}")
        total_positions = sum(positions.values())
        save_state()
    except Exception as e:
        logging.error(f"Failed to initialize positions: {e}")

def sync_positions_with_coinbase(owned_coins):
    global positions, buy_prices, total_positions
    current_holdings = {f"{currency}-USD" for currency in owned_coins if currency not in ['USD', 'USDC']}
    
    for pair in list(positions.keys()):
        if pair not in current_holdings:
            logging.info(f"Removing position {pair} from tracking as it is no longer held in Coinbase.")
            del positions[pair]
            del buy_prices[pair]
    
    for currency in owned_coins:
        if currency in ['USD', 'USDC']:
            continue
        pair = f"{currency}-USD"
        try:
            accounts = client.get_accounts()['accounts']
            acc = next(acc for acc in accounts if acc.currency == currency)
            balance = float(acc.available_balance['value'])
            if balance > 0:
                ticker = client.get_product(product_id=pair)
                current_price = float(ticker['price'])
                if pair not in positions or sum(size for _, size in buy_prices.get(pair, [])) != balance:
                    if pair not in positions or positions[pair] < MAX_POSITIONS_PER_COIN:
                        positions[pair] = 1
                        buy_prices[pair] = [(current_price, balance)]
                        logging.info(f"Updated position for {pair}: {balance} units at ${current_price:.8f}")
                    else:
                        logging.warning(f"Cannot update position for {pair}: already at MAX_POSITIONS_PER_COIN ({MAX_POSITIONS_PER_COIN})")
                else:
                    if pair in buy_prices and buy_prices[pair]:
                        buy_price = buy_prices[pair][0][0]
                        buy_prices[pair] = [(buy_price, balance)]
                        positions[pair] = 1
                        logging.info(f"Corrected size for {pair}: {balance} units at ${buy_price:.8f}")
        except Exception as e:
            logging.error(f"Failed to sync position for {pair}: {e}")
    
    total_positions = sum(positions.values())
    logging.info(f"Synced positions with Coinbase. Total open positions: {total_positions}")
    save_state()

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def get_trending_coins(owned_coins):
    trending = {}
    try:
        products = client.get_products()['products']
        coinbase_pairs = {p.product_id for p in products if p.quote_currency_id == 'USD'}
        for p in products:
            if p.product_id in TRADING_PAIRS:
                min_market_funds = getattr(p, 'min_market_funds', 1.0)
                product_details[p.product_id] = {
                    'base_increment': float(p.base_increment),
                    'min_market_funds': float(min_market_funds)
                }
    except Exception as e:
        logging.error(f"Coinbase products fetch failed: {e}")
        return trending, ""

    current_time = time.time()
    reason = ""
    for pair in TRADING_PAIRS:
        if pair not in coinbase_pairs:
            continue
        try:
            trades = client.get_market_trades(pair, limit=50)['trades']
            prices = [float(trade['price']) for trade in trades]
            highs = prices
            lows = prices
            closes = prices[::-1]
            
            if len(prices) < max(RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, 50):
                logging.info(f"{pair}: Not enough data for indicators (need {max(RSI_PERIOD + 1, MACD_SLOW + MACD_SIGNAL, 50)} prices, got {len(prices)})")
                continue

            rsi = calculate_rsi(prices)
            macd, signal_line, _ = calculate_macd(prices)
            atr = calculate_atr(highs, lows, closes)

            if any(v is None for v in [rsi, macd, signal_line, atr]):
                logging.info(f"{pair}: Indicator calculation failed - RSI: {rsi}, MACD: {macd}, Signal: {signal_line}, ATR: {atr}")
                continue

            current_price = closes[-1]
            logging.info(f"{pair}: Current = {current_price:.8f}, RSI = {rsi:.2f}, MACD = {macd:.8f}, Signal = {signal_line:.8f}, ATR = {atr:.8f}")

            if atr * ATR_MULTIPLIER > current_price * 0.05:
                logging.info(f"{pair}: Skipping - too volatile (ATR = {atr:.8f})")
                continue

            buy_signal = False
            reason = ""
            if pair == "BTC-USD":
                one_hour_low = get_one_hour_low(pair, current_time)
                if one_hour_low is None:
                    logging.info(f"{pair}: Skipping - could not determine 1-hour low")
                    continue
                if current_price < one_hour_low or rsi < RSI_BUY_THRESHOLD:
                    buy_signal = True
                    reason = (f"new 1-hour low (current: ${current_price:.8f}, 1-hour low: ${one_hour_low:.8f})" if current_price < one_hour_low else
                              f"RSI={rsi:.2f}")
            else:
                if rsi <= RSI_BUY_THRESHOLD and macd > signal_line:
                    buy_signal = True
                    reason = f"RSI={rsi:.2f}, MACD crossover"

            if buy_signal:
                target_usd = TRADE_SIZE_USD
                size = target_usd / current_price
                base_increment = product_details.get(pair, {}).get('base_increment', 0.0001)
                size = max(round(size / base_increment) * base_increment, base_increment)
                trending[pair] = str(size)
                logging.info(f"{pair}: Buy signal - {reason}")
            else:
                if pair == "BTC-USD":
                    logging.info(f"{pair}: No buy signal - current price ${current_price:.8f} not below 1-hour low ${one_hour_low:.8f}, RSI={rsi:.2f} (needs < {RSI_BUY_THRESHOLD})")
                else:
                    logging.info(f"{pair}: No buy signal - RSI={rsi:.2f} (needs <= {RSI_BUY_THRESHOLD}), MACD={macd:.8f}, Signal={signal_line:.8f} (needs MACD > Signal)")
        except Exception as e:
            logging.error(f"{pair}: Indicator calculation failed: {e}")

    if not trending:
        logging.warning("No trending coins detected with buy signals.")
    else:
        logging.info("Trending pairs: " + ", ".join(trending.keys()))
    return trending, reason

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def place_order(pair, side, price, size, reason):
    global total_profit, positions, buy_prices, total_positions, last_sell_times, last_buy_times
    if side == 'BUY':
        base_increment = product_details.get(pair, {}).get('base_increment', 0.0001)
        size = float(size)
        size = max(round(size / base_increment) * base_increment, base_increment)
        size_str = f"{size:.8f}".rstrip('0').rstrip('.')
    else:
        size = float(size)
        size_str = f"{size:.8f}".rstrip('0').rstrip('.')

    if PAPER_TRADING:
        logging.info(f"Paper Trading: {side} {size_str} {pair} at ${price:.8f} ({reason}) - Simulated")
        audit_logger.info(f"Paper Trading: {side} {size_str} {pair} at ${price:.8f} ({reason})")
        if side == 'BUY':
            if pair not in positions:
                positions[pair] = 0
                buy_prices[pair] = []
            positions[pair] += 1
            buy_prices[pair].append((price, size))
            last_buy_times[pair] = time.time()
            total_positions += 1
        elif side == 'SELL':
            if pair in positions and positions[pair] > 0 and buy_prices.get(pair):
                total_size = sum(size for _, size in buy_prices[pair])
                if total_size == 0:
                    logging.warning(f"No size available to sell for {pair}.")
                    return
                weighted_buy_price = sum(buy_price * size for buy_price, size in buy_prices[pair]) / total_size
                positions[pair] -= 1
                total_positions -= 1
                if positions[pair] == 0:
                    del positions[pair]
                gross_profit = (price - weighted_buy_price) * size
                net_profit = gross_profit * (1 - 0.005)
                logging.info(f"Paper Trading - {pair} Sell: Gross Profit=${gross_profit:.4f}, Net Profit=${net_profit:.4f}")
                total_profit += net_profit
                buy_prices[pair] = []
                logging.info(f"Paper Trading: Profit from sell: ${net_profit:.4f}, Total Profit: ${total_profit:.4f}")
                last_sell_times[pair] = time.time()
                if not buy_prices[pair]:
                    del buy_prices[pair]
        save_state()
        return

    client_order_id = f"{pair}-{side}-{int(time.time())}"
    try:
        order = client.market_order(
            product_id=pair,
            side=side,
            base_size=size_str,
            client_order_id=client_order_id
        )
    except Exception as e:
        logging.error(f"Order placement failed for {pair}: {e}")
        raise e

    quote = choice(MEME_QUOTES[side.lower()])
    logging.info(f"{pair}: {side} at ${price:.8f} ({reason}) - {quote} - Order: {order}")
    audit_logger.info(f"{pair}: {side} at ${price:.8f} ({reason}) - {quote} - Order: {order}")
    
    if order['success']:
        if side == 'BUY':
            if pair not in positions:
                positions[pair] = 0
                buy_prices[pair] = []
            positions[pair] += 1
            buy_prices[pair].append((price, size))
            last_buy_times[pair] = time.time()
            total_positions += 1
        elif side == 'SELL':
            if pair in positions and positions[pair] > 0 and buy_prices.get(pair):
                total_size = sum(size for _, size in buy_prices[pair])
                if total_size == 0:
                    logging.warning(f"No size available to sell for {pair}.")
                    return
                weighted_buy_price = sum(buy_price * size for buy_price, size in buy_prices[pair]) / total_size
                positions[pair] -= 1
                total_positions -= 1
                if positions[pair] == 0:
                    del positions[pair]
                gross_profit = (price - weighted_buy_price) * size
                net_profit = gross_profit * (1 - 0.005)
                logging.info(f"{pair} Sell: Gross Profit=${gross_profit:.4f}, Net Profit=${net_profit:.4f}")
                total_profit += net_profit
                buy_prices[pair] = []
                logging.info(f"{pair}: Profit from sell: ${net_profit:.4f}, Total Profit: ${total_profit:.4f}")
                last_sell_times[pair] = time.time()
                if not buy_prices[pair]:
                    del buy_prices[pair]
        save_state()
    else:
        logging.error(f"Order failed for {pair}: {order['error_response']}")
        usd_balance, owned_coins = check_funds()
        sync_positions_with_coinbase(owned_coins)

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def check_funds():
    accounts = client.get_accounts()['accounts']
    account_balances = []
    owned_coins = []
    for acc in accounts:
        balance = "N/A"
        if hasattr(acc, 'available_balance') and isinstance(acc.available_balance, dict) and 'value' in acc.available_balance:
            balance = acc.available_balance['value']
            if float(balance) > 0:
                owned_coins.append(acc.currency)
        account_balances.append(f"{acc.currency}: {balance}")
    logging.info("All accounts: " + ", ".join(account_balances))
    
    usd_variants = ['USD', 'CBI spot USD', 'CBI_SPOT_USD']
    usd_account = next((acc for acc in accounts if acc.currency in usd_variants), None)
    if usd_account:
        usd_attrs = vars(usd_account)
        logging.info(f"USD account details: {usd_attrs}")
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

def sell_all_positions():
    global positions, buy_prices, total_positions, last_sell_times
    for pair in list(buy_prices.keys()):
        try:
            ticker = client.get_product(product_id=pair)
            current_price = float(ticker['price'])
            _, owned_coins = check_funds()
            currency = pair.split('-USD')[0]
            accounts = client.get_accounts()['accounts']
            acc = next((acc for acc in accounts if acc.currency == currency), None)
            actual_size = float(acc.available_balance['value']) if acc and float(acc.available_balance['value']) > 0 else 0
            if actual_size == 0:
                logging.warning(f"No balance available for {pair} to sell.")
                continue
            place_order(pair, 'SELL', current_price, actual_size, "force sell to free capital")
            usd_balance, owned_coins = check_funds()
            sync_positions_with_coinbase(owned_coins)
        except Exception as e:
            logging.error(f"Failed to force sell {pair}: {e}")

def update_trading_pairs(owned_coins):
    global PAIRS
    new_pairs, reason = get_trending_coins(owned_coins)
    PAIRS = new_pairs
    return reason

def main():
    global total_positions, pause_until, last_config_check
    logging.info("Advanced Profit-Only Trading Bot: Targeting $100/Day! 🚀")
    load_state()
    if total_positions == 0:
        initialize_positions()
    else:
        logging.info(f"Positions loaded from database, skipping initialization. Total positions: {total_positions}")
    
    usd_balance, owned_coins = check_funds()
    sync_positions_with_coinbase(owned_coins)
    
    # Force sell DOGE to free up capital
    logging.info("Forcing sell of DOGE to free up capital...")
    try:
        ticker = client.get_product(product_id="DOGE-USD")
        current_price = float(ticker['price'])
        accounts = client.get_accounts()['accounts']
        acc = next((acc for acc in accounts if acc.currency == "DOGE"), None)
        actual_size = float(acc.available_balance['value']) if acc and float(acc.available_balance['value']) > 0 else 0
        if actual_size > 0:
            place_order("DOGE-USD", "SELL", current_price, actual_size, "manual sell to free capital")
        else:
            logging.warning("No DOGE balance available to sell.")
    except Exception as e:
        logging.error(f"Failed to manually sell DOGE: {e}")
    
    # Force sell FLOKI to free up capital
    logging.info("Forcing sell of FLOKI to free up capital...")
    try:
        ticker = client.get_product(product_id="FLOKI-USD")
        current_price = float(ticker['price'])
        accounts = client.get_accounts()['accounts']
        acc = next((acc for acc in accounts if acc.currency == "FLOKI"), None)
        actual_size = float(acc.available_balance['value']) if acc and float(acc.available_balance['value']) > 0 else 0
        if actual_size > 0:
            place_order("FLOKI-USD", "SELL", current_price, actual_size, "manual sell to free capital")
        else:
            logging.warning("No FLOKI balance available to sell.")
    except Exception as e:
        logging.error(f"Failed to manually sell FLOKI: {e}")
    
    usd_balance, owned_coins = check_funds()
    sync_positions_with_coinbase(owned_coins)
    
    last_trend_update = 0
    reason = ""
    
    while True:
        try:
            current_time = time.time()
            
            # Check for remote config updates every hour
            if current_time - last_config_check >= config_check_interval:
                update_config()
                # Update global variables after config change
                global MIN_SELL_PROFIT, BTC_SELL_PROFIT, STOP_LOSS_PERCENT, INTERVAL, TREND_WINDOW, ONE_HOUR_SECONDS
                global RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BOLLINGER_PERIOD, BOLLINGER_K
                global ATR_PERIOD, ATR_MULTIPLIER, POSITION_SIZE_PERCENT, MAX_POSITIONS_PER_COIN
                global MAX_OPEN_POSITIONS, LOW_FUNDS_THRESHOLD, CRITICAL_LOW_THRESHOLD, MINIMUM_RESERVE
                global PAUSE_DURATION, TRADE_SIZE_USD, FORCE_SELL_TIMEOUT, RSI_BUY_THRESHOLD, TRADING_PAIRS, PAPER_TRADING
                MIN_SELL_PROFIT = config.get('min_sell_profit', 0.01)
                BTC_SELL_PROFIT = config.get('btc_sell_profit', 0.01)
                STOP_LOSS_PERCENT = config.get('stop_loss_percent', 0.05)
                INTERVAL = config.get('interval', 60)
                TREND_WINDOW = config.get('trend_window', 600)
                ONE_HOUR_SECONDS = config.get('one_hour_seconds', 3600)
                RSI_PERIOD = config.get('rsi_period', 14)
                MACD_FAST = config.get('macd_fast', 12)
                MACD_SLOW = config.get('macd_slow', 26)
                MACD_SIGNAL = config.get('macd_signal', 9)
                BOLLINGER_PERIOD = config.get('bollinger_period', 20)
                BOLLINGER_K = config.get('bollinger_k', 2)
                ATR_PERIOD = config.get('atr_period', 14)
                ATR_MULTIPLIER = config.get('atr_multiplier', 2.0)
                POSITION_SIZE_PERCENT = config.get('position_size_percent', 0.03)
                MAX_POSITIONS_PER_COIN = config.get('max_positions_per_coin', 1)
                MAX_OPEN_POSITIONS = config.get('max_open_positions', 6)
                LOW_FUNDS_THRESHOLD = config.get('low_funds_threshold', 10.0)
                CRITICAL_LOW_THRESHOLD = config.get('critical_low_threshold', 1.0)
                MINIMUM_RESERVE = config.get('minimum_reserve', 5.0)
                PAUSE_DURATION = config.get('pause_duration', 600)
                TRADE_SIZE_USD = config.get('trade_size_usd', 30.0)
                FORCE_SELL_TIMEOUT = config.get('force_sell_timeout', 1800)
                RSI_BUY_THRESHOLD = config.get('rsi_buy_threshold', 75)
                TRADING_PAIRS = config.get('trading_pairs', [
                    "BTC-USD",
                    "BONK-USD", "PNUT-USD", "KAITO-USD", "TURBO-USD",
                    "MOODENG-USD", "DOGE-USD", "XRP-USD",
                    "ETH-USD", "SOL-USD", "ADA-USD", "SHIB-USD", "PEPE-USD",
                    "WIF-USD", "FLOKI-USD", "MOG-USD"
                ])
                PAPER_TRADING = config.get('paper_trading', False)
                last_config_check = current_time

            if current_time < pause_until:
                logging.info(f"Bot is paused until {datetime.fromtimestamp(pause_until)}. Sleeping for {int(pause_until - current_time)} seconds.")
                time.sleep(min(INTERVAL, pause_until - current_time))
                continue

            usd_balance, owned_coins = check_funds()
            sync_positions_with_coinbase(owned_coins)
            total_positions = sum(positions.values())
            logging.info(f"Total open positions: {total_positions}")

            if usd_balance < CRITICAL_LOW_THRESHOLD:
                logging.warning("USD balance critically low. Pausing trading...")
                pause_until = current_time + PAUSE_DURATION
                continue

            if usd_balance < LOW_FUNDS_THRESHOLD:
                logging.info(f"USD balance (${usd_balance:.2f}) below threshold (${LOW_FUNDS_THRESHOLD:.2f}). Skipping buy trades.")
                continue

            if current_time - last_trend_update >= TREND_WINDOW:
                if usd_balance < 0.5:
                    logging.warning("Low USD balance! Skipping buy trades...")
                    reason = ""
                else:
                    reason = update_trading_pairs(owned_coins)
                    last_trend_update = current_time

            for pair in list(PAIRS.keys()):
                try:
                    ticker = client.get_product(product_id=pair)
                    current_price = float(ticker['price'])
                except Exception as e:
                    logging.error(f"{pair}: Price fetch failed: {e}")
                    continue

                can_buy = False
                logging.info(f"Checking buy for {pair}: positions={positions.get(pair, 0)}, buy_prices={buy_prices.get(pair, [])}")
                if positions.get(pair, 0) < MAX_POSITIONS_PER_COIN:
                    if total_positions < MAX_OPEN_POSITIONS:
                        can_buy = True
                    else:
                        logging.info(f"Cannot buy {pair}: at MAX_OPEN_POSITIONS ({MAX_OPEN_POSITIONS})")
                else:
                    logging.info(f"Cannot buy {pair}: already at MAX_POSITIONS_PER_COIN ({MAX_POSITIONS_PER_COIN})")

                if can_buy:
                    last_buy_time = last_buy_times.get(pair, 0)
                    last_sell_time = last_sell_times.get(pair, 0)
                    if current_time - last_buy_time < COOLDOWN_PERIOD:
                        logging.info(f"{pair}: In cooldown period after last buy. Skipping...")
                        continue
                    if current_time - last_sell_time < COOLDOWN_AFTER_SELL:
                        logging.info(f"{pair}: In cooldown period after last sell. Skipping...")
                        continue
                    
                    size = PAIRS[pair]
                    cost = current_price * float(size)
                    if usd_balance - cost < MINIMUM_RESERVE:
                        logging.warning(f"Cannot buy {pair} - would reduce USD balance below minimum reserve (${MINIMUM_RESERVE:.2f}). Cost: ${cost:.2f}, Available: ${usd_balance:.2f}.")
                        continue
                    if cost > usd_balance:
                        logging.warning(f"Insufficient funds for {pair}. Cost: ${cost:.2f}, Available: ${usd_balance:.2f}.")
                        continue
                    place_order(pair, 'BUY', current_price, size, reason)
                
                if pair in positions and pair in buy_prices:
                    threshold = BTC_SELL_PROFIT if pair == "BTC-USD" else MIN_SELL_PROFIT
                    stop_loss = STOP_LOSS_PERCENT
                    _, owned_coins = check_funds()
                    currency = pair.split('-USD')[0]
                    accounts = client.get_accounts()['accounts']
                    acc = next((acc for acc in accounts if acc.currency == currency), None)
                    actual_size = float(acc.available_balance['value']) if acc and float(acc.available_balance['value']) > 0 else 0
                    if actual_size == 0:
                        continue
                    total_size = sum(size for _, size in buy_prices[pair])
                    weighted_buy_price = sum(buy_price * size for buy_price, size in buy_prices[pair]) / total_size
                    if current_price >= weighted_buy_price * (1 + threshold):
                        place_order(pair, 'SELL', current_price, actual_size, f"profit target (threshold {threshold*100:.1f}%)")
                        usd_balance, owned_coins = check_funds()
                        sync_positions_with_coinbase(owned_coins)
                    elif current_price <= weighted_buy_price * (1 - stop_loss):
                        logging.warning(f"{pair} hit stop-loss at ${current_price:.8f}, but skipping to avoid loss.")
                    elif pair in last_buy_times:
                        time_since_buy = current_time - last_buy_times[pair]
                        if time_since_buy >= FORCE_SELL_TIMEOUT:
                            place_order(pair, 'SELL', current_price, actual_size, "time-based sell to free capital")
                            usd_balance, owned_coins = check_funds()
                            sync_positions_with_coinbase(owned_coins)

            time.sleep(INTERVAL)
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()