# Filename: upbit_data_collector.py

import requests
import websocket
import json
import threading
from collections import deque
import time

# -------------------------
# 1️⃣ 설정
# -------------------------
CHECK_INTERVAL = 30      # 거래량 체크 간격 (초)
VOLUME_THRESHOLD = 1.2   # 최근 대비 거래량 몇 배 증가 시 급증으로 판단
TOP_N = 10               # 상위 N개 급증 코인 선택
VOLUME_WINDOW = 6        # 비교를 위한 이동 윈도우(샘플 수)
SPAWN_WS = {}            # WebSocket 스레드 저장

tick_data = {}

# -------------------------
# 2️⃣ REST API로 거래량 조회
# -------------------------
def get_markets():
    url = "https://api.upbit.com/v1/market/all"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        markets = [m['market'] for m in resp.json() if m['market'].startswith("KRW-")]
        return markets
    except Exception as e:
        print(f"get_markets error: {e}")
        return []

def get_tickers(markets):
    if not markets:
        return []
    url = "https://api.upbit.com/v1/ticker"
    params = {"markets": ",".join(markets)}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"get_tickers error: {e}")
        return []

# -------------------------
# 3️⃣ WebSocket 구독
# -------------------------
def ws_on_message(ws, message):
    data = json.loads(message)
    symbol = data["code"]
    tick_data[symbol] = {
        "price": data["trade_price"],
        "volume": data["trade_volume"],
        "side": data["ask_bid"],
        "timestamp": data["timestamp"]
    }
    print(f"{symbol} | price: {data['trade_price']} | volume: {data['trade_volume']} | side: {data['ask_bid']}")

def start_ws(symbols):
    payload = [
        {"ticket": "tick_service"},
        {"type": "trade", "codes": symbols, "isOnlyRealtime": True}
    ]
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_message=ws_on_message
    )
    threading.Thread(target=lambda: ws.run_forever(), daemon=True).start()
    time.sleep(1)  # WS 연결 안정화
    ws.send(json.dumps(payload))
    return ws

# 가격 급상승 10개 종목 가져오기
prev_prices = {}  # 직전 종가 저장용

def detect_price_spike():
    global SPAWN_WS, prev_prices
    markets = get_markets()
    tickers = get_tickers(markets)

    spike_symbols = []

    for t in tickers:
        symbol = t["market"]
        price = t["trade_price"]
        prev_price = prev_prices.get(symbol, t["prev_closing_price"])  # 처음엔 직전 종가 사용
        prev_prices[symbol] = price  # 현재 가격 저장

        # 가격 상승률 계산
        price_change = (price - prev_price) / prev_price

        spike_symbols.append((symbol, price_change))

    # 상위 10개 상승 종목 선택
    spike_symbols = sorted(spike_symbols, key=lambda x: x[1], reverse=True)[:TOP_N]
    top_symbols = [s[0] for s in spike_symbols]

    # 신규 종목만 WebSocket 구독
    new_symbols = [s for s in top_symbols if s not in SPAWN_WS]
    if new_symbols:
        print("Price spike detected:", new_symbols)
        SPAWN_WS.update({s: True for s in new_symbols})
        start_ws(new_symbols)

# -------------------------
# 5️⃣ 메인 루프
# -------------------------
if __name__ == "__main__":
    print(f"Starting Upbit price spike collector. interval={CHECK_INTERVAL}s top_n={TOP_N}")
    try:
        while True:
            detect_price_spike()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("Stopped by user")
