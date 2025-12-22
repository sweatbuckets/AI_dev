# Filename: realtime_action_check.py

import time
import json
import logging
import threading
from collections import deque

import requests
import pandas as pd
import torch
import torch.nn as nn
import websocket
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1️⃣ 기본 설정
# =========================
MODEL_PATH = "models/cnn_lstm_model.pth"

INTERVAL_SEC = 30
SEQ_LEN = 10                 
LABEL_THRESHOLD = 0.008

FEATURE_COLS = [
    'open','high','low','close',
    'volume','tick_count',
    'bid_volume','ask_volume'
]

ACTION_MAP = {'hold':0,'buy':1,'sell':2}
INV_ACTION_MAP = {v:k for k,v in ACTION_MAP.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# =========================
# 2️⃣ 상승률 가장 높은 코인 선택
# =========================
def select_top_symbol():
    url = "https://api.upbit.com/v1/ticker"
    markets = requests.get("https://api.upbit.com/v1/market/all").json()
    krw_markets = [m['market'] for m in markets if m['market'].startswith("KRW-")]

    res = requests.get(url, params={"markets": ",".join(krw_markets)}).json()
    df = pd.DataFrame(res)
    df['change_rate'] = df['signed_change_rate'].abs()

    top = df.sort_values("change_rate", ascending=False).iloc[0]['market']
    return top

# =========================
# 3️⃣ CNN-LSTM 모델
# =========================
class CNNLSTM(nn.Module):
    def __init__(self, per_step_feature, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(per_step_feature, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])

# =========================
# 4️⃣ WebSocket Collector
# =========================
class WSTickCollector:
    def __init__(self, market, maxlen=5000):
        self.market = market
        self.ticks = deque(maxlen=maxlen)
        self.orderbooks = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def on_message(self, ws, message):
        data = json.loads(message)
        with self.lock:
            if 'trade_price' in data:
                self.ticks.append({
                    'trade_price': data['trade_price'],
                    'trade_volume': data['trade_volume'],
                    'timestamp': data['timestamp']
                })
            elif 'orderbook_units' in data:
                ts = data['timestamp']
                for u in data['orderbook_units']:
                    self.orderbooks.append({
                        'timestamp': ts,
                        'bid_size': u['bid_size'],
                        'ask_size': u['ask_size'],
                        'bid_price': u['bid_price'],
                        'ask_price': u['ask_price']
                    })

    def on_open(self, ws):
        payload = [
            {"ticket":"ml"},
            {"type":"trade","codes":[self.market],"isOnlyRealtime":True},
            {"type":"orderbook","codes":[self.market]}
        ]
        ws.send(json.dumps(payload))
        logging.info("WebSocket subscribed for %s", self.market)

    def start(self):
        def run():
            ws = websocket.WebSocketApp(
                "wss://api.upbit.com/websocket/v1",
                on_message=self.on_message,
                on_open=self.on_open
            )
            ws.run_forever()

        threading.Thread(target=run, daemon=True).start()
        time.sleep(1)

    def pop_all(self):
        with self.lock:
            t = list(self.ticks)
            o = list(self.orderbooks)
            self.ticks.clear()
            self.orderbooks.clear()
        return t, o

# =========================
# 5️⃣ 인터벌 집계
# =========================
def aggregate_interval(ticks, orderbooks):
    if not ticks:
        return None

    df = pd.DataFrame(ticks)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)

    ohlc = df['trade_price'].resample(f'{INTERVAL_SEC}s').ohlc()
    ohlc['volume'] = df['trade_volume'].resample(f'{INTERVAL_SEC}s').sum()
    ohlc['tick_count'] = df['trade_price'].resample(f'{INTERVAL_SEC}s').count()

    if orderbooks:
        ob = pd.DataFrame(orderbooks)
        ob['timestamp'] = pd.to_datetime(ob['timestamp'], unit='ms', utc=True)
        ob.set_index('timestamp', inplace=True)
        ob = ob.resample(f'{INTERVAL_SEC}s').agg({
            'bid_size':'sum','ask_size':'sum'
        })
        ohlc = ohlc.join(ob, how='left')
    else:
        ohlc['bid_size'] = ohlc['ask_size'] = 0

    ohlc.rename(columns={'bid_size':'bid_volume','ask_size':'ask_volume'}, inplace=True)
    ohlc.reset_index(inplace=True)

    closes = ohlc['close'].values
    actions = ['hold'] * len(closes)
    for i in range(len(closes)-1):
        r = (closes[i+1] - closes[i]) / closes[i]
        if r >= LABEL_THRESHOLD:
            actions[i] = 'buy'
        elif r <= -LABEL_THRESHOLD:
            actions[i] = 'sell'
    ohlc['action'] = actions

    return ohlc

# =========================
# 6️⃣ 메인 루프
# =========================
if __name__ == "__main__":
    market = select_top_symbol()
    logging.info("Selected top symbol: %s", market)

    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model = CNNLSTM(len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    logging.info("CNN-LSTM model loaded on device: %s", device)

    collector = WSTickCollector(market)
    collector.start()
    logging.info("WebSocket collector started for %s", market)

    history = deque(maxlen=SEQ_LEN)
    last_interval = None

    y_true, y_pred = [], []

    while True:
        time.sleep(INTERVAL_SEC)
        ticks, obs = collector.pop_all()
        agg = aggregate_interval(ticks, obs)
        if agg is None or len(agg) < 2:
            continue

        row = agg.iloc[-2]   # 마지막은 아직 라벨 없으므로 그 전의 인터벌로 비교
        interval = row['timestamp']

        if last_interval is not None and interval <= last_interval:
            continue
        last_interval = interval

        fv = [row[c] for c in FEATURE_COLS]
        history.append(fv)

        if len(history) < SEQ_LEN:
            logging.info("[%s] Warming up (%d/%d)", interval, len(history), SEQ_LEN)
            continue

        X = torch.tensor([list(history)], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(X).argmax(1).item()

        true = ACTION_MAP[row['action']]
        y_true.append(true)
        y_pred.append(pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        logging.info(
            "[%s] Pred=%s True=%s | Acc=%.3f F1=%.3f",
            interval, INV_ACTION_MAP[pred], row['action'], acc, f1
        )