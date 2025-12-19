# Filename: realtime_action_check.py

import os
import time
import json
import logging
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
import websocket

# -------------------------
# 1️⃣ 설정
# -------------------------
MODEL_PATH = "models/lstm_train.pt"
SEQ_LEN = 20
FEATURE_COLS = [
    'last_return','mean_return','std_return','slope','accel','price_dev',
    'volume_ratio','activity_score','cusum_pos','cusum_neg','cusum_last'
]
ACTION_MAP = {'hold':0,'buy':1,'sell':2}
INV_ACTION_MAP = {v:k for k,v in ACTION_MAP.items()}
INTERVAL_SEC = 10
HISTORY_WINDOW = 200
MARKETS = ["KRW-BTC"]  # 예시

# -------------------------
# 2️⃣ 모델 정의
# -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# -------------------------
# 3️⃣ 모델 로드
# -------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = LSTMClassifier(num_features=len(FEATURE_COLS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
logging.info("Loaded model on device: %s", device)

# -------------------------
# 4️⃣ WebSocket 수집
# -------------------------
histories = {m: deque(maxlen=HISTORY_WINDOW) for m in MARKETS}
last_interval = {m: None for m in MARKETS}

def aggregate_ticks(ticks):
    if not ticks:
        return None
    df = pd.DataFrame(ticks)
    if df.empty: return None

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(pd.Series([int(time.time()*1000)]*len(df)), unit='ms')

    df['interval'] = df['timestamp'].dt.floor(f'{INTERVAL_SEC}s')
    agg = df.groupby('interval').agg(
        open=('trade_price','first'),
        high=('trade_price','max'),
        low=('trade_price','min'),
        close=('trade_price','last'),
        volume=('trade_volume','sum'),
        tick_count=('trade_price','count')
    ).reset_index()

    if len(agg) < 2:
        return None

    agg['last_return'] = agg['close'].pct_change().fillna(0)
    agg['mean_return'] = agg['last_return'].rolling(3, min_periods=1).mean()
    agg['std_return'] = agg['last_return'].rolling(3, min_periods=1).std().fillna(0)
    agg['slope'] = agg['close'].diff().fillna(0)
    agg['accel'] = agg['slope'].diff().fillna(0)
    agg['price_dev'] = agg['close'] - agg['close'].rolling(3, min_periods=1).mean()
    rolling_vol_mean = agg['volume'].rolling(3, min_periods=1).mean().replace(0, pd.NA)
    agg['volume_ratio'] = agg['volume'] / rolling_vol_mean
    agg['activity_score'] = agg['tick_count'] * agg['volume_ratio']
    agg['cusum_pos'] = agg['last_return'].apply(lambda x: max(x,0)).cumsum()
    agg['cusum_neg'] = agg['last_return'].apply(lambda x: min(x,0)).cumsum()
    agg['cusum_last'] = agg['last_return'].rolling(3, min_periods=1).sum()

    # action label 계산
    agg['action'] = 'hold'
    closes = agg['close'].values
    for i in range(len(closes)-1):
        change = (closes[i+1]-closes[i])/closes[i]
        if change >= 0.03:
            agg.loc[i,'action'] = 'buy'
        elif change <= -0.03:
            agg.loc[i,'action'] = 'sell'

    return agg

class WSTickCollector:
    def __init__(self, markets):
        self.markets = markets
        self.ticks = {m: deque() for m in markets}
        self.lock = False
        self.ws = None

    def on_message(self, ws, message):
        data = json.loads(message)
        market = data.get('code') or data.get('market')
        if not market or market not in self.markets:
            return
        self.ticks[market].append(data)

    def on_open(self, ws):
        payload = [{"ticket":"realtime"},{"type":"trade","codes":self.markets,"isOnlyRealtime":True}]
        ws.send(json.dumps(payload))
        logging.info("Subscribed WebSocket for %d symbols", len(self.markets))

    def start(self):
        def run_ws():
            self.ws = websocket.WebSocketApp(
                "wss://api.upbit.com/websocket/v1",
                on_message=self.on_message,
                on_open=self.on_open
            )
            self.ws.run_forever()
        import threading
        t = threading.Thread(target=run_ws, daemon=True)
        t.start()
        time.sleep(1)

    def pop_all(self):
        out = {m:list(self.ticks[m]) for m in self.markets}
        for m in self.markets:
            self.ticks[m].clear()
        return out

# -------------------------
# 5️⃣ 실시간 예측 + 맞았는지 확인
# -------------------------
collector = WSTickCollector(MARKETS)
collector.start()
logging.info("Realtime collector started")

try:
    while True:
        time.sleep(INTERVAL_SEC)
        ticks_data = collector.pop_all()

        for market in MARKETS:
            ticks = ticks_data.get(market, [])
            agg = aggregate_ticks(ticks)
            if agg is None or agg.empty:
                continue

            for idx, row in agg.iterrows():
                interval = row['interval']
                if last_interval[market] is not None and interval <= last_interval[market]:
                    continue

                histories[market].append(row)
                last_interval[market] = interval

                if len(histories[market]) >= SEQ_LEN + 1:
                    # 시퀀스 준비
                    seq = list(histories[market])[-(SEQ_LEN+1):-1]
                    X_input = torch.tensor([[ [r[col] for col in FEATURE_COLS] for r in seq ]], dtype=torch.float32).to(device)
                    y_true = ACTION_MAP[row['action']]

                    # 예측
                    with torch.no_grad():
                        output = model(X_input)
                        pred = output.argmax(dim=1).item()

                    # 결과 출력
                    match = "✅" if pred == y_true else "❌"
                    print(f"[{interval}] Market: {market} | Pred: {INV_ACTION_MAP[pred]} | True: {row['action']} {match}")

except KeyboardInterrupt:
    print("Stopped by user")
