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
import joblib
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# =========================
# 1️⃣ 기본 설정
# =========================
MODEL_PATH = "models/cnn_lstm_model.pth"
SCALER_PATH = "models/feature_scaler.pkl"

INTERVAL_SEC = 30
SEQ_LEN = 10
LABEL_THRESHOLD = 0.008

FEATURE_COLS = [
    'slope','accel','last_return',
    'cusum_pos','cusum_neg',
    'volume_ratio','bid_ask_imbalance','spread_ratio'
]

ACTION_MAP = {'sell':0,'hold':1,'buy':2}
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
# 5️⃣ Feature 계산
# =========================
def compute_features(df):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['last_return'] = df['close'].pct_change().fillna(0)

    slopes, accels = [0.0], [0.0]
    for i in range(1, len(df)):
        s = (df.at[i,'close'] - df.at[i-1,'close']) / df.at[i-1,'close']
        a = s - slopes[-1]
        slopes.append(s)
        accels.append(a)
    df['slope'] = slopes
    df['accel'] = accels

    cp, cn = [0.0], [0.0]
    for r in df['last_return'].iloc[1:]:
        cp.append(max(0, cp[-1]+r))
        cn.append(min(0, cn[-1]+r))
    df['cusum_pos'] = cp
    df['cusum_neg'] = cn

    roll_vol = df['volume'].rolling(2).mean().shift(1)
    df['volume_ratio'] = (df['volume']/roll_vol).replace([float('inf'), -float('inf')], 0).fillna(0)

    denom = df['bid_volume'] + df['ask_volume']
    df['bid_ask_imbalance'] = (df['bid_volume'] - df['ask_volume'])/denom.replace(0,1)

    mid = (df['bid_price'] + df['ask_price']) / 2
    df['spread_ratio'] = (df['ask_price'] - df['bid_price'])/mid.replace(0,1)

    return df.iloc[2:].reset_index(drop=True)

# =========================
# 6️⃣ 인터벌 집계
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
            'bid_size':'sum','ask_size':'sum',
            'bid_price':'last','ask_price':'last'
        })
        ohlc = ohlc.join(ob, how='left')
    else:
        ohlc[['bid_size','ask_size','bid_price','ask_price']] = 0

    ohlc.rename(columns={'bid_size':'bid_volume','ask_size':'ask_volume'}, inplace=True)
    ohlc.reset_index(inplace=True)

    return ohlc

# =========================
# 7️⃣ 메인 루프
# =========================
if __name__ == "__main__":
    market = select_top_symbol()
    logging.info("Selected top symbol: %s", market)

    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model = CNNLSTM(len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    logging.info("CNN-LSTM model loaded on device: %s", device)

    scaler = joblib.load(SCALER_PATH)
    collector = WSTickCollector(market)
    collector.start()
    logging.info("WebSocket collector started for %s", market)

    market_history = pd.DataFrame()
    history = deque(maxlen=SEQ_LEN)
    last_interval = None
    y_true, y_pred = [], []
    pending_pred = None  # (pred_timestamp, pred_class)

    while True:
        time.sleep(INTERVAL_SEC)
        ticks, obs = collector.pop_all()
        agg = aggregate_interval(ticks, obs)
        if agg is None or len(agg) < 2:
            continue

        # market_history 누적
        market_history = pd.concat([market_history, agg]).drop_duplicates('timestamp').reset_index(drop=True)
        feat_df = compute_features(market_history)

        # 새로 생긴 인터벌만 처리
        new_rows = feat_df[feat_df['timestamp'] > (last_interval if last_interval else pd.Timestamp(0, tz='UTC'))]
        if new_rows.empty:
            continue

        for _, row in new_rows.iterrows():
            fv = [row[c] for c in FEATURE_COLS]
            fv_scaled = scaler.transform([fv])[0]
            history.append(fv_scaled)

            # 이전 인터벌 timestamp 업데이트
            last_interval = row['timestamp']

            # 한 구간 지연 평가:
            # t에서 만든 예측은 t+1 row가 들어왔을 때 row['last_return']로 평가
            if pending_pred is not None:
                pred_ts, prev_pred = pending_pred
                r = row['last_return']
                if r >= LABEL_THRESHOLD:
                    true = ACTION_MAP['buy']
                elif r <= -LABEL_THRESHOLD:
                    true = ACTION_MAP['sell']
                else:
                    true = ACTION_MAP['hold']

                y_true.append(true)
                y_pred.append(prev_pred)

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')

                logging.info(
                    "[pred@%s eval@%s] Pred=%s True=%s | Acc=%.3f F1=%.3f",
                    pred_ts,
                    row['timestamp'],
                    INV_ACTION_MAP[prev_pred],
                    INV_ACTION_MAP[true],
                    acc,
                    f1
                )

            # SEQ_LEN 이상이면 바로 예측
            if len(history) < SEQ_LEN:
                logging.info("[%s] Warming up (%d/%d)", row['timestamp'], len(history), SEQ_LEN)
                continue

            X = torch.from_numpy(np.array(history, dtype=np.float32))  # history (deque of ndarray) → single numpy array
            X = X.unsqueeze(0).to(device)  # 배치 차원 추가 + device로 이동
            
            with torch.no_grad():
                pred = model(X).argmax(1).item()

            # 이번 예측은 다음 인터벌에서 평가
            pending_pred = (row['timestamp'], pred)
