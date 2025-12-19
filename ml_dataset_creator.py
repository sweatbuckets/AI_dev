# Filename: ml_dataset_creator.py

import requests
import pandas as pd
import numpy as np
import time
import os
import logging
import websocket
import threading
import json
from collections import deque
import upbit_data_collector

# 1️.하이퍼파라미터 설정
MARKET = "KRW-BTC"
INTERVAL_SEC = 30
LABEL_THRESHOLD = 0.008  # 0.8% 수익률 기준
HISTORY_WINDOW = 100
CSV_FILENAME = "ml_dataset.csv"
CSV_DIR = "data"
SELECT_TOP_N = 5
SEQ_LEN_SEC = 300  # 예측에 쓸 과거 데이터 기간 : 5분
MAX_SEQUENCES = 10000   


# 2. Upbit 틱 데이터 가져오기
class WSTickCollector:
    def __init__(self, markets):
        self.markets = list(markets)
        self.ticks = {m: deque() for m in self.markets}
        self.lock = threading.Lock()
        self.ws = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            market = data.get('code') or data.get('market')
            if not market:
                return
            tick = {}
            for f in ('trade_price','trade_volume','trade_date','trade_time','timestamp','trade_timestamp','ask_bid'):
                if f in data:
                    tick[f] = data[f]
            tick['market'] = market

            with self.lock:
                if market in self.ticks:
                    self.ticks[market].append(tick)
        except Exception as e:
            logging.debug("WS message parse error: %s", e)

    def on_open(self, ws):
        payload = [
            {"ticket": "ml_dataset_collector"},
            {"type": "trade", "codes": self.markets, "isOnlyRealtime": True}
        ]
        ws.send(json.dumps(payload))
        logging.info("WebSocket subscription sent for %d symbols", len(self.markets))

    def start(self):
        def run_ws():
            self.ws = websocket.WebSocketApp(
                "wss://api.upbit.com/websocket/v1",
                on_message=self.on_message,
                on_open=self.on_open
            )
            self.ws.run_forever()

        t = threading.Thread(target=run_ws, daemon=True)
        t.start()
        time.sleep(1)

    def pop_all(self):
        with self.lock:
            items = {m: list(self.ticks.get(m, [])) for m in self.markets}
            for m in self.markets:
                self.ticks[m].clear()
        return items


# 3. 틱 데이터 30초 단위 집계(30초봉 직접 제작)
def aggregate_ticks(ticks, interval_sec=30):
    if not ticks:
        return pd.DataFrame(columns=['interval','open','high','low','close','volume','tick_count'])

    df = pd.DataFrame(ticks)
    if df.empty:
        return pd.DataFrame(
            columns=['interval','open','high','low','close','volume','tick_count']
        )

    # timestamp 처리
    if 'timestamp' in df.columns and df['timestamp'].notnull().any():
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
    elif 'trade_timestamp' in df.columns and df['trade_timestamp'].notnull().any():
        df['timestamp'] = pd.to_datetime(df['trade_timestamp'].astype('int64'), unit='ms')
    elif 'trade_date' in df.columns and 'trade_time' in df.columns:
        dt_str = df['trade_date'].astype(str) + df['trade_time'].astype(str)
        df['timestamp'] = pd.to_datetime(dt_str, format='%Y%m%d%H%M%S', errors='coerce')
    else:
        now_ms = int(time.time()*1000)
        df['timestamp'] = pd.to_datetime([now_ms]*len(df), unit='ms')

    df = df.sort_values('timestamp').set_index('timestamp')

    # 종목별 + interval별 OHLCVT
    ohlcvt = df.resample(f'{interval_sec}s').agg(
        open=('trade_price','first'),
        high=('trade_price','max'),
        low=('trade_price','min'),
        close=('trade_price','last'),
        volume=('trade_volume','sum'),
        tick_count=('trade_price','count')
    )

    if ohlcvt.empty:
        return ohlcvt.reset_index().rename(columns={'timestamp':'interval'})
    
    # 틱 없는 구간 처리
    mask = ohlcvt['tick_count'] == 0
    ohlcvt.loc[mask, ['open','high','low','close']] = np.nan
    
    ohlcvt['close'] = ohlcvt['close'].ffill()
    for col in ('open', 'high', 'low'):
        ohlcvt[col] = ohlcvt[col].fillna(ohlcvt['close'])

    ohlcvt['volume'] = ohlcvt['volume'].fillna(0)
    ohlcvt['tick_count'] = ohlcvt['tick_count'].fillna(0)

    return ohlcvt.reset_index().rename(columns={'timestamp':'interval'})



# 단일 종목 특징(feature) 계산
def compute_features_one_market(agg: pd.DataFrame, df_ticks: pd.DataFrame | None = None, interval_sec: int = 30, verbose: bool = False):
    if agg is None or len(agg) == 0:
        return None  # 데이터 없으면 종료

    agg = agg.copy().reset_index(drop=True)

    # -------------------
    # 가격 관련 feature
    # -------------------
    agg['last_return'] = agg['close'].pct_change().fillna(0.0)
    agg['slope'] = agg['close'].diff().fillna(0.0)
    agg['accel'] = agg['slope'].diff().fillna(0.0)

    # 거래량 feature
    rolling_vol_mean = agg['volume'].rolling(3, min_periods=1).mean().replace(0, np.nan)
    agg['volume_ratio'] = (agg['volume'] / rolling_vol_mean).replace([np.inf, -np.inf], 0).fillna(0)

    # CUSUM
    cusum_pos = np.zeros(len(agg))
    cusum_neg = np.zeros(len(agg))
    for i in range(1, len(agg)):
        delta = agg['last_return'].iloc[i]
        cusum_pos[i] = max(0.0, cusum_pos[i-1] + delta)
        cusum_neg[i] = min(0.0, cusum_neg[i-1] + delta)
    agg['cusum_pos'] = cusum_pos
    agg['cusum_neg'] = cusum_neg

    # tick feature 초기화
    agg['bid_ask_imbalance'] = 0.0
    agg['spread_ratio'] = 0.0

    if df_ticks is not None and not df_ticks.empty:
        df_ticks = df_ticks.copy()
        if 'timestamp' in df_ticks.columns:
            ts = pd.to_datetime(df_ticks['timestamp'], unit='ms')
        elif 'trade_timestamp' in df_ticks.columns:
            ts = pd.to_datetime(df_ticks['trade_timestamp'], unit='ms')
        df_ticks['interval'] = ts.dt.floor(f'{interval_sec}s')

        # bid/ask imbalance
        if {'bid_volume','ask_volume'}.issubset(df_ticks.columns):
            ba_sum = df_ticks.groupby('interval')[['bid_volume','ask_volume']].sum()
            denom = ba_sum['bid_volume'] + ba_sum['ask_volume']
            imbalance = ((ba_sum['bid_volume'] - ba_sum['ask_volume']) / denom).replace([np.inf, -np.inf], 0).fillna(0)
            agg['bid_ask_imbalance'] = imbalance.reindex(agg['interval']).fillna(0).values

        # spread ratio
        if {'bid_price','ask_price'}.issubset(df_ticks.columns):
            spread = (df_ticks['ask_price'] - df_ticks['bid_price']) / ((df_ticks['ask_price'] + df_ticks['bid_price']) / 2)
            spread_mean = spread.groupby(df_ticks['interval']).mean()
            agg['spread_ratio'] = spread_mean.reindex(agg['interval']).fillna(0).values

    FEATURE_COLS = [
        'slope','accel','last_return','cusum_pos','cusum_neg',
        'volume_ratio','bid_ask_imbalance','spread_ratio'
    ]

    # -------------------
    # t0 feature 로그
    # -------------------
    if verbose and len(agg) >= 3:
        t0_index = 2  # ML-ready 첫 feature
        print("\n--- t0 feature 계산 로그 ---")
        print("사용된 OHLCV (최근 3개 interval):")
        print(agg.loc[t0_index-2:t0_index, ['open','high','low','close','volume']])
        print("이전 2개 interval feature:")
        print(agg.loc[t0_index-2:t0_index-1, FEATURE_COLS])
        print("계산된 t0 feature:")
        print(agg.loc[t0_index, FEATURE_COLS])
        print("--- 끝 ---\n")

    # ML용 df에는 3번째 interval부터 저장
    df_ml = agg[['interval'] + FEATURE_COLS]
    if len(df_ml) <= 2:
        return pd.DataFrame(columns=['interval'] + FEATURE_COLS)  # 초기 interval도 반환하지만 빈 DF
    else:
        return df_ml.iloc[2:].reset_index(drop=True)

# 다수 종목 특징(feature) 계산
market_history: dict[str, pd.DataFrame] = {}
logged_t0: dict[str, bool] = {}

def build_features_all_markets(ticks_by_market: dict[str, list[dict]], interval_sec: int = 30):
    features_by_market: dict[str, pd.DataFrame] = {}

    for market, ticks in ticks_by_market.items():
        if market not in market_history:
            market_history[market] = pd.DataFrame()
        if market not in logged_t0:
            logged_t0[market] = False    

        # 1️⃣ OHLCV 집계
        agg_ohlc = aggregate_ticks(ticks, interval_sec=interval_sec)
        if agg_ohlc.empty:
            continue

        # 2️⃣ 기존 히스토리에 누적
        market_history[market] = pd.concat([market_history[market], agg_ohlc]).reset_index(drop=True)

        # 3️⃣ feature 계산 (로그는 한 번만)
        agg_features = compute_features_one_market(
            market_history[market],
            interval_sec=interval_sec,
            verbose=(not logged_t0[market] and len(market_history[market]) >= 3)
        )

        # 로그 찍었으면 상태 업데이트
        if not logged_t0[market] and len(market_history[market]) >= 3:
            logged_t0[market] = True

        # 4️⃣ ML-ready feature 로그
        if len(agg_features) == 0:
            logging.info("Market %s: initial intervals, waiting for full features...", market)
        else:
            logging.info("Market %s: %d ML-ready intervals/features", market, len(agg_features))
            features_by_market[market] = agg_features

    return features_by_market

def create_sequences_one_market(
    df_feat: pd.DataFrame,
    features: list[str],
    seq_len_sec: int,
    interval_sec: int,
    threshold: float
):
    seq_len = seq_len_sec // interval_sec
    label_len = 30 // interval_sec

    if len(df_feat) < seq_len + label_len:
        return None, None

    X, Y = [], []
    # closes 대신 last_return 사용
    returns = df_feat['last_return'].values

    for i in range(len(df_feat) - seq_len - label_len + 1):
        seq_x = df_feat.iloc[i:i+seq_len][features].values
        X.append(seq_x)

        # 시퀀스 끝에서 다음 label_len 구간의 수익률 합으로 라벨 계산
        future_return = np.prod(1 + returns[i+seq_len:i+seq_len+label_len]) - 1

        if future_return >= threshold:
            Y.append(2)
        elif future_return <= -threshold:
            Y.append(0)
        else:
            Y.append(1)

    return np.array(X), np.array(Y)


def create_sequences_all_markets(
    features_by_market: dict[str, pd.DataFrame],
    features: list[str],
    seq_len_sec: int,
    interval_sec: int,
    threshold: float
    ):
    
    X_all, Y_all = [], []

    for market, df_feat in features_by_market.items():
        X_m, Y_m = create_sequences_one_market(
            df_feat=df_feat,
            features=features,
            seq_len_sec=seq_len_sec,
            interval_sec=interval_sec,
            threshold=threshold
        )

        if X_m is None or len(X_m) == 0:
            continue

        X_all.append(X_m)
        Y_all.append(Y_m)

    if not X_all:
        return None, None

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all

# -------------------------
# 5️⃣ CSV 저장
# -------------------------
def save_sequence_csv(X_all, Y_all, feature_dim, seq_len, save_path):

    if X_all is None or len(X_all) == 0:
        return

    num_sequences = X_all.shape[0]

    # 3D 배열 -> 2D DataFrame으로 변환
    # 각 시퀀스를 1D로 펼침: seq_len * feature_dim
    X_flat = X_all.reshape(num_sequences, seq_len * feature_dim)

    # 컬럼 이름 생성: f'feature0_t0', 'feature0_t1', ...
    columns = []
    for t in range(seq_len):
        for f in range(feature_dim):
            columns.append(f'feature{f}_t{t}')
    df = pd.DataFrame(X_flat, columns=columns)

    # 라벨 추가
    df['label'] = Y_all

    # 누적 저장: append 모드, 파일 없으면 header 출력
    df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

# -------------------------
# 6️⃣ Top symbols
# -------------------------
def select_top_symbols(n):
    markets = upbit_data_collector.get_markets()
    tickers = upbit_data_collector.get_tickers(markets)
    spike_symbols = []
    for t in tickers:
        symbol = t.get('market')
        price = t.get('trade_price')
        prev_price = t.get('prev_closing_price') or price
        try:
            change = (price - prev_price) / prev_price if prev_price else 0
        except:
            change = 0
        spike_symbols.append((symbol, change))
    spike_symbols = sorted(spike_symbols, key=lambda x: x[1], reverse=True)[:n]
    return [s[0] for s in spike_symbols]

# -------------------------
# 7️⃣ Main
# -------------------------


FEATURE_COLS = [
    'slope',
    'accel',
    'last_return',
    'cusum_pos',
    'cusum_neg',
    'volume_ratio',
    'bid_ask_imbalance',
    'spread_ratio'
]
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.info("Starting Upbit ML dataset collector...")


    SAVE_DIR = "./dataset"
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = f"{SAVE_DIR}/sequence_dataset.csv"

    seq_len = SEQ_LEN_SEC // INTERVAL_SEC
    total_sequences_saved = 0     # 누적 시퀀스 카운트

    # ======================
    # 1️⃣ 종목 선택
    # ======================
    symbols = select_top_symbols(n=SELECT_TOP_N)
    if not symbols:
        logging.error("No symbols selected. Exiting.")
        raise SystemExit(1)

    logging.info("Selected symbols: %s", symbols)

    # ======================
    # 2️⃣ WebSocket 시작
    # ======================
    collector = WSTickCollector(symbols)
    collector.start()
    logging.info("WebSocket collector started for %d symbols", len(symbols))

    # ======================
    # 3️⃣ 메인 루프
    # ======================
    try:
        while True: 
            time.sleep(INTERVAL_SEC)

            # 틱 수집
            ticks_by_market = collector.pop_all()
            tick_summary = {m: len(ticks_by_market.get(m, [])) for m in symbols}
            logging.info("Interval collected ticks: %s", tick_summary)

            # feature 계산
            features_by_market = build_features_all_markets(
                ticks_by_market=ticks_by_market,
                interval_sec=INTERVAL_SEC
            )

            if not features_by_market:
                logging.info("No features generated this interval")
                continue

            # 시퀀스 생성
            X_all, Y_all = create_sequences_all_markets(
                features_by_market=features_by_market,
                features=FEATURE_COLS,
                seq_len_sec=SEQ_LEN_SEC,
                interval_sec=INTERVAL_SEC,
                threshold=LABEL_THRESHOLD
            )

            if X_all is None:
                logging.info("Not enough data for sequences yet")
                continue

            # CSV 저장
            if total_sequences_saved < MAX_SEQUENCES:
                save_sequence_csv(
                    X_all=X_all,
                    Y_all=Y_all,
                    feature_dim=len(FEATURE_COLS),
                    seq_len=SEQ_LEN_SEC // INTERVAL_SEC,
                    save_path=SAVE_PATH
                )
                total_sequences_saved += len(X_all)
                logging.info("Saved sequences: %d (total=%d, shape=%s)", len(X_all), total_sequences_saved, X_all.shape)
            else:
                logging.info("Reached max sequences (%d). Stopping CSV save.", MAX_SEQUENCES)
                break

    except KeyboardInterrupt:
        logging.info("Stopped by user")