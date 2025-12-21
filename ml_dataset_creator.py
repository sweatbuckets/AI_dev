# Filename: ml_dataset_creator.py

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
INTERVAL_SEC = 30
LABEL_THRESHOLD = 0.008  # 0.8% 수익률 기준
SELECT_TOP_N = 5
SEQ_LEN_SEC = 300  # 예측에 쓸 과거 데이터 기간 : 5분 = 시퀀스 데이터 구성하는 인터벌 개수 : 10
MAX_SEQUENCES = 2000   

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


# 2. Upbit 틱 데이터 가져오기
class WSTickCollector:
    def __init__(self, markets):
        self.markets = list(markets)
        self.ticks = {m: deque() for m in self.markets}
        self.orderbooks = {m: deque() for m in self.markets}
        self.lock = threading.Lock()
        self.ws = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            market = data.get('code')
            if not market:
                return
        

            # -----------------
            # TRADE
            # -----------------
            if 'trade_price' in data:
                tick = {
                    'market': market,
                    'trade_price': data.get('trade_price'),
                    'trade_volume': data.get('trade_volume'),
                    'timestamp': data.get('timestamp'),
                    'ask_bid': data.get('ask_bid'),
                }
                with self.lock:
                    self.ticks[market].append(tick)


            # -----------------
            # ORDERBOOK
            # -----------------
            elif 'orderbook_units' in data:
                ts = data.get('timestamp')
                with self.lock:
                    for u in data['orderbook_units']:
                        self.orderbooks[market].append({
                            'market': market,
                            'timestamp': ts,
                            'bid_price': u['bid_price'],
                            'bid_size': u['bid_size'],
                            'ask_price': u['ask_price'],
                            'ask_size': u['ask_size'],
                        })


        except Exception as e:
            logging.debug("WS message parse error: %s", e)

    def on_open(self, ws):
        payload = [
            {"ticket": "ml_dataset_collector"},
            {"type": "trade", 
             "codes": self.markets,
             "isOnlyRealtime": True
            },
            {
            "type": "orderbook",
            "codes": self.markets
            }
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
            trade_items = {m: list(self.ticks[m]) for m in self.markets}
            orderbook_items = {m: list(self.orderbooks[m]) for m in self.markets}

            for m in self.markets:
                self.ticks[m].clear()
                self.orderbooks[m].clear()

        return trade_items, orderbook_items



# 틱 데이터 30초 단위 집계(30초봉 직접 제작)
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

# orderbook 데이터 30초 단위 집계
def aggregate_orderbook(orderbooks, interval_sec=30):
    if not orderbooks:
        return pd.DataFrame(columns=[
            'interval',
            'bid_volume',
            'ask_volume',
            'bid_price',
            'ask_price'
        ])

    df = pd.DataFrame(orderbooks)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['interval'] = df['timestamp'].dt.floor(f'{interval_sec}s')

    agg = df.groupby('interval').agg(
        bid_volume=('bid_size','sum'),
        ask_volume=('ask_size','sum'),
        bid_price=('bid_price','mean'),
        ask_price=('ask_price','mean'),
    )

    return agg.reset_index()

# 틱 + 오더북 합친 인터벌 데이터 생성
def aggregate_interval(ticks, orderbooks, interval_sec=30):
    ohlcv = aggregate_ticks(ticks, interval_sec)
    ob = aggregate_orderbook(orderbooks, interval_sec)

    if ohlcv.empty:
        return None

    if not ob.empty:
        ohlcv = ohlcv.merge(ob, on='interval', how='left')
    else:
        ohlcv[['bid_volume','ask_volume','bid_price','ask_price']] = 0.0

    return ohlcv

# 단일 종목 특징(feature) 계산
def compute_features_one_market(
        agg: pd.DataFrame,
        interval_sec: int = 30, 
        verbose: bool = False):
    if agg is None or len(agg) == 0:
        return None  # 데이터 없으면 종료
    
    agg = agg.copy()
    # 시간순 정렬
    agg = agg.sort_values('interval').reset_index(drop=True)

    # -------------------
    # 가격 관련 feature
    # -------------------
    # last_return은 pct_change
    agg['last_return'] = agg['close'].pct_change().fillna(0.0)

    # slope와 accel 계산 (변화율 기준)
    slopes = [0.0]  # 첫 interval slope = 0
    accels = [0.0]  # 첫 interval accel = 0
    for i in range(1, len(agg)):
        slope_i = (agg.at[i, 'close'] - agg.at[i-1, 'close']) / agg.at[i-1, 'close']
        slopes.append(slope_i)
        prev_slope = slopes[i-1] if i > 1 else 0.0
        accel_i = slope_i - prev_slope
        accels.append(accel_i)
    agg['slope'] = slopes
    agg['accel'] = accels

    # 거래량 feature (이전 2 interval 평균, 본인 제외)
    rolling_vol_mean = agg['volume'].rolling(2, min_periods=1).mean().shift(1)
    agg['volume_ratio'] = (agg['volume'] / rolling_vol_mean).replace([np.inf, -np.inf], 0).fillna(0)

    # CUSUM
    cusum_pos = [0.0]
    cusum_neg = [0.0]
    for i in range(1, len(agg)):
        delta = agg.at[i, 'last_return']
        cusum_pos.append(max(0.0, cusum_pos[i-1] + delta))
        cusum_neg.append(min(0.0, cusum_neg[i-1] + delta))
    agg['cusum_pos'] = cusum_pos
    agg['cusum_neg'] = cusum_neg

    # tick feature
    # bid/ask imbalance
    if {'bid_volume','ask_volume'}.issubset(agg.columns):
        denom = agg['bid_volume'] + agg['ask_volume']
        agg['bid_ask_imbalance'] = np.where(
            denom != 0,
            (agg['bid_volume'] - agg['ask_volume']) / denom,
            0
        )
    else:
        agg['bid_ask_imbalance'] = np.nan

    # spread ratio
    if {'bid_price','ask_price'}.issubset(agg.columns):
        mid = (agg['bid_price'] + agg['ask_price']) / 2
        agg['spread_ratio'] = np.where(
            mid != 0,
            (agg['ask_price'] - agg['bid_price']) / mid,
            0
        )
    else:
        agg['spread_ratio'] = np.nan

    FEATURE_COLS = [
        'slope','accel','last_return','cusum_pos','cusum_neg',
        'volume_ratio','bid_ask_imbalance','spread_ratio'
    ]

    # ML용 df에는 3번째 interval부터 저장
    df_ml = agg[['interval'] + FEATURE_COLS]
    if len(df_ml) <= 2:
        return pd.DataFrame(columns=['interval'] + FEATURE_COLS)
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
        market_history[market] = pd.concat([market_history[market], agg_ohlc]) \
            .drop_duplicates(subset=['interval']) \
            .sort_values('interval') \
            .reset_index(drop=True)
        
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
        if agg_features is None or agg_features.empty:
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
        return None, None, None

    X, Y, seq_ids = [], [], []
    
    returns = df_feat['last_return'].values

    for i in range(len(df_feat) - seq_len - label_len + 1):
        seq_x = df_feat.iloc[i:i+seq_len][features].values
        X.append(seq_x)

        # 시퀀스 끝에서 다음 label_len 구간의 수익률로 라벨 계산
        future_return = np.prod(1 + returns[i+seq_len:i+seq_len+label_len]) - 1

        if future_return >= threshold:
            Y.append(2)
        elif future_return <= -threshold:
            Y.append(0)
        else:
            Y.append(1)

        # 수정: 시퀀스 시작 interval 기준으로 seq_ids 저장
        seq_ids.append(df_feat['interval'].iloc[i])  

    return np.array(X), np.array(Y), seq_ids


# dedup용
def create_sequences_all_markets(
    features_by_market: dict[str, pd.DataFrame],
    features: list[str],
    seq_len_sec: int,
    interval_sec: int,
    threshold: float
):
    X_all, Y_all, seq_ids_all = [], [], []

    seq_len = seq_len_sec // interval_sec
    label_len = 30 // interval_sec

    for market, df_feat in features_by_market.items():
        if len(df_feat) < seq_len:
            continue

        X_m, Y_m, seq_ids_m = create_sequences_one_market(
            df_feat=df_feat,
            features=features,
            seq_len_sec=seq_len_sec,
            interval_sec=interval_sec,
            threshold=threshold
        )

        if X_m is None or len(X_m) == 0:
            continue

        # sequence_id = (market, start_interval)
        seq_ids_all.extend([(market, sid) for sid in seq_ids_m])
        X_all.append(X_m)
        Y_all.append(Y_m)

    if not X_all:
        return None, None, None

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all, seq_ids_all

# -------------------------
# CSV 저장
# -------------------------
def save_sequence_csv(X_all, Y_all, feature_dim, seq_len, save_path):

    if X_all is None or len(X_all) == 0:
        logging.info("No sequences to save.")
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
# Top symbols
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

def log_t0_feature(agg, df_feat, feature_cols, t0_index: int = 2):
    print("\n--- t0 feature 계산 로그 ---")
    print("사용된 OHLCV (최근 3개 interval):")
    print(agg.loc[t0_index:t0_index+2, ['open','high','low','close','volume']])

    print("계산된 t0 feature:")
    print(df_feat.loc[t0_index, feature_cols])
    print("--- 끝 ---\n")

# -------------------------
#  Main
# -------------------------
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
    saved_intervals = set()           # interval 단위 중복 방지
    features_by_market = {}
    market_history = {}
    logged_t0 = {}
    saved_sequence_ids = set()      # 시퀀스 단위 저장 체크 및 MAX_SEQUENCES 제한용
    try:
        while True: 
            time.sleep(INTERVAL_SEC)

            # 틱 수집
            ticks_by_market, orderbooks_by_market = collector.pop_all()
            tick_summary = {m: len(ticks_by_market.get(m, [])) for m in symbols}
            logging.info("Interval collected ticks: %s", tick_summary)

            # feature 계산
            for market in symbols:
                ticks = ticks_by_market.get(market, [])
                orderbooks = orderbooks_by_market.get(market, [])

                # interval 단위 집계 (OHLCV + orderbook)
                agg = aggregate_interval(
                    ticks=ticks,
                    orderbooks=orderbooks,
                    interval_sec=INTERVAL_SEC
                )
                if agg is None or agg.empty:
                    continue

                agg['interval'] = pd.to_datetime(agg['interval'])

                # 히스토리 누적 (중복 interval 제거 필수)
                if market not in market_history:
                    market_history[market] = agg.copy()
                    logged_t0[market] = False
                else:
                    market_history[market] = (
                        pd.concat([market_history[market], agg], ignore_index=True)
                        .drop_duplicates(subset=['interval'])
                        .sort_values('interval')
                        .reset_index(drop=True)
                    )

                # feature 계산
                df_feat = compute_features_one_market(
                    agg=market_history[market],
                    interval_sec=INTERVAL_SEC,
                    verbose=False
                )

                # t0 feature 로그
                if not logged_t0[market] and len(df_feat) >= 1:
                    log_t0_feature(
                        agg=market_history[market],
                        df_feat=df_feat,
                        feature_cols=FEATURE_COLS,
                        t0_index=0
                    )
                    logged_t0[market] = True

                # features_by_market 누적
                if df_feat is not None and not df_feat.empty:
                    if market not in features_by_market:
                        features_by_market[market] = df_feat
                        added_count = len(df_feat)
                    else:
                        old_len = len(features_by_market[market])
                        features_by_market[market] = pd.concat(
                            [features_by_market[market], df_feat],
                            ignore_index=True
                        ).drop_duplicates(subset=['interval']).reset_index(drop=True)
                        added_count = len(features_by_market[market]) - old_len

                    logging.info(
                        "Market %s: +%d ML-ready intervals (total=%d)",
                        market,
                        added_count,
                        len(features_by_market[market])
                    )
            # 시퀀스 생성
            X_all, Y_all, seq_ids = create_sequences_all_markets(
                features_by_market=features_by_market,
                features=FEATURE_COLS,
                seq_len_sec=SEQ_LEN_SEC,
                interval_sec=INTERVAL_SEC,
                threshold=LABEL_THRESHOLD
            )

            if X_all is None:
                logging.info("No sequences yet")
                continue


            # 이미 저장한 시퀀스 이후 것만 추출(dedup)
            # ---------------------------
            # dedup + CSV 저장
            # ---------------------------
    
            X_new, Y_new = [], []

            for x, y, (market, start_interval) in zip(X_all, Y_all, seq_ids):
                start_interval = pd.to_datetime(start_interval)

                # 시퀀스 단위 dedup만 수행
                if (market, start_interval) in saved_sequence_ids:
                    continue

                saved_sequence_ids.add((market, start_interval))
                X_new.append(x)
                Y_new.append(y)

            # CSV 저장
            if X_new and len(saved_sequence_ids) < MAX_SEQUENCES:
                X_new = np.array(X_new)
                Y_new = np.array(Y_new)

                save_sequence_csv(
                    X_all=X_new,
                    Y_all=Y_new,
                    feature_dim=len(FEATURE_COLS),
                    seq_len=SEQ_LEN_SEC // INTERVAL_SEC,
                    save_path=SAVE_PATH
                )

                logging.info(
                    "Saved new sequences: %d (total=%d, shape=%s)",
                    len(X_new),
                    len(saved_sequence_ids),
                    X_new.shape
                )

            elif len(saved_sequence_ids) >= MAX_SEQUENCES:
                logging.info("Reached max sequences (%d). Stopping CSV save.", MAX_SEQUENCES)
                break

            # CSV 저장 후 features_by_market에서 interval 제거
            for market, df in features_by_market.items():
                mask = pd.Series(True, index=df.index)
                for _, start_interval in seq_ids:
                    if _ != market:
                        continue
                    # 시퀀스 interval 범위 계산
                seq_start = start_interval
                seq_end = start_interval + pd.Timedelta(seconds=SEQ_LEN_SEC - INTERVAL_SEC)
                mask &= ~((df['interval'] >= seq_start) & (df['interval'] <= seq_end))
            features_by_market[market] = df[mask].reset_index(drop=True)


    except KeyboardInterrupt:
        logging.info("Stopped by user")