# UpbitExp

업비트 실시간 체결/호가 데이터를 기반으로 30초 단위 피처를 만들고, CNN-LSTM으로 `sell / hold / buy` 액션을 예측하는 프로젝트입니다.

## 주요 구성
- `ml_dataset_creator.py`: 실시간 데이터 수집 + 시퀀스 데이터셋(`dataset/sequence_dataset.csv`) 생성
- `train/train_cnn_lstm.py`: 모델 학습 및 아티팩트 저장(`models/`)
- `realtime_action_infer.py`: 실시간 액션 추론
- `realtime_action_check.py`: 실시간 추론 + 간이 성능 모니터링(Acc/F1)
- `upbit_data_collector.py`: 마켓/티커 조회 및 보조 수집 유틸

## 요구 사항
- Python 3.10+
- macOS/Linux (MPS 지원 시 Apple Silicon 가속 사용)

## 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행 순서
1. 데이터셋 생성/갱신
```bash
python ml_dataset_creator.py
```

2. 모델 학습
```bash
python train/train_cnn_lstm.py
```
- 결과물:
  - `models/cnn_lstm_model.pth`
  - `models/feature_scaler.pkl`
  - `fig/*.png`

3. 실시간 추론
```bash
python realtime_action_infer.py
```

4. 실시간 성능 체크(선택)
```bash
python realtime_action_check.py
```

## 모델/라벨 개요
- 입력: 최근 5분(30초 * 10스텝) 시퀀스
- 피처 수: 스텝당 8개 (`slope`, `accel`, `last_return`, `cusum_pos`, `cusum_neg`, `volume_ratio`, `bid_ask_imbalance`, `spread_ratio`)
- 라벨: 다음 30초 수익률 기준
  - `buy`: `>= +0.8%`
  - `sell`: `<= -0.8%`
  - 그 외 `hold`

## 주의 사항
- 현재 저장소는 **신호 생성/검증 중심**이며, 실제 주문 API 호출 코드는 포함되어 있지 않습니다.
- `dataset/`, `models/`는 생성 산출물로 `.gitignore` 대상입니다.
