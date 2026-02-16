# Repository Guidelines

## 프로젝트 구조 및 모듈 구성
이 저장소는 업비트 실시간 데이터를 이용해 매매 신호를 예측하는 Python 기반 ML 프로젝트입니다.
- `ml_dataset_creator.py`: 체결/호가 WebSocket 데이터를 30초 단위로 집계하고, 피처를 계산해 `dataset/sequence_dataset.csv`에 시퀀스 데이터를 누적 저장합니다.
- `train/train_cnn_lstm.py`: CNN-LSTM 모델을 학습하고 결과물을 `models/`에 저장합니다.
- `realtime_action_infer.py`: 저장된 모델/스케일러로 실시간 `sell/hold/buy` 추론을 수행합니다.
- `realtime_action_check.py`: 실시간 추론과 함께 정확도/매크로 F1을 모니터링합니다.
- `upbit_data_collector.py`: 마켓/티커 조회 및 급등 종목 감지 유틸리티입니다.
- `dataset/`, `models/`, `fig/`: 생성 데이터, 모델 산출물, 학습 시각화 파일 경로입니다.

## 빌드, 테스트, 개발 명령
먼저 가상환경을 만들고 의존성을 설치하세요.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
주요 실행 명령:
```bash
python ml_dataset_creator.py         # 실시간 수집 + 시퀀스 데이터셋 생성/갱신
python train/train_cnn_lstm.py       # 모델 학습, 스케일러/모델 저장, 그래프 생성
python realtime_action_infer.py      # 실시간 액션 추론 루프 실행
python realtime_action_check.py      # 실시간 추론 + 성능 지표 확인
```

## 코딩 스타일 및 네이밍 규칙
- Python PEP 8 기준, 들여쓰기는 스페이스 4칸을 사용합니다.
- 함수/변수는 `snake_case`, 상수는 `UPPER_SNAKE_CASE`를 사용합니다(예: `INTERVAL_SEC`).
- 학습/추론 간 `FEATURE_COLS` 순서와 의미를 반드시 일치시킵니다.
- 반복 실행 루프에서는 `print`보다 `logging.info` 사용을 우선합니다.

## 테스트 가이드
현재 `tests/` 기반의 정식 테스트 스위트는 없습니다.
- 변경 사항은 영향받는 스크립트를 직접 실행해 동작을 검증합니다.
- 모델 관련 변경 시 다음을 확인하세요: 데이터 shape, 라벨 분포, train/val macro-F1 추이, `models/` 산출물 저장 여부.
- 테스트를 추가할 경우 `pytest`를 사용하고 `tests/test_<module>.py` 형식을 따르세요.

## 커밋 및 PR 가이드
히스토리 기준으로 Conventional Commit을 사용합니다.
- `feat: ...` 기능 추가
- `fix: ...` 버그 수정
커밋은 작고 목적이 명확해야 합니다.
PR에는 아래 내용을 포함하세요.
- 변경 목적 및 범위
- 영향 파일과 동작 변화
- 재현/실행 명령
- 모델/추론 변경 시 전후 지표 또는 로그

## 보안 및 설정 팁
- API 키, 비밀값, 인증 정보는 커밋하지 마세요.
- `dataset/`, `models/`는 기본적으로 생성 산출물로 취급하고, 스냅샷 버전 관리가 필요할 때만 의도적으로 포함하세요.
