# AML Risk Engine

AML(Anti-Money Laundering) 리스크 스코어링 엔진입니다.

## 개요

블록체인 트랜잭션을 분석하여 AML 리스크 점수를 계산하는 고급 시스템입니다. 룰 기반 평가와 머신러닝 기반 이상행동 탐지를 결합하여 정확한 리스크 평가를 제공합니다.

## 주요 기능

- **룰 기반 평가**: YAML 설정 파일을 통한 유연한 룰 관리
- **이상행동 탐지**: IsolationForest, LOF, XGBoost, LightGBM 등 다양한 ML 모델
- **고급 피처 추출**: 그래프 기반 중심성, fanout/hop 분석, 시간/금액 패턴 분석
- **RESTful API**: FastAPI 기반의 고성능 웹 API
- **종합 분석 보고서**: 상세한 분석 결과와 권장사항 제공

## 프로젝트 구조

```
aml-risk-engine/
├─ README.md
├─ requirements.txt
├─ src/
│  └─ riskengine/
│     ├─ __init__.py
│     ├─ rules/           # 룰북 (YAML + 검증 코드)
│     │  ├─ default_light.yaml
│     │  └─ schema.json
│     ├─ detectors/       # 이상행동 탐지 모듈
│     │  ├─ features.py   # 피처 추출 (fanout, hop, amount variance 등)
│     │  ├─ unsupervised.py   # IsolationForest, LOF 등
│     │  └─ supervised.py     # XGBoost, LightGBM 등
│     ├─ scoring.py       # 룰 기반 점수 + 이상행동 탐지 결과 집계
│     ├─ api.py           # FastAPI 엔드포인트 (POST /score)
│     └─ utils.py         # 유틸리티 함수들
├─ data/
│  └─ samples/tx.jsonl    # 샘플 데이터
└─ tests/
   └─ test_scoring.py
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 서버 실행

```bash
python -m uvicorn src.riskengine.api:app --reload
```

### 3. Docker 실행

```bash
docker build -t aml-risk-engine .
docker run -p 8000:8000 aml-risk-engine
```

## API 사용법

### 기본 리스크 스코어 계산

```bash
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{
       "address": "0x742d35Cc6631C0532925a3b8D35Cc6631C053292",
       "transactions": [
         {
           "from_address": "0x742d35Cc6631C0532925a3b8D35Cc6631C053292",
           "to_address": "0x8ba1f109551bD432803012645Hac136c0532925",
           "value": 1500000000000000000,
           "timestamp": 1704067200,
           "gas": 21000,
           "gasPrice": 20000000000
         }
       ],
       "window_hours": 24
     }'
```

### 종합 분석 보고서 생성

```bash
curl -X POST "http://localhost:8000/analyze/summary" \
     -H "Content-Type: application/json" \
     -d '{...}'  # 동일한 요청 형식
```

### 모델 상태 확인

```bash
curl -X GET "http://localhost:8000/models/status"
```

## 응답 형식

```json
{
  "address": "0x742d35Cc6631C0532925a3b8D35Cc6631C053292",
  "risk_score": 0.65,
  "risk_level": "MEDIUM",
  "weighted_score": 0.45,
  "anomaly_scores": {
    "isolation_forest": 0.3,
    "local_outlier_factor": 0.2,
    "combined_score": 0.25
  },
  "applied_rules": [
    {
      "name": "high_frequency_trading",
      "description": "고빈도 거래 패턴 감지",
      "risk_multiplier": 1.5
    }
  ],
  "feature_contributions": {
    "centrality_tx_frequency": 0.15,
    "exposure_total_value": 0.20,
    ...
  },
  "score_breakdown": {
    "rule_based": 0.45,
    "anomaly_detection": 0.25,
    "rule_multipliers": 1.5
  }
}
```

## 설정

### 룰 정책 설정

`src/riskengine/rules/default_light.yaml` 파일에서 다음을 설정할 수 있습니다:

- 피처별 가중치
- 리스크 레벨 임계값
- 룰 기반 평가 조건
- 이상행동 탐지 모델 가중치

### 환경 변수

- `POLICY_PATH`: 사용할 정책 파일 경로
- `MODEL_DIR`: 사전 훈련된 모델 저장 디렉토리

## 개발 가이드

### 새로운 피처 추가

`src/riskengine/detectors/features.py`에서 피처 추출 함수를 확장할 수 있습니다.

### 새로운 탐지 모델 추가

`src/riskengine/detectors/` 디렉토리에 새로운 탐지기를 추가할 수 있습니다.

### 테스트 실행

```bash
pytest tests/
```
