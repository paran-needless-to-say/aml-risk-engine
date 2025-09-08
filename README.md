# AML Risk Engine

AML(Anti-Money Laundering) 리스크 스코어링 엔진입니다.

## 개요

블록체인 트랜잭션을 분석하여 AML 리스크 점수를 계산하는 고급 시스템입니다. 룰 기반 평가와 머신러닝 기반 이상행동 탐지를 결합하여 정확한 리스크 평가를 제공합니다.

## 🎯 리스크 스코어링 동작원리

### 핵심 개념

**5단계 파이프라인으로 블록체인 거래의 AML 위험도를 0-1점(5등급)으로 자동 평가**

```
피처 가중합 → 룰 평가 → AI 이상탐지 → 점수 집계 → 등급 분류
```

### 5단계 파이프라인 상세

#### 1️⃣ **피처 가중합** (기본 위험도 계산)

```python
# 각 피처에 중요도 가중치를 적용하여 기본 점수 산출
거래량(0.5) × 가중치(0.3) + 거래빈도(0.8) × 가중치(0.2) = 0.31점
```

#### 2️⃣ **룰 평가** (정책 기반 위험 조건 체크)

```yaml
# YAML 룰북의 조건들을 확인하고 위험 배수 적용
- "USD 10,000 이상 거래" → 위험배수 1.5배
- "10분간 3회 이상 거래" → 위험배수 1.2배
→ 총 배수: 1.5 × 1.2 = 1.8배
```

#### 3️⃣ **AI 이상탐지** (숨겨진 비정상 패턴 발견)

```python
# 머신러닝 모델들로 정상 거래와 다른 패턴 감지
IsolationForest: 0.3, XGBoost: 0.4 → 통합 이상도: 0.35
```

#### 4️⃣ **점수 집계** (모든 점수 통합)

```python
# 룰 기반(70%) + AI 기반(30%) 가중 평균
최종점수 = (룰점수 × 0.7) + (AI점수 × 0.3)
예시: (0.558 × 0.7) + (0.35 × 0.3) = 0.496점
```

#### 5️⃣ **등급 분류** (5단계 리스크 레벨)

```
🟢 VERY_LOW: 0-29점   (매우 안전)
🟡 LOW: 30-49점       (낮은 위험)
🟠 MEDIUM: 50-69점    (중간 위험)
🔴 HIGH: 70-89점      (높은 위험)
⚫ CRITICAL: 90-100점  (매우 위험)
```

### 실제 계산 예시

**입력 데이터:**

```json
{
  "거래량": 5000000000000000000, // 5 ETH
  "거래빈도": 30, // 시간당 30회
  "시간패턴": 0.8 // 불규칙성 지수
}
```

**계산 과정:**

```
1단계: 피처 가중합
- 거래량 정규화: 5 ETH → 0.6
- 빈도 정규화: 30회 → 0.3
- 가중합: (0.6×0.4) + (0.3×0.3) = 0.33

2단계: 룰 평가
- 고액거래 룰: 5 ETH × $2000 = $10,000 ≥ $10,000 ✓ (1.5배)
- 고빈도 룰: 30회/시간 → 5회/10분 ≥ 3회 ✓ (1.2배)
- 룰 적용 점수: 0.33 × 1.5 × 1.2 = 0.594

3단계: AI 이상탐지
- IsolationForest: 0.4 (이상 패턴 감지)
- XGBoost: 0.3 (위험 확률)
- 통합 점수: 0.35

4단계: 점수 집계
- 최종점수: (0.594 × 0.7) + (0.35 × 0.3) = 0.521

5단계: 등급 분류
- 52.1점 → "MEDIUM" 🟠
```

### 설정 기반 유연성

모든 임계값과 가중치는 `default_light.yaml`에서 설정 가능:

```yaml
# 리스크 등급 임계값
thresholds:
  risk_score:
    low: 30 # 조정 가능
    medium: 50 # 조정 가능
    high: 70 # 조정 가능
    critical: 90 # 조정 가능

# 집계 비율
aggregate:
  rule_weight: 0.7 # 룰 기반 70%
  ai_weight: 0.3 # AI 기반 30%
```

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

### 기본 응답 구조

```json
{
  "address": "0x742d35Cc6631C0532925a3b8D35Cc6631C053292",
  "risk_score": 0.521, // 최종 위험도 점수 (0-1)
  "risk_level": "MEDIUM", // 5단계 등급 (VERY_LOW/LOW/MEDIUM/HIGH/CRITICAL)
  "weighted_score": 0.33, // 1단계: 피처 가중합 점수
  "anomaly_scores": {
    // 3단계: AI 이상탐지 결과
    "isolation_forest": 0.4,
    "local_outlier_factor": 0.2,
    "xgboost": 0.3,
    "combined_score": 0.35
  },
  "applied_rules": [
    // 2단계: 적용된 룰들
    {
      "rule_id": "B-501",
      "name": "고액 거래",
      "description": "USD 10,000 이상 거래 감지",
      "risk_multiplier": 1.5,
      "condition": "usd_amount >= 10000"
    },
    {
      "rule_id": "tempo.B-101",
      "name": "버스트 거래 (10분)",
      "description": "10분간 3회 이상 거래",
      "risk_multiplier": 1.2,
      "condition": "tx_count(10m) >= 3"
    }
  ],
  "feature_contributions": {
    // 각 피처의 기여도
    "exposure_total_value": 0.24, // 거래량 기여도
    "centrality_tx_frequency": 0.09, // 거래빈도 기여도
    "behavior_interval_std": 0.05 // 시간패턴 기여도
  },
  "score_breakdown": {
    // 4단계: 점수 집계 상세
    "rule_based": 0.594, // 룰 기반 점수 (가중합 × 배수들)
    "anomaly_detection": 0.35, // AI 이상탐지 점수
    "rule_multipliers": 1.8, // 적용된 총 위험 배수
    "final_calculation": "(0.594×0.7) + (0.35×0.3) = 0.521"
  }
}
```

### 리스크 레벨별 의미

| 등급            | 점수 범위 | 설명             | 권장 조치            |
| --------------- | --------- | ---------------- | -------------------- |
| 🟢 **VERY_LOW** | 0-29점    | 매우 안전한 거래 | 정상 처리            |
| 🟡 **LOW**      | 30-49점   | 낮은 위험도      | 정상 처리, 기록 보관 |
| 🟠 **MEDIUM**   | 50-69점   | 중간 위험도      | 추가 검토 권장       |
| 🔴 **HIGH**     | 70-89점   | 높은 위험도      | 상세 조사 필요       |
| ⚫ **CRITICAL** | 90-100점  | 매우 위험        | 즉시 차단/신고 고려  |

## 설정

### 룰 정책 설정

`src/riskengine/rules/default_light.yaml` 파일에서 다음을 설정할 수 있습니다:

#### 리스크 등급 임계값

```yaml
thresholds:
  risk_score:
    low: 30 # VERY_LOW → LOW 경계
    medium: 50 # LOW → MEDIUM 경계
    high: 70 # MEDIUM → HIGH 경계
    critical: 90 # HIGH → CRITICAL 경계
```

#### 점수 집계 비율

```yaml
aggregate:
  method: weighted_sum
  rule_weight: 0.7 # 룰 기반 70%
  ai_weight: 0.3 # AI 기반 30%
```

#### 피처별 가중치

```yaml
feature_weights:
  exposure_total_value: 0.25 # 거래량 중요도
  centrality_tx_frequency: 0.20 # 거래빈도 중요도
  behavior_interval_std: 0.15 # 시간패턴 중요도
  # ... 기타 피처들
```

#### 룰 기반 평가 조건

```yaml
axes:
  compliance: # 법적 규제 위반
    C-001:
      condition: "sanctions.contains(counterparty)"
      score: 80
  exposure: # 위험 노출도
    E-101:
      condition: "mixer.used and usd_amount >= 20"
      weight: 0.8
  behavior: # 행동 패턴
    tempo:
      B-101:
        condition: "tx_count(10m) >= 3"
        score: 15
```

#### AI 모델 가중치

```yaml
anomaly_weights:
  isolation_forest: 0.3 # IsolationForest 비중
  local_outlier_factor: 0.2 # LOF 비중
  xgboost: 0.3 # XGBoost 비중
  lightgbm: 0.2 # LightGBM 비중
```

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
