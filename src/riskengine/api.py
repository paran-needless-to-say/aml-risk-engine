"""
FastAPI 기반 리스크 스코어링 API

POST /score 엔드포인트를 제공합니다.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uvicorn

from .detectors.features import extract_advanced_features
from .scoring import calculate_risk_score, RiskScorer
from .utils import validate_address, validate_transaction, create_summary_report


app = FastAPI(
    title="AML Risk Engine API",
    description="블록체인 트랜잭션의 AML 리스크 스코어를 계산하는 API",
    version="0.1.0"
)


class TransactionData(BaseModel):
    """트랜잭션 데이터 모델"""
    from_address: str = Field(..., description="송신자 주소")
    to_address: str = Field(..., description="수신자 주소") 
    value: float = Field(..., description="거래 금액")
    timestamp: Optional[int] = Field(None, description="타임스탬프")
    gas: Optional[float] = Field(None, description="가스 한도")
    gasPrice: Optional[float] = Field(None, description="가스 가격")
    hash: Optional[str] = Field(None, description="트랜잭션 해시")


class ScoreRequest(BaseModel):
    """스코어 요청 모델"""
    address: str = Field(..., description="분석할 주소")
    transactions: List[TransactionData] = Field(..., description="트랜잭션 리스트")
    window_hours: Optional[int] = Field(24, description="분석 시간 윈도우 (시간)")
    policy_path: Optional[str] = Field(None, description="정책 파일 경로")


class ScoreResponse(BaseModel):
    """스코어 응답 모델"""
    address: str = Field(..., description="분석된 주소")
    risk_score: float = Field(..., description="리스크 스코어 (0-1)")
    risk_level: str = Field(..., description="리스크 레벨")
    weighted_score: float = Field(..., description="가중합 스코어")
    applied_rules: List[Dict[str, Any]] = Field(..., description="적용된 룰들")
    feature_contributions: Dict[str, float] = Field(..., description="피처 기여도")
    features: Dict[str, float] = Field(..., description="추출된 피처들")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AML Risk Engine API",
        "version": "0.1.0",
        "endpoints": {
            "score": "POST /score - 리스크 스코어 계산",
            "health": "GET /health - 헬스체크"
        }
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "aml-risk-engine"}


@app.post("/score", response_model=ScoreResponse)
async def score_address(request: ScoreRequest):
    """
    주소의 AML 리스크 스코어를 계산합니다.
    
    Args:
        request: 스코어 요청 데이터
    
    Returns:
        리스크 스코어 결과
    """
    try:
        # 트랜잭션 데이터를 딕셔너리 형태로 변환
        transactions = [tx.dict() for tx in request.transactions]
        
        # 트랜잭션 데이터에서 from/to 키 매핑
        for tx in transactions:
            tx['from'] = tx.pop('from_address')
            tx['to'] = tx.pop('to_address')
        
        # 주소 유효성 검증
        if not validate_address(request.address):
            raise HTTPException(status_code=400, detail="잘못된 주소 형식입니다")
        
        # 트랜잭션 유효성 검증
        for i, tx in enumerate(transactions):
            validation = validate_transaction(tx)
            if not validation.is_valid:
                raise HTTPException(
                    status_code=400, 
                    detail=f"트랜잭션 {i} 유효성 검증 실패: {', '.join(validation.errors)}"
                )
        
        # 고급 피처 추출
        features = extract_advanced_features(
            transactions=transactions,
            address=request.address,
            window_hours=request.window_hours
        )
        
        # 리스크 스코어 계산
        score_result = calculate_risk_score(
            features=features,
            policy_path=request.policy_path,
            transactions=transactions,
            address=request.address,
            use_anomaly_detection=True
        )
        
        # 응답 생성
        return ScoreResponse(
            address=request.address,
            risk_score=score_result['risk_score'],
            risk_level=score_result['risk_level'],
            weighted_score=score_result['weighted_score'],
            applied_rules=score_result['applied_rules'],
            feature_contributions=score_result['feature_contributions'],
            features=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스코어 계산 오류: {str(e)}")


@app.post("/score/batch")
async def score_addresses_batch(addresses: List[str], 
                              transactions: List[TransactionData],
                              window_hours: Optional[int] = 24,
                              policy_path: Optional[str] = None):
    """
    여러 주소의 리스크 스코어를 일괄 계산합니다.
    
    Args:
        addresses: 분석할 주소 리스트
        transactions: 트랜잭션 리스트
        window_hours: 분석 시간 윈도우
        policy_path: 정책 파일 경로
    
    Returns:
        각 주소별 리스크 스코어 결과
    """
    results = []
    
    # 트랜잭션 데이터를 딕셔너리 형태로 변환
    tx_dicts = [tx.dict() for tx in transactions]
    for tx in tx_dicts:
        tx['from'] = tx.pop('from_address')
        tx['to'] = tx.pop('to_address')
    
    for address in addresses:
        try:
            request = ScoreRequest(
                address=address,
                transactions=transactions,
                window_hours=window_hours,
                policy_path=policy_path
            )
            
            result = await score_address(request)
            results.append(result)
            
        except Exception as e:
            results.append({
                "address": address,
                "error": str(e)
            })
    
    return {"results": results}


@app.post("/analyze/summary")
async def create_analysis_summary(request: ScoreRequest):
    """
    종합 분석 보고서를 생성합니다.
    
    Args:
        request: 분석 요청 데이터
    
    Returns:
        종합 분석 보고서
    """
    try:
        # 트랜잭션 데이터를 딕셔너리 형태로 변환
        transactions = [tx.dict() for tx in request.transactions]
        
        # 트랜잭션 데이터에서 from/to 키 매핑
        for tx in transactions:
            tx['from'] = tx.pop('from_address')
            tx['to'] = tx.pop('to_address')
        
        # 주소 유효성 검증
        if not validate_address(request.address):
            raise HTTPException(status_code=400, detail="잘못된 주소 형식입니다")
        
        # 피처 추출
        features = extract_advanced_features(
            transactions=transactions,
            address=request.address,
            window_hours=request.window_hours
        )
        
        # 리스크 스코어 계산
        score_result = calculate_risk_score(
            features=features,
            policy_path=request.policy_path,
            transactions=transactions,
            address=request.address,
            use_anomaly_detection=True
        )
        
        # 종합 보고서 생성
        summary_report = create_summary_report(
            transactions=transactions,
            address=request.address,
            features=features,
            risk_score=score_result
        )
        
        return summary_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 보고서 생성 오류: {str(e)}")


@app.get("/models/status")
async def get_model_status():
    """모델들의 상태를 확인합니다."""
    return {
        "models": {
            "rule_based": "available",
            "unsupervised_anomaly": "available",
            "supervised_anomaly": "not_trained"
        },
        "version": "0.2.0",
        "last_updated": "2024-01-01"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
