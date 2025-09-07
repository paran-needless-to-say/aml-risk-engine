#!/usr/bin/env python3
"""
AML 리스크 엔진 실험 실행 스크립트

백엔드 데이터로 다양한 주소들에 대해 리스크 분석을 실행하고 결과를 저장합니다.
"""

import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import time

def load_processed_data() -> List[Dict[str, Any]]:
    """처리된 트랜잭션 데이터를 로드합니다."""
    with open('data/processed/transactions.json', 'r') as f:
        return json.load(f)

def get_unique_addresses(transactions: List[Dict[str, Any]]) -> List[str]:
    """트랜잭션에서 고유 주소들을 추출합니다."""
    addresses = set()
    for tx in transactions:
        if tx.get('from'):
            addresses.add(tx['from'])
        if tx.get('to'):
            addresses.add(tx['to'])
    return list(addresses)

def create_api_request(address: str, transactions: List[Dict[str, Any]], 
                      window_hours: int = 24) -> Dict[str, Any]:
    """API 요청 데이터를 생성합니다."""
    # 해당 주소와 관련된 트랜잭션만 필터링
    relevant_txs = [
        tx for tx in transactions 
        if tx.get('from') == address or tx.get('to') == address
    ]
    
    # API 형식으로 변환
    api_transactions = []
    for tx in relevant_txs[:10]:  # 최대 10개만
        api_tx = {
            'from_address': tx.get('from', ''),
            'to_address': tx.get('to', ''),
            'value': float(tx.get('value', 0)),
            'timestamp': int(tx.get('timestamp', 0)),
            'gas': float(tx.get('gas', 21000)),
            'gasPrice': 20000000000  # 기본값
        }
        api_transactions.append(api_tx)
    
    return {
        'address': address,
        'transactions': api_transactions,
        'window_hours': window_hours
    }

def call_risk_api(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """리스크 분석 API를 호출합니다."""
    try:
        response = requests.post(
            'http://localhost:8000/score',
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                'success': True,
                'result': response.json()
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_experiment_results(results: List[Dict[str, Any]], 
                          experiment_name: str = None) -> str:
    """실험 결과를 저장합니다."""
    if not experiment_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"experiment_{timestamp}"
    
    # 결과 디렉토리 생성
    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # JSON 형태로 저장
    json_file = results_dir / f"{experiment_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # CSV 형태로 요약 저장
    csv_file = results_dir / f"{experiment_name}_summary.csv"
    summary_data = []
    
    for result in results:
        if result['api_result']['success']:
            api_result = result['api_result']['result']
            summary_data.append({
                'address': result['address'],
                'transaction_count': result['transaction_count'],
                'risk_score': api_result.get('risk_score', 0),
                'risk_level': api_result.get('risk_level', 'UNKNOWN'),
                'applied_rules_count': len(api_result.get('applied_rules', [])),
                'features_count': len(api_result.get('features', {})),
                'success': True
            })
        else:
            summary_data.append({
                'address': result['address'],
                'transaction_count': result['transaction_count'],
                'risk_score': 0,
                'risk_level': 'ERROR',
                'applied_rules_count': 0,
                'features_count': 0,
                'success': False,
                'error': result['api_result']['error']
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    
    return str(json_file)

def run_batch_experiment(max_addresses: int = 10) -> None:
    """배치 실험을 실행합니다."""
    print("🚀 AML 리스크 엔진 배치 실험 시작!")
    print("="*60)
    
    # 1. 데이터 로드
    print("📥 트랜잭션 데이터 로딩...")
    transactions = load_processed_data()
    print(f"✅ {len(transactions)}개의 트랜잭션 로드됨")
    
    # 2. 고유 주소 추출
    print("📍 고유 주소 추출...")
    addresses = get_unique_addresses(transactions)
    print(f"✅ {len(addresses)}개의 고유 주소 발견")
    
    # 3. 분석할 주소 선택
    test_addresses = addresses[:max_addresses]
    print(f"🎯 {len(test_addresses)}개 주소 분석 예정")
    
    # 4. 각 주소별 분석 실행
    results = []
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n[{i}/{len(test_addresses)}] 분석 중: {address[:10]}...")
        
        # API 요청 데이터 생성
        request_data = create_api_request(address, transactions)
        
        if not request_data['transactions']:
            print(f"  ⚠️  관련 트랜잭션 없음")
            continue
        
        print(f"  📊 관련 트랜잭션: {len(request_data['transactions'])}개")
        
        # API 호출
        api_result = call_risk_api(request_data)
        
        # 결과 저장
        result_data = {
            'address': address,
            'transaction_count': len(request_data['transactions']),
            'request_data': request_data,
            'api_result': api_result,
            'timestamp': datetime.now().isoformat()
        }
        
        if api_result['success']:
            risk_data = api_result['result']
            print(f"  🎯 리스크 점수: {risk_data.get('risk_score', 0):.3f}")
            print(f"  📈 리스크 레벨: {risk_data.get('risk_level', 'UNKNOWN')}")
            print(f"  🔧 적용 룰: {len(risk_data.get('applied_rules', []))}개")
        else:
            print(f"  ❌ 분석 실패: {api_result['error']}")
        
        results.append(result_data)
        
        # API 과부하 방지
        time.sleep(0.5)
    
    # 5. 결과 저장
    print(f"\n💾 실험 결과 저장 중...")
    result_file = save_experiment_results(results)
    print(f"✅ 결과 저장 완료: {result_file}")
    
    # 6. 요약 통계
    print(f"\n📊 실험 결과 요약:")
    print("="*40)
    
    successful = [r for r in results if r['api_result']['success']]
    failed = [r for r in results if not r['api_result']['success']]
    
    print(f"✅ 성공: {len(successful)}개")
    print(f"❌ 실패: {len(failed)}개")
    
    if successful:
        risk_scores = [r['api_result']['result']['risk_score'] for r in successful]
        risk_levels = [r['api_result']['result']['risk_level'] for r in successful]
        
        print(f"📈 평균 리스크 점수: {sum(risk_scores)/len(risk_scores):.3f}")
        print(f"📊 리스크 레벨 분포:")
        
        from collections import Counter
        level_counts = Counter(risk_levels)
        for level, count in level_counts.items():
            print(f"  {level}: {count}개")

def run_single_experiment(address: str = None) -> None:
    """단일 주소 실험을 실행합니다."""
    print("🔍 단일 주소 리스크 분석 실험")
    print("="*40)
    
    transactions = load_processed_data()
    
    if not address:
        # 첫 번째 주소 사용
        address = transactions[0]['from']
    
    print(f"📍 분석 주소: {address}")
    
    request_data = create_api_request(address, transactions)
    print(f"📊 관련 트랜잭션: {len(request_data['transactions'])}개")
    
    api_result = call_risk_api(request_data)
    
    if api_result['success']:
        result = api_result['result']
        
        print(f"\n🎯 분석 결과:")
        print(f"  리스크 점수: {result['risk_score']:.3f}")
        print(f"  리스크 레벨: {result['risk_level']}")
        print(f"  적용된 룰: {len(result['applied_rules'])}개")
        print(f"  추출된 피처: {len(result['features'])}개")
        
        # 상세 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"single_experiment_{timestamp}.json"
        
        experiment_data = {
            'address': address,
            'request_data': request_data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 상세 결과 저장: {filename}")
    else:
        print(f"❌ 분석 실패: {api_result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            max_addr = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            run_batch_experiment(max_addr)
        elif sys.argv[1] == "single":
            addr = sys.argv[2] if len(sys.argv) > 2 else None
            run_single_experiment(addr)
        else:
            print("사용법: python run_experiments.py [batch|single] [옵션]")
    else:
        # 기본: 단일 실험
        run_single_experiment()
