#!/usr/bin/env python3
"""
Excel 데이터 처리 스크립트

백엔드에서 받은 xlsx 파일을 AML 리스크 엔진에서 사용할 수 있는 형태로 변환합니다.
"""

from pathlib import Path
from src.riskengine.utils import (
    load_excel_data, 
    convert_excel_to_standard_format, 
    save_processed_data,
    preview_excel_data
)
from src.riskengine import calculate_risk_score, extract_advanced_features

def main():
    # 1. Excel 파일 경로 설정
    transactions_file = "data/transactions.xlsx"
    blocks_file = "data/blocks.xlsx"
    
    print("🚀 Excel 데이터 처리 시작!")
    
    # 2. Excel 파일들 미리보기
    print("\n" + "="*50)
    print("📖 Excel 파일들 미리보기")
    print("="*50)
    
    try:
        print("📁 Transactions 파일:")
        preview_excel_data(transactions_file, n_rows=3)
        print("\n📁 Blocks 파일:")
        preview_excel_data(blocks_file, n_rows=1)
    except Exception as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return
    
    # 3. Excel 데이터 로드
    print("\n" + "="*50)
    print("📥 Excel 데이터 로딩")
    print("="*50)
    
    # 트랜잭션 데이터 로드 (헤더 문제 해결)
    import pandas as pd
    df_tx = pd.read_excel(transactions_file, header=0)
    
    # 첫 번째 행이 실제 헤더인 경우 처리
    if df_tx.iloc[0, 0] == 'hash':
        new_headers = df_tx.iloc[0].values
        df_tx = df_tx[1:].copy()
        df_tx.columns = new_headers
    
    raw_data = df_tx.to_dict('records')
    print(f"✅ {len(raw_data)}개의 트랜잭션을 로드했습니다.")
    
    # 4. 표준 형식으로 변환
    print("\n" + "="*50)
    print("🔄 데이터 형식 변환")
    print("="*50)
    
    # 백엔드 데이터에 맞는 컬럼 매핑
    column_mapping = {
        'from_address': 'from',
        'to_address': 'to',
        'value': 'value',
        'gas': 'gas',
        'block_timestamp': 'timestamp',
        'hash': 'hash',
        'nonce': 'nonce',
        'transaction_type': 'transaction_type'
    }
    
    transactions = convert_excel_to_standard_format(raw_data, column_mapping)
    print(f"✅ {len(transactions)}개의 유효한 트랜잭션으로 변환했습니다.")
    
    if len(transactions) == 0:
        print("⚠️  변환된 트랜잭션이 없습니다. 컬럼 매핑을 확인해주세요.")
        print("📋 Excel 컬럼명:", list(raw_data[0].keys()) if raw_data else "데이터 없음")
        return
    
    # 5. 변환된 데이터 저장
    print("\n" + "="*50)
    print("💾 처리된 데이터 저장")
    print("="*50)
    
    save_processed_data(transactions, "data/processed/transactions.jsonl", "jsonl")
    save_processed_data(transactions, "data/processed/transactions.json", "json")
    
    # 6. 샘플 분석 실행
    print("\n" + "="*50)
    print("🔍 샘플 리스크 분석")
    print("="*50)
    
    if transactions:
        # 첫 번째 트랜잭션의 from 주소로 분석
        sample_address = transactions[0].get('from')
        if sample_address:
            print(f"📍 분석 대상 주소: {sample_address}")
            
            try:
                # 고급 피처 추출
                features = extract_advanced_features(transactions, sample_address)
                print(f"✅ {len(features)}개의 피처를 추출했습니다.")
                
                # 리스크 스코어 계산
                result = calculate_risk_score(
                    features=features,
                    transactions=transactions,
                    address=sample_address,
                    use_anomaly_detection=True
                )
                
                print(f"\n📊 리스크 분석 결과:")
                print(f"  🎯 리스크 점수: {result['risk_score']:.3f}")
                print(f"  📈 리스크 레벨: {result['risk_level']}")
                print(f"  🔧 적용된 룰: {len(result['applied_rules'])}개")
                
            except Exception as e:
                print(f"❌ 분석 실패: {e}")
    
    print("\n🎉 데이터 처리 완료!")
    print(f"📁 처리된 파일:")
    print(f"  - data/processed/transactions.jsonl")
    print(f"  - data/processed/transactions.json")

if __name__ == "__main__":
    main()
