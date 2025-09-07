"""
유틸리티 모듈

공통으로 사용되는 헬퍼 함수들과 데이터 처리 유틸리티를 제공합니다.
"""

import json
import yaml
import hashlib
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """검증 결과를 담는 데이터 클래스"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def validate_address(address: str) -> bool:
    """
    이더리움 주소 형식을 검증합니다.
    
    Args:
        address: 검증할 주소
        
    Returns:
        유효한 주소인지 여부
    """
    if not isinstance(address, str):
        return False
    
    # 기본 형식 검증
    if not address.startswith('0x'):
        return False
    
    if len(address) != 42:  # 0x + 40자
        return False
    
    # 16진수 문자만 포함하는지 확인
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False


def validate_transaction(transaction: Dict[str, Any]) -> ValidationResult:
    """
    트랜잭션 데이터의 유효성을 검증합니다.
    
    Args:
        transaction: 검증할 트랜잭션 데이터
        
    Returns:
        검증 결과
    """
    errors = []
    warnings = []
    
    # 필수 필드 확인
    required_fields = ['from', 'to', 'value']
    for field in required_fields:
        if field not in transaction:
            errors.append(f"필수 필드 누락: {field}")
        elif transaction[field] is None:
            errors.append(f"필수 필드가 None: {field}")
    
    # 주소 형식 검증
    if 'from' in transaction and transaction['from']:
        if not validate_address(transaction['from']):
            errors.append(f"잘못된 from 주소 형식: {transaction['from']}")
    
    if 'to' in transaction and transaction['to']:
        if not validate_address(transaction['to']):
            errors.append(f"잘못된 to 주소 형식: {transaction['to']}")
    
    # 값 유효성 검증
    if 'value' in transaction:
        try:
            value = float(transaction['value'])
            if value < 0:
                errors.append("거래 금액은 음수일 수 없습니다")
        except (ValueError, TypeError):
            errors.append(f"잘못된 value 형식: {transaction['value']}")
    
    # 가스 관련 필드 검증
    if 'gas' in transaction and transaction['gas'] is not None:
        try:
            gas = float(transaction['gas'])
            if gas <= 0:
                warnings.append("가스 한도가 0 이하입니다")
        except (ValueError, TypeError):
            warnings.append(f"잘못된 gas 형식: {transaction['gas']}")
    
    if 'gasPrice' in transaction and transaction['gasPrice'] is not None:
        try:
            gas_price = float(transaction['gasPrice'])
            if gas_price <= 0:
                warnings.append("가스 가격이 0 이하입니다")
        except (ValueError, TypeError):
            warnings.append(f"잘못된 gasPrice 형식: {transaction['gasPrice']}")
    
    # 타임스탬프 검증
    if 'timestamp' in transaction and transaction['timestamp'] is not None:
        try:
            timestamp = int(transaction['timestamp'])
            # 합리적인 범위 확인 (2009년 이후, 현재 시간 이전)
            if timestamp < 1230768000:  # 2009-01-01
                warnings.append("타임스탬프가 너무 이릅니다")
            elif timestamp > int(time.time()) + 3600:  # 현재 시간 + 1시간
                warnings.append("타임스탬프가 미래입니다")
        except (ValueError, TypeError):
            warnings.append(f"잘못된 timestamp 형식: {transaction['timestamp']}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def normalize_address(address: str) -> str:
    """
    주소를 정규화합니다 (소문자 변환).
    
    Args:
        address: 정규화할 주소
        
    Returns:
        정규화된 주소
    """
    if not isinstance(address, str):
        return ""
    
    return address.lower().strip()


def calculate_transaction_hash(transaction: Dict[str, Any]) -> str:
    """
    트랜잭션의 해시를 계산합니다.
    
    Args:
        transaction: 트랜잭션 데이터
        
    Returns:
        트랜잭션 해시
    """
    # 해시 계산에 사용할 필드들
    hash_fields = ['from', 'to', 'value', 'gas', 'gasPrice', 'timestamp']
    
    # 정렬된 필드값들로 문자열 생성
    hash_data = []
    for field in hash_fields:
        if field in transaction:
            hash_data.append(f"{field}:{transaction[field]}")
    
    hash_string = "|".join(hash_data)
    
    # SHA-256 해시 계산
    return hashlib.sha256(hash_string.encode()).hexdigest()


def convert_wei_to_ether(wei_value: Union[int, float, str]) -> float:
    """
    Wei를 Ether로 변환합니다.
    
    Args:
        wei_value: Wei 단위 값
        
    Returns:
        Ether 단위 값
    """
    try:
        wei = float(wei_value)
        return wei / 1e18
    except (ValueError, TypeError):
        return 0.0


def convert_ether_to_wei(ether_value: Union[int, float, str]) -> int:
    """
    Ether를 Wei로 변환합니다.
    
    Args:
        ether_value: Ether 단위 값
        
    Returns:
        Wei 단위 값
    """
    try:
        ether = float(ether_value)
        return int(ether * 1e18)
    except (ValueError, TypeError):
        return 0


def format_timestamp(timestamp: Union[int, float], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    타임스탬프를 포맷된 문자열로 변환합니다.
    
    Args:
        timestamp: Unix 타임스탬프
        format_str: 날짜 형식 문자열
        
    Returns:
        포맷된 날짜 문자열
    """
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime(format_str)
    except (ValueError, TypeError, OSError):
        return "Invalid timestamp"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    설정 파일을 로드합니다 (YAML 또는 JSON).
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {config_path.suffix}")
    except Exception as e:
        raise Exception(f"설정 파일 로드 실패: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    설정을 파일로 저장합니다.
    
    Args:
        config: 저장할 설정 딕셔너리
        config_path: 저장할 파일 경로
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {config_path.suffix}")
    except Exception as e:
        raise Exception(f"설정 파일 저장 실패: {e}")


def batch_process_transactions(transactions: List[Dict[str, Any]], 
                              batch_size: int = 1000) -> List[List[Dict[str, Any]]]:
    """
    트랜잭션들을 배치로 나눕니다.
    
    Args:
        transactions: 트랜잭션 리스트
        batch_size: 배치 크기
        
    Returns:
        배치별로 나뉜 트랜잭션 리스트들
    """
    batches = []
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        batches.append(batch)
    return batches


def calculate_statistics(values: List[Union[int, float]]) -> Dict[str, float]:
    """
    값들의 통계를 계산합니다.
    
    Args:
        values: 값 리스트
        
    Returns:
        통계 딕셔너리
    """
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    values = [float(v) for v in values if v is not None]
    
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q25': 0.0,
            'q75': 0.0
        }
    
    return {
        'count': len(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    }


def detect_outliers(values: List[Union[int, float]], 
                   method: str = 'iqr', 
                   threshold: float = 1.5) -> Tuple[List[int], List[Union[int, float]]]:
    """
    이상치를 탐지합니다.
    
    Args:
        values: 값 리스트
        method: 탐지 방법 ('iqr', 'zscore')
        threshold: 임계값
        
    Returns:
        (이상치 인덱스 리스트, 이상치 값 리스트)
    """
    if not values:
        return [], []
    
    values = np.array([float(v) for v in values if v is not None])
    
    if len(values) == 0:
        return [], []
    
    outlier_indices = []
    
    if method == 'iqr':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0].tolist()
        
    elif method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            outlier_indices = np.where(z_scores > threshold)[0].tolist()
    
    outlier_values = [values[i] for i in outlier_indices]
    
    return outlier_indices, outlier_values


def create_summary_report(transactions: List[Dict[str, Any]], 
                         address: str,
                         features: Dict[str, float],
                         risk_score: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 결과 요약 보고서를 생성합니다.
    
    Args:
        transactions: 트랜잭션 리스트
        address: 분석 대상 주소
        features: 추출된 피처들
        risk_score: 리스크 스코어 결과
        
    Returns:
        요약 보고서 딕셔너리
    """
    # 트랜잭션 통계
    values = [float(tx.get('value', 0)) for tx in transactions]
    value_stats = calculate_statistics(values)
    
    # 시간 분석
    timestamps = [tx.get('timestamp') for tx in transactions if tx.get('timestamp')]
    time_range = {
        'start': format_timestamp(min(timestamps)) if timestamps else "N/A",
        'end': format_timestamp(max(timestamps)) if timestamps else "N/A",
        'duration_hours': (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) >= 2 else 0
    }
    
    # 주소 분석
    counterparties = set()
    for tx in transactions:
        from_addr = normalize_address(tx.get('from', ''))
        to_addr = normalize_address(tx.get('to', ''))
        target_addr = normalize_address(address)
        
        if from_addr == target_addr and to_addr:
            counterparties.add(to_addr)
        elif to_addr == target_addr and from_addr:
            counterparties.add(from_addr)
    
    return {
        'analysis_summary': {
            'target_address': normalize_address(address),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'transaction_count': len(transactions),
            'unique_counterparties': len(counterparties),
            'time_range': time_range
        },
        'transaction_statistics': {
            'value_stats': value_stats,
            'total_volume': sum(values),
            'average_value': value_stats['mean'],
            'largest_transaction': value_stats['max']
        },
        'risk_assessment': {
            'risk_score': risk_score.get('risk_score', 0.0),
            'risk_level': risk_score.get('risk_level', 'UNKNOWN'),
            'confidence': 'HIGH' if len(transactions) >= 10 else 'LOW'
        },
        'key_features': {
            'transaction_frequency': features.get('centrality_tx_frequency', 0),
            'unique_addresses': features.get('centrality_unique_addresses', 0),
            'total_exposure': features.get('exposure_total_value', 0),
            'behavioral_score': features.get('behavior_repeat_tx_ratio', 0)
        },
        'alerts': _generate_alerts(features, risk_score),
        'recommendations': _generate_recommendations(risk_score)
    }


def _generate_alerts(features: Dict[str, float], risk_score: Dict[str, Any]) -> List[str]:
    """경고 메시지를 생성합니다."""
    alerts = []
    
    if risk_score.get('risk_level') == 'HIGH':
        alerts.append("🚨 높은 리스크 수준이 탐지되었습니다")
    
    if features.get('centrality_tx_frequency', 0) > 100:
        alerts.append("⚠️ 고빈도 거래 패턴이 감지되었습니다")
    
    if features.get('exposure_max_tx_value', 0) > 1000000:
        alerts.append("💰 대액 거래가 감지되었습니다")
    
    if features.get('centrality_unique_addresses', 0) > 50:
        alerts.append("🔗 다수의 주소와 연결되어 있습니다")
    
    return alerts


def _generate_recommendations(risk_score: Dict[str, Any]) -> List[str]:
    """권장사항을 생성합니다."""
    recommendations = []
    
    risk_level = risk_score.get('risk_level', 'UNKNOWN')
    
    if risk_level == 'HIGH':
        recommendations.extend([
            "추가 조사가 필요합니다",
            "관련 거래 내역을 상세히 검토하세요",
            "규제 당국에 보고를 고려하세요"
        ])
    elif risk_level == 'MEDIUM':
        recommendations.extend([
            "지속적인 모니터링이 필요합니다",
            "거래 패턴 변화를 주시하세요"
        ])
    elif risk_level == 'LOW':
        recommendations.append("정상적인 거래 패턴으로 보입니다")
    
    return recommendations


def load_excel_data(file_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Excel 파일에서 트랜잭션 데이터를 로드합니다.
    
    Args:
        file_path: Excel 파일 경로
        sheet_name: 시트명 (None이면 첫 번째 시트)
    
    Returns:
        트랜잭션 딕셔너리 리스트
    """
    try:
        # Excel 파일 읽기
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 컬럼명을 소문자로 변환 및 공백 제거
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # NaN 값 처리
        df = df.fillna('')
        
        # 딕셔너리 리스트로 변환
        transactions = df.to_dict('records')
        
        return transactions
        
    except Exception as e:
        raise Exception(f"Excel 파일 로드 실패: {e}")


def convert_excel_to_standard_format(transactions: List[Dict[str, Any]], 
                                   column_mapping: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Excel에서 로드한 데이터를 표준 트랜잭션 형식으로 변환합니다.
    
    Args:
        transactions: Excel에서 로드한 원본 데이터
        column_mapping: 컬럼 매핑 딕셔너리 {'excel_column': 'standard_column'}
    
    Returns:
        표준 형식으로 변환된 트랜잭션 리스트
    """
    # 기본 컬럼 매핑
    default_mapping = {
        'from': 'from',
        'from_address': 'from',
        'sender': 'from',
        'to': 'to', 
        'to_address': 'to',
        'receiver': 'to',
        'recipient': 'to',
        'value': 'value',
        'amount': 'value',
        'eth_value': 'value',
        'timestamp': 'timestamp',
        'time': 'timestamp',
        'date': 'timestamp',
        'gas': 'gas',
        'gas_limit': 'gas',
        'gas_price': 'gasPrice',
        'gasprice': 'gasPrice',
        'hash': 'hash',
        'tx_hash': 'hash',
        'transaction_hash': 'hash',
        'block_number': 'blockNumber',
        'block': 'blockNumber'
    }
    
    # 사용자 매핑이 있으면 업데이트
    if column_mapping:
        default_mapping.update(column_mapping)
    
    converted_transactions = []
    
    for tx in transactions:
        converted_tx = {}
        
        # 컬럼 매핑 적용
        for excel_col, standard_col in default_mapping.items():
            if excel_col in tx:
                value = tx[excel_col]
                
                # 데이터 타입 변환
                if standard_col in ['value', 'gas', 'gasPrice']:
                    try:
                        converted_tx[standard_col] = float(value) if value else 0.0
                    except (ValueError, TypeError):
                        converted_tx[standard_col] = 0.0
                        
                elif standard_col in ['timestamp', 'blockNumber']:
                    try:
                        if isinstance(value, str) and value:
                            # 날짜 문자열인 경우 타임스탬프로 변환
                            if '-' in str(value) or '/' in str(value):
                                dt = pd.to_datetime(value)
                                converted_tx[standard_col] = int(dt.timestamp())
                            else:
                                converted_tx[standard_col] = int(float(value))
                        else:
                            converted_tx[standard_col] = int(float(value)) if value else 0
                    except (ValueError, TypeError):
                        converted_tx[standard_col] = 0
                        
                else:
                    # 문자열 필드 (주소, 해시 등)
                    converted_tx[standard_col] = str(value).strip() if value else ""
        
        # 필수 필드 검증
        if converted_tx.get('from') and converted_tx.get('to'):
            converted_transactions.append(converted_tx)
    
    return converted_transactions


def save_processed_data(transactions: List[Dict[str, Any]], 
                       output_path: str, 
                       format: str = 'jsonl') -> None:
    """
    처리된 데이터를 파일로 저장합니다.
    
    Args:
        transactions: 저장할 트랜잭션 데이터
        output_path: 출력 파일 경로
        format: 저장 형식 ('jsonl', 'json', 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'jsonl':
            import jsonlines
            with jsonlines.open(output_path, mode='w') as writer:
                for tx in transactions:
                    writer.write(tx)
                    
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transactions, f, indent=2, ensure_ascii=False)
                
        elif format == 'csv':
            df = pd.DataFrame(transactions)
            df.to_csv(output_path, index=False)
            
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
            
        print(f"데이터가 저장되었습니다: {output_path}")
        
    except Exception as e:
        raise Exception(f"데이터 저장 실패: {e}")


def preview_excel_data(file_path: str, sheet_name: Optional[str] = None, n_rows: int = 5) -> None:
    """
    Excel 파일의 데이터를 미리보기합니다.
    
    Args:
        file_path: Excel 파일 경로
        sheet_name: 시트명
        n_rows: 표시할 행 수
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        print(f"📊 Excel 파일 정보: {file_path}")
        print(f"📏 데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
        print(f"📋 컬럼명: {list(df.columns)}")
        print(f"\n📖 상위 {n_rows}행 미리보기:")
        print(df.head(n_rows).to_string())
        
        # 결측값 확인
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  결측값:")
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col}: {count}개")
    
    except Exception as e:
        print(f"❌ 파일 미리보기 실패: {e}")
