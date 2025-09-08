"""
리스크 스코어링 모듈

룰 기반 점수와 이상행동 탐지 결과를 집계하여 최종 risk_score를 계산합니다.
"""

# ML 모델들과 피처 추출함수 가져옴
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import numpy as np

from .detectors.unsupervised import UnsupervisedAnomalyDetector, create_default_unsupervised_detector
from .detectors.supervised import SupervisedAnomalyDetector, create_default_supervised_detector
from .detectors.features import extract_advanced_features

# RiskScorer 클래스 정의
class RiskScorer:
    """리스크 스코어 계산기"""
    
    def __init__(self, 
                 policy_path: Optional[str] = None,
                 unsupervised_detector: Optional[UnsupervisedAnomalyDetector] = None,
                 supervised_detector: Optional[SupervisedAnomalyDetector] = None):
        """
        Args:
            policy_path: 정책 파일 경로. None이면 기본 정책 사용
            unsupervised_detector: 비지도 학습 탐지기
            supervised_detector: 지도 학습 탐지기
        """
        if policy_path is None:
            # 기본 정책 파일 경로
            policy_path = Path(__file__).parent / "rules" / "default_light.yaml" # 이 기본 정책 파일에서 룰들을 읽어옴
        
        self.policy = self._load_policy(policy_path) # 정책 파일을 로딩함
        self.weights = self.policy.get('feature_weights', {}) # 피처별 가중치
        self.rules = self.policy.get('rules', []) # 룰들
        self.thresholds = self.policy.get('thresholds', {}) # 리스크 레벨 임계값
        self.anomaly_weights = self.policy.get('anomaly_weights', {}) # 이상행동 탐지기들의 가중치
        
        # 이상행동 탐지기들
        self.unsupervised_detector = unsupervised_detector
        self.supervised_detector = supervised_detector
    
    def _load_policy(self, policy_path: str) -> Dict[str, Any]:
        """정책 파일을 로딩합니다."""
        policy_path = Path(policy_path)
        
        if not policy_path.exists():
            # 기본 정책 반환
            return self._get_default_policy()
        
        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"정책 파일 로딩 실패: {e}. 기본 정책을 사용합니다.")
            return self._get_default_policy()
    
    def _get_default_policy(self) -> Dict[str, Any]:
        """기본 정책을 반환합니다."""
        return {
            'version': 'v1.0',
            'name': 'rulebook_light_mvp',
            'thresholds': {
                'risk_score': {
                    'low': 30,
                    'medium': 50,
                    'high': 70,
                    'critical': 90
                }
            },
            'aggregate': {
                'method': 'weighted_sum',
                'normalize': True
            },
            'axes': {
                'compliance': {
                    'C-001': {
                        'name': '제재 대상 직접 접촉',
                        'condition': 'sanctions.contains(counterparty) and usd_amount >= 1',
                        'score': 80,
                        'description': 'OFAC/UN/EU 제재 명단 위반'
                    }
                },
                'exposure': {
                    'E-101': {
                        'name': '믹서 직접 사용',
                        'condition': 'mixer.used and usd_amount >= 20',
                        'weight': 0.8,
                        'description': '자금 추적 어려움'
                    }
                },
                'behavior': {
                    'tempo': {
                        'B-101': {
                            'name': '버스트 거래 (10분)',
                            'condition': 'tx_count(10m) >= 3',
                            'score': 15,
                            'description': '비정상적 빈도'
                        },
                        'B-102': {
                            'name': '급속 연속 (1분)',
                            'condition': 'tx_count(1m) >= 5',
                            'score': 20,
                            'description': '매우 의심스러운 패턴'
                        }
                    },
                    'value': {
                        'B-501': {
                            'name': '고액 거래',
                            'condition': 'usd_amount >= 10000',
                            'score': 15,
                            'description': '고액 경고'
                        }
                    }
                }
            },
            'anomaly_weights': {
                'isolation_forest': 0.3,
                'local_outlier_factor': 0.2,
                'xgboost': 0.3,
                'lightgbm': 0.2
            }
        }
    


    ## ===== 메인 리스크 스코어 계산 함수 ===== ##
    def calculate_risk_score(self, 
                           features: Dict[str, float],
                           transactions: Optional[List[Dict[str, Any]]] = None,
                           address: Optional[str] = None) -> Dict[str, Any]:
        """
        🎯 메인 함수: 5단계 파이프라인으로 리스크 스코어를 계산합니다.
        
        전체 흐름:
        피처 가중합 → 룰 평가 → AI 이상탐지 → 점수 집계 → 등급 분류
        
        Args:
            features: 추출된 피처 딕셔너리 (예: {'거래량': 1000, '빈도': 50})
            transactions: 트랜잭션 리스트 (고급 AI 분석용, 선택사항)
            address: 분석 대상 주소 (네트워크 분석용, 선택사항)
        
        Returns:
            리스크 스코어 결과 (점수, 등급, 상세 분석 포함)
        """
        # === 1단계: 기본 피처들의 가중합 계산 ===
        # 각 피처(거래량, 빈도 등)에 중요도 가중치를 곱해서 기본 점수 산출
        # 예: 거래량(0.5) × 가중치(0.3) + 빈도(0.8) × 가중치(0.2) = 0.31점
        weighted_score = self._calculate_weighted_score(features)
        
        # === 2단계: C/E/B 축별 룰북 기반 위험 조건 평가 ===
        # C축: 법적 위반 점수 가산 (80점, 25점 등)
        # E축: 위험 노출 가중치 적용 (0.8배, 0.9배 등)
        # B축: 행동 패턴 점수 가산 (15점, 20점 등)
        rule_results = self._evaluate_rules(features)
        
        # === 3단계: AI 기반 이상행동 패턴 탐지 ===
        # IsolationForest, XGBoost 등의 머신러닝 모델로 비정상 패턴 감지
        # 정상적이지 않은 거래 패턴을 자동으로 찾아냄
        anomaly_scores = self._calculate_anomaly_scores(features, transactions, address)
        
        # === 4단계: 모든 점수들을 하나로 합치기 ===
        # 룰 기반 점수(70%) + AI 이상탐지 점수(30%)로 최종 점수 계산
        final_score = self._aggregate_scores(weighted_score, rule_results, anomaly_scores)
        
        # === 5단계: 최종 점수를 4단계 등급으로 분류 ===
        # 0-1점 → VERY_LOW/LOW/MEDIUM/HIGH/CRITICAL 등급으로 변환
        risk_level = self._determine_risk_level(final_score)
        
        return {
            'risk_score': final_score,
            'risk_level': risk_level,
            'weighted_score': weighted_score,
            'anomaly_scores': anomaly_scores,
            'applied_rules': rule_results,
            'feature_contributions': self._calculate_feature_contributions(features),
            'score_breakdown': {
                'rule_based': weighted_score,
                'anomaly_detection': anomaly_scores.get('combined_score', 0.0),
                'rule_multipliers': sum(rule['risk_multiplier'] for rule in rule_results)
            }
        }
    
    def _calculate_weighted_score(self, features: Dict[str, float]) -> float:
        """
        📊 1단계: 피처들의 가중합을 계산합니다.
        
        동작 원리:
        1. 각 피처를 0-1 범위로 정규화 (공정한 비교를 위해)
        2. 정규화된 값에 중요도 가중치를 곱함
        3. 모든 가중 점수를 합쳐서 가중 평균 계산
        
        예시:
        - 거래량: 1000 ETH → 정규화: 0.5 → 가중치 0.3 곱하기 → 0.15
        - 거래빈도: 50회 → 정규화: 0.8 → 가중치 0.2 곱하기 → 0.16
        - 최종: (0.15 + 0.16) / (0.3 + 0.2) = 0.62점
        """
        score = 0.0          # 누적 점수
        total_weight = 0.0   # 총 가중치
        
        # YAML에서 설정한 각 피처별 가중치를 순회
        for feature_name, weight in self.weights.items():
            if feature_name in features:  # 해당 피처가 실제 데이터에 있는지 확인
                # 피처 값을 0-1 범위로 정규화 (거래량 1000 ETH → 0.5 같은 식)
                normalized_value = self._normalize_feature_value(feature_name, features[feature_name])
                # 정규화된 값 × 가중치를 누적
                score += normalized_value * weight
                total_weight += weight
        
        # 가중 평균 계산: 총점 ÷ 총가중치
        return score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_feature_value(self, feature_name: str, value: float) -> float:
        """피처 값을 0-1 범위로 정규화합니다."""
        # 간단한 시그모이드 정규화
        if value <= 0:
            return 0.0
        
        # 피처별 정규화 파라미터 (실제로는 데이터 기반으로 조정)
        if 'frequency' in feature_name or 'addresses' in feature_name:
            return min(1.0, value / 100.0)
        elif 'value' in feature_name:
            return min(1.0, np.log1p(value) / 20.0)
        elif 'ratio' in feature_name:
            return min(1.0, value)
        else:
            return min(1.0, value / 10.0)
    
    def _evaluate_rules(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        ⚖️ 2단계: C/E/B 축별 룰북 기반 위험 조건들을 평가합니다.
        
        C/E/B 축별 처리 방식:
        - C축 (Compliance): 법적 규제 위반 → score 가산 (80점, 25점 등)
        - E축 (Exposure): 위험 노출도 → weight 적용 (0.8배, 0.9배 등)
        - B축 (Behavior): 행동 패턴 → score 가산 (15점, 20점 등)
        
        반환 값:
        {
            'C_score': 80,           # C축 누적 점수
            'E_weight': 0.8,         # E축 누적 가중치
            'B_score': 35,           # B축 누적 점수
            'applied_rules': [...],   # 적용된 룰 상세 내역
        }
        """
        # C/E/B 축별 점수/가중치 초기화
        C_score = 0      # Compliance: 점수 누적
        E_weight = 1.0   # Exposure: 가중치 곱셈 (기본 1.0)
        B_score = 0      # Behavior: 점수 누적
        applied_rules = []  # 적용된 룰 내역
        
        # YAML 파일에서 axes 섹션 가져오기
        axes = self.policy.get('axes', {})
        
        # 각 축별로 룰 평가 및 C/E/B 처리
        for axis_name, axis_rules in axes.items():
            axis_applied_rules, axis_score, axis_weight = self._evaluate_axis_rules_ceb(
                axis_name, axis_rules, features)
            
            # 적용된 룰들 추가
            applied_rules.extend(axis_applied_rules)
            
            # 축별 점수/가중치 집계
            if axis_name == 'compliance':
                C_score += axis_score
            elif axis_name == 'exposure':
                # E축: soft scaling으로 극단적 감점 방지
                # 기존: 0.9 × 0.8 × 0.7 = 0.504 (극단적 감점)
                # 개선: exp(log(0.9) + log(0.8) + log(0.7)) = 더 부드러운 감점
                if axis_weight < 1.0:  # 위험 가중치인 경우만 적용
                    import math
                    log_weight = math.log(axis_weight) if axis_weight > 0 else -10
                    if not hasattr(self, '_exposure_log_sum'):
                        self._exposure_log_sum = 0
                    self._exposure_log_sum += log_weight
                else:
                    E_weight *= axis_weight
            elif axis_name == 'behavior':
                # B축: log scaling으로 동일 패턴 반복시 과잉 알람 방지
                # 기존: 15 + 15 + 15 = 45점 (선형 증가)
                # 개선: log(1 + 15) + log(1 + 15) + log(1 + 15) = 더 부드러운 증가
                if axis_score > 0:
                    import math
                    log_scaled_score = math.log(1 + axis_score) * 10  # 10배 스케일링으로 적절한 범위 유지
                    B_score += log_scaled_score
                else:
                    B_score += axis_score
        
        # E축 soft scaling 최종 계산
        if hasattr(self, '_exposure_log_sum') and self._exposure_log_sum != 0:
            import math
            E_weight = math.exp(self._exposure_log_sum)
            # 계산 후 초기화
            delattr(self, '_exposure_log_sum')
        
        return {
            'C_score': C_score,
            'E_weight': E_weight, 
            'B_score': B_score,
            'applied_rules': applied_rules
        }
    
    def _evaluate_axis_rules(self, axis_name: str, axis_rules: Dict[str, Any], features: Dict[str, float]) -> List[Dict[str, Any]]:
        """특정 축의 룰들을 평가합니다."""
        applied_rules = []
        
        for rule_id, rule_data in axis_rules.items():
            # 중첩된 구조 처리 (behavior.tempo.B-101 같은)
            if isinstance(rule_data, dict) and 'condition' not in rule_data:
                # 하위 카테고리가 있는 경우
                for sub_rule_id, sub_rule_data in rule_data.items():
                    if isinstance(sub_rule_data, dict) and 'condition' in sub_rule_data:
                        applied_rules.extend(self._evaluate_single_rule(f"{rule_id}.{sub_rule_id}", sub_rule_data, features))
            else:
                # 직접적인 룰인 경우
                if isinstance(rule_data, dict) and 'condition' in rule_data:
                    applied_rules.extend(self._evaluate_single_rule(rule_id, rule_data, features))
        
        return applied_rules
    
    def _evaluate_axis_rules_ceb(self, axis_name: str, axis_rules: Dict[str, Any], features: Dict[str, float]) -> tuple:
        """
        특정 C/E/B 축의 룰들을 평가하고 축별 점수/가중치를 계산합니다.
        
        Args:
            axis_name: 축 이름 ('compliance', 'exposure', 'behavior')
            axis_rules: 해당 축의 룰 데이터
            features: 피처 데이터
            
        Returns:
            (applied_rules, axis_score, axis_weight)
            - applied_rules: 적용된 룰 리스트
            - axis_score: C/B축의 경우 누적 점수, E축의 경우 0
            - axis_weight: E축의 경우 누적 가중치, C/B축의 경우 1.0
        """
        applied_rules = []
        axis_score = 0      # C/B축용 점수 누적
        axis_weight = 1.0   # E축용 가중치 누적
        
        for rule_id, rule_data in axis_rules.items():
            # 중첩된 구조 처리 (behavior.tempo.B-101 같은)
            if isinstance(rule_data, dict) and 'condition' not in rule_data:
                # 하위 카테고리가 있는 경우 (behavior 축의 tempo, topology 등)
                for sub_rule_id, sub_rule_data in rule_data.items():
                    if isinstance(sub_rule_data, dict) and 'condition' in sub_rule_data:
                        sub_applied_rules, sub_score, sub_weight = self._evaluate_single_rule_ceb(
                            f"{rule_id}.{sub_rule_id}", sub_rule_data, features, axis_name)
                        applied_rules.extend(sub_applied_rules)
                        axis_score += sub_score
                        axis_weight *= sub_weight
            else:
                # 직접적인 룰인 경우
                if isinstance(rule_data, dict) and 'condition' in rule_data:
                    rule_applied_rules, rule_score, rule_weight = self._evaluate_single_rule_ceb(
                        rule_id, rule_data, features, axis_name)
                    applied_rules.extend(rule_applied_rules)
                    axis_score += rule_score
                    axis_weight *= rule_weight
        
        return applied_rules, axis_score, axis_weight
    
    def _evaluate_single_rule_ceb(self, rule_id: str, rule_data: Dict[str, Any], features: Dict[str, float], axis_name: str) -> tuple:
        """
        C/E/B 축 기반으로 단일 룰을 평가합니다.
        
        Args:
            rule_id: 룰 ID
            rule_data: 룰 데이터 (condition, score, weight 등)
            features: 피처 데이터
            axis_name: 축 이름 ('compliance', 'exposure', 'behavior')
            
        Returns:
            (applied_rules, rule_score, rule_weight)
        """
        applied_rules = []
        rule_score = 0      # C/B축용 점수
        rule_weight = 1.0   # E축용 가중치
        
        condition = rule_data.get('condition', '')
        
        try:
            # 조건 평가: 조건에 맞는지 체크
            if self._evaluate_simple_condition(condition, features):
                # 적용된 룰 상세 정보 저장
                applied_rule = {
                    'rule_id': rule_id,
                    'name': rule_data.get('name', rule_id),
                    'description': rule_data.get('description', ''),
                    'condition': condition,
                    'axis': axis_name
                }
                
                # C/E/B 축별 처리
                if axis_name in ['compliance', 'behavior']:
                    # C축/B축: score 가산 방식
                    rule_score = rule_data.get('score', 0)
                    applied_rule['score'] = rule_score
                    applied_rule['type'] = 'score_addition'
                    
                elif axis_name == 'exposure':
                    # E축: weight 곱셈 방식
                    rule_weight = rule_data.get('weight', 1.0)
                    applied_rule['weight'] = rule_weight
                    applied_rule['type'] = 'weight_multiplication'
                
                applied_rules.append(applied_rule)
                
        except Exception as e:
            print(f"룰 평가 오류 ({rule_id}): {e}")
        
        return applied_rules, rule_score, rule_weight
    
    def _evaluate_single_rule(self, rule_id: str, rule_data: Dict[str, Any], features: Dict[str, float]) -> List[Dict[str, Any]]:
        """단일 룰을 평가합니다."""
        applied_rules = []
        condition = rule_data.get('condition', '')
        
        try:
            # 간단한 조건 평가 (실제로는 더 정교한 파서 필요)
            if self._evaluate_simple_condition(condition, features):
                applied_rules.append({
                    'rule_id': rule_id,
                    'name': rule_data.get('name', rule_id),
                    'description': rule_data.get('description', ''),
                    'score': rule_data.get('score', 0),
                    'weight': rule_data.get('weight', 1.0),
                    'risk_multiplier': rule_data.get('risk_multiplier', 1.0),
                    'condition': condition
                })
        except Exception as e:
            print(f"룰 평가 오류 ({rule_id}): {e}")
        
        return applied_rules
    
    def _evaluate_simple_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """
        🔍 룰 조건 평가기: YAML의 조건 문자열을 실제로 평가합니다.
        
        지원하는 조건들:
        - "usd_amount >= 10000": 1만달러 이상 거래
        - "tx_count(10m) >= 3": 10분간 3회 이상 거래
        - "tx_count(1m) >= 5": 1분간 5회 이상 거래
        
        동작 방식:
        1. 조건 문자열을 파싱
        2. 피처 데이터에서 해당 값을 찾음
        3. 수학적 비교 수행 (>=, >, < 등)
        4. True/False 결과 반환
        """
        # 현재는 기본적인 패턴 매칭 방식 사용
        # 실제 운영에서는 더 정교한 조건 파서(AST 등) 필요
        
        # === 고액 거래 조건: "usd_amount >= 10000" ===
        if 'usd_amount >= 10000' in condition:
            total_value = features.get('exposure_total_value', 0)  # Wei 단위 총 거래량
            # Wei → ETH → USD 변환 (1 ETH = 10^18 Wei, 1 ETH ≈ $2000)
            # 실제로는 실시간 가격 오라클 API 필요
            usd_amount = total_value * 0.000000000000000001 * 2000
            return usd_amount >= 10000  # 1만달러 이상이면 True
        
        # === 고빈도 거래 조건: "tx_count(10m) >= 3" ===
        if 'tx_count(10m) >= 3' in condition:
            tx_frequency = features.get('tx_frequency_per_hour', 0)  # 시간당 거래 빈도
            # 시간당 빈도를 10분당 빈도로 변환 (1시간 = 6 × 10분)
            tx_count_10m = tx_frequency / 6
            return tx_count_10m >= 3  # 10분간 3회 이상이면 True
        
        # === 초고빈도 거래 조건: "tx_count(1m) >= 5" ===
        if 'tx_count(1m) >= 5' in condition:
            tx_frequency = features.get('tx_frequency_per_hour', 0)  # 시간당 거래 빈도
            # 시간당 빈도를 1분당 빈도로 변환 (1시간 = 60분)
            tx_count_1m = tx_frequency / 60
            return tx_count_1m >= 5  # 1분간 5회 이상이면 True
        
        # 구현되지 않은 조건은 False 반환 (안전한 기본값)
        return False
    
    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """조건을 평가합니다."""
        # 안전한 조건 평가를 위한 간단한 파서
        # 실제로는 더 정교한 파서나 AST 사용 권장
        
        # 지원하는 연산자들
        operators = ['>', '<', '>=', '<=', '==', '!=']
        
        for op in operators:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = float(parts[1].strip())
                    
                    if left in features:
                        left_value = features[left]
                        
                        if op == '>':
                            return left_value > right
                        elif op == '<':
                            return left_value < right
                        elif op == '>=':
                            return left_value >= right
                        elif op == '<=':
                            return left_value <= right
                        elif op == '==':
                            return left_value == right
                        elif op == '!=':
                            return left_value != right
                break
        
        return False
    
    def _calculate_anomaly_scores(self, 
                                features: Dict[str, float],
                                transactions: Optional[List[Dict[str, Any]]] = None,
                                address: Optional[str] = None) -> Dict[str, float]:
        """이상행동 탐지 점수들을 계산합니다."""
        anomaly_scores = {}
        
        # 고급 피처 추출 (트랜잭션 데이터가 있는 경우)
        if transactions and address:
            try:
                advanced_features = extract_advanced_features(transactions, address)
                # 기본 피처와 고급 피처 결합
                combined_features = {**features, **advanced_features}
            except Exception as e:
                print(f"고급 피처 추출 실패: {e}")
                combined_features = features
        else:
            combined_features = features
        
        # 비지도 학습 탐지
        if self.unsupervised_detector and self.unsupervised_detector.is_fitted:
            try:
                unsupervised_scores = self.unsupervised_detector.predict_anomaly_scores(combined_features)
                anomaly_scores.update(unsupervised_scores)
            except Exception as e:
                print(f"비지도 학습 탐지 실패: {e}")
        
        # 지도 학습 탐지
        if self.supervised_detector and self.supervised_detector.is_fitted:
            try:
                supervised_prob = self.supervised_detector.predict_proba(combined_features)
                anomaly_scores['supervised'] = supervised_prob
            except Exception as e:
                print(f"지도 학습 탐지 실패: {e}")
        
        # 가중 평균으로 통합 점수 계산
        if anomaly_scores:
            combined_score = self._calculate_weighted_anomaly_score(anomaly_scores)
            anomaly_scores['combined_score'] = combined_score
        else:
            anomaly_scores['combined_score'] = 0.0
        
        return anomaly_scores
    
    def _calculate_weighted_anomaly_score(self, anomaly_scores: Dict[str, float]) -> float:
        """이상행동 탐지 점수들의 가중 평균을 계산합니다."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, score in anomaly_scores.items():
            if method in self.anomaly_weights:
                weight = self.anomaly_weights[method]
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _aggregate_scores(self, 
                         weighted_score: float, 
                         rule_results: List[Dict[str, Any]], 
                         anomaly_scores: Dict[str, float]) -> float:
        """
        🎯 4단계: 모든 점수들을 하나로 합쳐서 최종 리스크 점수를 계산합니다.
        
        집계 공식:
        최종점수 = (룰기반점수 × 70%) + (AI이상탐지점수 × 30%)
        
        예시 계산:
        1. 기본점수: 0.4 (1단계 가중합)
        2. 룰배수 적용: 0.4 × 1.5 × 1.2 = 0.72 (고액+고빈도 룰 적용)
        3. AI점수: 0.3 (이상 패턴 감지)
        4. 최종: (0.72 × 0.7) + (0.3 × 0.3) = 0.594점
        """
        # === 1. 룰 기반 점수 계산 ===
        rule_score = weighted_score  # 1단계에서 계산한 기본 가중합 점수
        
        # === 2. 적용된 룰들의 위험 배수를 누적 적용 ===
        # 예: 고액거래 룰(1.5배) + 고빈도 룰(1.2배) = 1.5 × 1.2 = 1.8배
        for rule in rule_results:
            multiplier = rule.get('risk_multiplier', 1.0)  # 룰의 위험 배수 (기본 1.0)
            rule_score *= multiplier  # 누적으로 배수 적용
        
        # === 3. AI 이상행동 탐지 점수 ===
        anomaly_score = anomaly_scores.get('combined_score', 0.0)
        
        # === 4. 최종 집계: 룰 70% + AI 30% 가중 평균 ===
        # 룰 기반이 더 중요하다고 판단하여 70% 가중치 부여
        final_score = (rule_score * 0.7) + (anomaly_score * 0.3)
        
        # === 5. 점수를 0-1 범위로 제한 ===
        # 계산 오류로 음수나 1 초과가 되는 것을 방지
        return min(1.0, max(0.0, final_score))
    
    def _determine_risk_level(self, score: float) -> str:
        """
        🎚️ 5단계: 최종 점수를 4단계 리스크 등급으로 분류합니다.
        
        등급 체계 (YAML 설정값 기준):
        🟢 VERY_LOW: 0-29점 (매우 안전)
        🟡 LOW: 30-49점 (낮은 위험)
        🟠 MEDIUM: 50-69점 (중간 위험)
        🔴 HIGH: 70-89점 (높은 위험)
        ⚫ CRITICAL: 90-100점 (매우 위험)
        
        예시:
        - 점수 0.594 → 59.4점 → "MEDIUM" 등급
        - 점수 0.856 → 85.6점 → "HIGH" 등급
        """
        # === 1. 점수를 0-100 범위로 변환 ===
        # 내부적으로는 0-1 범위를 사용하지만, 사람이 이해하기 쉽게 100점 만점으로 변환
        score_100 = score * 100
        
        # === 2. YAML 파일에서 각 등급의 임계값 로드 ===
        risk_thresholds = self.thresholds.get('risk_score', {})
        
        # 각 등급별 임계값 (YAML에서 설정, 없으면 기본값 사용)
        critical_threshold = risk_thresholds.get('critical', 90)  # 매우 위험: 90점 이상
        high_threshold = risk_thresholds.get('high', 70)          # 높은 위험: 70점 이상
        medium_threshold = risk_thresholds.get('medium', 50)      # 중간 위험: 50점 이상
        low_threshold = risk_thresholds.get('low', 30)           # 낮은 위험: 30점 이상
        
        # === 3. 점수에 따른 등급 분류 (높은 등급부터 순서대로 체크) ===
        if score_100 >= critical_threshold:
            return 'CRITICAL'    # ⚫ 90점 이상: 매우 위험
        elif score_100 >= high_threshold:
            return 'HIGH'        # 🔴 70점 이상: 높은 위험
        elif score_100 >= medium_threshold:
            return 'MEDIUM'      # 🟠 50점 이상: 중간 위험
        elif score_100 >= low_threshold:
            return 'LOW'         # 🟡 30점 이상: 낮은 위험
        else:
            return 'VERY_LOW'    # 🟢 30점 미만: 매우 안전
    
    def _calculate_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """각 피처의 기여도를 계산합니다."""
        contributions = {}
        total_weight = sum(self.weights.values())
        
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                normalized_value = self._normalize_feature_value(feature_name, features[feature_name])
                contribution = (normalized_value * weight) / total_weight if total_weight > 0 else 0
                contributions[feature_name] = contribution
        
        return contributions


    def _evaluate_rules_old(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        기존 방식의 룰 평가 (호환성을 위해 보존)
        """
        applied_rules = []
        axes = self.policy.get('axes', {})
        
        for axis_name, axis_rules in axes.items():
            applied_rules.extend(self._evaluate_axis_rules_old(axis_name, axis_rules, features))
        
        return applied_rules
    
    def _evaluate_axis_rules_old(self, axis_name: str, axis_rules: Dict[str, Any], features: Dict[str, float]) -> List[Dict[str, Any]]:
        """기존 방식의 축별 룰 평가 (호환성을 위해 보존)"""
        applied_rules = []
        
        for rule_id, rule_data in axis_rules.items():
            if isinstance(rule_data, dict) and 'condition' not in rule_data:
                for sub_rule_id, sub_rule_data in rule_data.items():
                    if isinstance(sub_rule_data, dict) and 'condition' in sub_rule_data:
                        applied_rules.extend(self._evaluate_single_rule_old(f"{rule_id}.{sub_rule_id}", sub_rule_data, features))
            else:
                if isinstance(rule_data, dict) and 'condition' in rule_data:
                    applied_rules.extend(self._evaluate_single_rule_old(rule_id, rule_data, features))
        
        return applied_rules
    
    def _evaluate_single_rule_old(self, rule_id: str, rule_data: Dict[str, Any], features: Dict[str, float]) -> List[Dict[str, Any]]:
        """기존 방식의 단일 룰 평가 (호환성을 위해 보존)"""
        applied_rules = []
        condition = rule_data.get('condition', '')
        
        try:
            if self._evaluate_simple_condition(condition, features):
                applied_rules.append({
                    'rule_id': rule_id,
                    'name': rule_data.get('name', rule_id),
                    'description': rule_data.get('description', ''),
                    'score': rule_data.get('score', 0),
                    'weight': rule_data.get('weight', 1.0),
                    'risk_multiplier': rule_data.get('risk_multiplier', 1.0),
                    'condition': condition
                })
        except Exception as e:
            print(f"룰 평가 오류 ({rule_id}): {e}")
        
        return applied_rules

    def _aggregate_scores_ceb(self, 
                             weighted_score: float, 
                             rule_results: Dict[str, Any], 
                             anomaly_scores: Dict[str, float]) -> float:
        """
        🎯 C/E/B 축 기반 점수 집계: 올바른 C/E/B 축 공식을 사용합니다.
        
        C/E/B 축 집계 공식:
        1. 기본 점수 = weighted_score (피처 가중합)
        2. C축 점수 = C_score (컴플라이언스 위반 점수 가산)  
        3. B축 점수 = B_score (행동 패턴 점수 가산)
        4. E축 가중치 = E_weight (노출 위험 가중치 적용)
        5. 룰 기반 점수 = (기본점수 + C축점수 + B축점수) × E축가중치
        6. 최종 점수 = (룰기반점수 × 70%) + (AI점수 × 30%)
        
        예시 계산:
        - 기본점수: 30, C축: +80, B축: +30, E축: ×0.8
        - 룰기반점수: (30 + 80 + 30) × 0.8 = 112
        - AI점수: 25
        - 최종점수: (112 × 0.7) + (25 × 0.3) = 78.4 + 7.5 = 85.9
        """
        # === 1. C/E/B 축별 점수/가중치 추출 ===
        base_score = weighted_score                    # 기본 피처 가중합 점수
        C_score = rule_results.get('C_score', 0)      # C축: 컴플라이언스 위반 점수
        B_score = rule_results.get('B_score', 0)      # B축: 행동 패턴 점수  
        E_weight = rule_results.get('E_weight', 1.0)  # E축: 노출 위험 가중치
        
        # === 2. C/E/B 축 기반 룰 점수 계산 ===
        # 공식: (기본점수 + C축점수 + B축점수) × E축가중치
        rule_score = (base_score + C_score + B_score) * E_weight
        
        # === 3. AI 이상행동 탐지 점수 ===
        raw_anomaly_score = anomaly_scores.get('combined_score', 0.0)
        
        # === 4. 스케일 정규화: 둘 다 0-100 범위로 맞춤 ===
        # 룰 점수는 이미 0-100+ 범위, anomaly는 0-1 범위이므로 100배 스케일링
        normalized_rule_score = min(100.0, max(0.0, rule_score))
        normalized_anomaly_score = min(100.0, max(0.0, raw_anomaly_score * 100))
        
        # === 5. 최종 집계: 동일한 스케일에서 하이브리드 결합 ===
        final_score_100 = (normalized_rule_score * 0.7) + (normalized_anomaly_score * 0.3)
        
        # === 6. 0-1 범위로 최종 정규화 ===
        normalized_score = final_score_100 / 100.0
        
        return normalized_score


def calculate_risk_score(features: Dict[str, float], 
                        policy_path: Optional[str] = None,
                        transactions: Optional[List[Dict[str, Any]]] = None,
                        address: Optional[str] = None,
                        use_anomaly_detection: bool = False) -> Dict[str, Any]:
    """
    편의 함수: 피처들로부터 리스크 스코어를 계산합니다.
    
    Args:
        features: 추출된 피처 딕셔너리
        policy_path: 정책 파일 경로
        transactions: 트랜잭션 리스트
        address: 분석 대상 주소
        use_anomaly_detection: 이상행동 탐지 사용 여부
    
    Returns:
        리스크 스코어 결과
    """
    # 이상행동 탐지기 설정
    unsupervised_detector = None
    supervised_detector = None
    
    if use_anomaly_detection:
        try:
            unsupervised_detector = create_default_unsupervised_detector()
            # 간단한 더미 데이터로 훈련 (실제로는 사전 훈련된 모델 로드)
            dummy_features = [features] * 10  # 최소한의 데이터
            unsupervised_detector.fit(dummy_features)
        except Exception as e:
            print(f"이상행동 탐지기 초기화 실패: {e}")
    
    scorer = RiskScorer(policy_path, unsupervised_detector, supervised_detector)
    return scorer.calculate_risk_score(features, transactions, address)
