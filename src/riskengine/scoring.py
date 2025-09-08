"""
리스크 스코어링 모듈

룰 기반 점수와 이상행동 탐지 결과를 집계하여 최종 risk_score를 계산합니다.
"""

from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import numpy as np

from .detectors.unsupervised import UnsupervisedAnomalyDetector, create_default_unsupervised_detector
from .detectors.supervised import SupervisedAnomalyDetector, create_default_supervised_detector
from .detectors.features import extract_advanced_features


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
            policy_path = Path(__file__).parent / "rules" / "default_light.yaml"
        
        self.policy = self._load_policy(policy_path)
        self.weights = self.policy.get('feature_weights', {})
        self.rules = self.policy.get('rules', [])
        self.thresholds = self.policy.get('thresholds', {})
        self.anomaly_weights = self.policy.get('anomaly_weights', {})
        
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
                    'warn': 60,
                    'alert': 80
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
    
    def calculate_risk_score(self, 
                           features: Dict[str, float],
                           transactions: Optional[List[Dict[str, Any]]] = None,
                           address: Optional[str] = None) -> Dict[str, Any]:
        """
        피처들로부터 리스크 스코어를 계산합니다.
        
        Args:
            features: 추출된 피처 딕셔너리
            transactions: 트랜잭션 리스트 (고급 피처 추출용)
            address: 분석 대상 주소 (고급 피처 추출용)
        
        Returns:
            리스크 스코어 결과
        """
        # 1. 가중합 계산 (기본 피처 기반)
        weighted_score = self._calculate_weighted_score(features)
        
        # 2. 룰 평가
        rule_results = self._evaluate_rules(features)
        
        # 3. 이상행동 탐지 점수 계산
        anomaly_scores = self._calculate_anomaly_scores(features, transactions, address)
        
        # 4. 최종 스코어 집계
        final_score = self._aggregate_scores(weighted_score, rule_results, anomaly_scores)
        
        # 5. 리스크 레벨 결정
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
        """가중합을 계산합니다."""
        score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                # 피처 값을 0-1 범위로 정규화
                normalized_value = self._normalize_feature_value(feature_name, features[feature_name])
                score += normalized_value * weight
                total_weight += weight
        
        # 가중 평균
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
    
    def _evaluate_rules(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """새로운 C/E/B 축 룰북 구조를 평가합니다."""
        applied_rules = []
        
        # 새로운 축 구조에서 룰 추출
        axes = self.policy.get('axes', {})
        
        # 각 축(compliance, exposure, behavior)을 순회
        for axis_name, axis_rules in axes.items():
            applied_rules.extend(self._evaluate_axis_rules(axis_name, axis_rules, features))
        
        return applied_rules
    
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
        """간단한 조건을 평가합니다 (현재는 기본 구현만)."""
        # 현재는 기본적인 피처 기반 조건만 지원
        # 실제로는 더 복잡한 조건 파서가 필요
        
        # 고액 거래 조건 예시
        if 'usd_amount >= 10000' in condition:
            total_value = features.get('exposure_total_value', 0)
            # Wei를 USD로 변환 (간단한 예시, 실제로는 가격 오라클 필요)
            usd_amount = total_value * 0.000000000000000001 * 2000  # 대략적인 ETH 가격
            return usd_amount >= 10000
        
        # 버스트 거래 조건 예시
        if 'tx_count(10m) >= 3' in condition:
            tx_frequency = features.get('tx_frequency_per_hour', 0)
            # 10분당 거래 수 추정
            tx_count_10m = tx_frequency / 6
            return tx_count_10m >= 3
        
        if 'tx_count(1m) >= 5' in condition:
            tx_frequency = features.get('tx_frequency_per_hour', 0)
            # 1분당 거래 수 추정
            tx_count_1m = tx_frequency / 60
            return tx_count_1m >= 5
        
        # 기본적으로 false 반환 (구현되지 않은 조건)
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
        """모든 점수들을 집계하여 최종 스코어를 계산합니다."""
        # 1. 룰 기반 점수 (가중합)
        rule_score = weighted_score
        
        # 2. 룰 배수 적용
        for rule in rule_results:
            multiplier = rule.get('risk_multiplier', 1.0)
            rule_score *= multiplier
        
        # 3. 이상행동 탐지 점수
        anomaly_score = anomaly_scores.get('combined_score', 0.0)
        
        # 4. 최종 집계 (룰 기반 70%, 이상행동 탐지 30%)
        final_score = (rule_score * 0.7) + (anomaly_score * 0.3)
        
        # 0-1 범위로 클리핑
        return min(1.0, max(0.0, final_score))
    
    def _determine_risk_level(self, score: float) -> str:
        """리스크 레벨을 결정합니다 (4단계)."""
        # score를 0-100 범위로 변환 (기존 0-1 범위에서)
        score_100 = score * 100
        
        # YAML에서 risk_score 임계값 가져오기
        risk_thresholds = self.thresholds.get('risk_score', {})
        
        critical_threshold = risk_thresholds.get('critical', 90)
        high_threshold = risk_thresholds.get('high', 70)
        medium_threshold = risk_thresholds.get('medium', 50)
        low_threshold = risk_thresholds.get('low', 30)
        
        if score_100 >= critical_threshold:
            return 'CRITICAL'
        elif score_100 >= high_threshold:
            return 'HIGH'
        elif score_100 >= medium_threshold:
            return 'MEDIUM'
        elif score_100 >= low_threshold:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
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
