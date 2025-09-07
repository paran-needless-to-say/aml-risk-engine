"""
리스크 스코어링 모듈 테스트
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from src.riskengine.scoring import RiskScorer, calculate_risk_score
from src.riskengine.features import extract_features
from src.riskengine.io import load_transaction_data


class TestRiskScorer:
    """RiskScorer 클래스 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.scorer = RiskScorer()
        
        # 테스트용 피처 데이터
        self.test_features = {
            'centrality_unique_addresses': 25.0,
            'centrality_tx_frequency': 50.0,
            'centrality_incoming_ratio': 0.6,
            'centrality_outgoing_ratio': 0.4,
            'exposure_total_value': 15000000.0,
            'exposure_avg_tx_value': 750000.0,
            'exposure_max_tx_value': 2000000.0,
            'behavior_interval_std': 30.0,
            'behavior_gas_price_std': 15.0,
            'behavior_repeat_tx_ratio': 0.2
        }
    
    def test_scorer_initialization_default(self):
        """기본 초기화 테스트"""
        scorer = RiskScorer()
        assert scorer.policy is not None
        assert 'feature_weights' in scorer.policy
        assert 'rules' in scorer.policy
        assert 'thresholds' in scorer.policy
    
    def test_scorer_initialization_with_policy(self):
        """정책 파일과 함께 초기화 테스트"""
        # 존재하지 않는 파일로 테스트 (기본 정책 사용)
        scorer = RiskScorer("/nonexistent/policy.yaml")
        assert scorer.policy is not None
    
    def test_calculate_risk_score_basic(self):
        """기본 리스크 스코어 계산 테스트"""
        result = self.scorer.calculate_risk_score(self.test_features)
        
        assert 'risk_score' in result
        assert 'risk_level' in result
        assert 'weighted_score' in result
        assert 'applied_rules' in result
        assert 'feature_contributions' in result
        
        assert 0 <= result['risk_score'] <= 1
        assert result['risk_level'] in ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']
    
    def test_weighted_score_calculation(self):
        """가중합 계산 테스트"""
        weighted_score = self.scorer._calculate_weighted_score(self.test_features)
        assert isinstance(weighted_score, float)
        assert 0 <= weighted_score <= 1
    
    def test_rule_evaluation(self):
        """룰 평가 테스트"""
        # 고빈도 거래 룰이 적용되는 피처
        high_freq_features = self.test_features.copy()
        high_freq_features['centrality_tx_frequency'] = 150.0
        
        applied_rules = self.scorer._evaluate_rules(high_freq_features)
        
        # 고빈도 거래 룰이 적용되었는지 확인
        rule_names = [rule['name'] for rule in applied_rules]
        assert 'high_frequency_trading' in rule_names
    
    def test_risk_level_determination(self):
        """리스크 레벨 결정 테스트"""
        assert self.scorer._determine_risk_level(0.1) == 'VERY_LOW'
        assert self.scorer._determine_risk_level(0.4) == 'LOW'
        assert self.scorer._determine_risk_level(0.7) == 'MEDIUM'
        assert self.scorer._determine_risk_level(0.9) == 'HIGH'
    
    def test_feature_contributions(self):
        """피처 기여도 계산 테스트"""
        contributions = self.scorer._calculate_feature_contributions(self.test_features)
        
        assert isinstance(contributions, dict)
        assert len(contributions) > 0
        
        # 기여도 합이 1에 가까운지 확인 (반올림 오차 고려)
        total_contribution = sum(contributions.values())
        assert 0.9 <= total_contribution <= 1.1
    
    def test_condition_evaluation(self):
        """조건 평가 테스트"""
        features = {'centrality_tx_frequency': 150.0, 'exposure_max_tx_value': 2000000.0}
        
        assert self.scorer._evaluate_condition('centrality_tx_frequency > 100', features) == True
        assert self.scorer._evaluate_condition('centrality_tx_frequency < 100', features) == False
        assert self.scorer._evaluate_condition('exposure_max_tx_value >= 2000000', features) == True
        assert self.scorer._evaluate_condition('exposure_max_tx_value <= 1000000', features) == False


class TestFeatureExtraction:
    """피처 추출 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        # 테스트용 트랜잭션 데이터
        self.test_transactions = [
            {
                'hash': '0x123...',
                'from': '0xaaa...',
                'to': '0xbbb...',
                'value': 1000000,
                'gas': 21000,
                'gasPrice': 20000000000,
                'timestamp': 1704067200
            },
            {
                'hash': '0x456...',
                'from': '0xbbb...',
                'to': '0xaaa...',
                'value': 500000,
                'gas': 21000,
                'gasPrice': 25000000000,
                'timestamp': 1704067260
            }
        ]
        
        self.test_address = '0xaaa...'
    
    def test_extract_features_basic(self):
        """기본 피처 추출 테스트"""
        features = extract_features(self.test_transactions, self.test_address)
        
        # 필수 피처들이 존재하는지 확인
        expected_features = [
            'centrality_unique_addresses',
            'centrality_tx_frequency', 
            'exposure_total_value',
            'behavior_interval_std'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    def test_extract_features_empty_transactions(self):
        """빈 트랜잭션 리스트 테스트"""
        features = extract_features([], self.test_address)
        
        # 0 값들이 올바르게 설정되는지 확인
        assert features['centrality_tx_frequency'] == 0
        assert features['exposure_total_value'] == 0
        assert features['centrality_unique_addresses'] == 0


class TestIOFunctions:
    """입출력 함수 테스트"""
    
    def test_load_transaction_data_file_not_exists(self):
        """존재하지 않는 파일 로딩 테스트"""
        with pytest.raises(FileNotFoundError):
            load_transaction_data("/nonexistent/file.jsonl", "file")
    
    def test_load_transaction_data_unsupported_type(self):
        """지원하지 않는 소스 타입 테스트"""
        with pytest.raises(ValueError):
            load_transaction_data("dummy", "unsupported_type")
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "data"}\n{"test2": "data2"}')
    @patch("jsonlines.open")
    def test_load_transaction_data_jsonl(self, mock_jsonlines, mock_file):
        """JSONL 파일 로딩 테스트"""
        # jsonlines.open의 모킹 설정
        mock_jsonlines.return_value.__enter__.return_value = [
            {"test": "data"},
            {"test2": "data2"}
        ]
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.suffix", ".jsonl"):
                result = load_transaction_data("test.jsonl", "file")
                assert len(result) == 2
                assert result[0]["test"] == "data"


class TestIntegration:
    """통합 테스트"""
    
    def test_end_to_end_scoring(self):
        """전체 스코어링 프로세스 테스트"""
        # 샘플 데이터 경로
        sample_file = Path(__file__).parent.parent / "data" / "samples" / "tx.jsonl"
        
        if sample_file.exists():
            # 실제 샘플 데이터로 테스트
            transactions = load_transaction_data(str(sample_file), "file")
            
            # 첫 번째 트랜잭션의 from 주소로 테스트
            if transactions:
                test_address = transactions[0].get('from')
                if test_address:
                    features = extract_features(transactions, test_address)
                    result = calculate_risk_score(features)
                    
                    assert 'risk_score' in result
                    assert 'risk_level' in result
                    assert 0 <= result['risk_score'] <= 1
    
    def test_api_data_models(self):
        """API 데이터 모델 테스트"""
        from src.riskengine.api import TransactionData, ScoreRequest
        
        # TransactionData 모델 테스트
        tx_data = TransactionData(
            from_address="0xaaa...",
            to_address="0xbbb...",
            value=1000000,
            timestamp=1704067200,
            gas=21000,
            gasPrice=20000000000,
            hash="0x123..."
        )
        
        assert tx_data.from_address == "0xaaa..."
        assert tx_data.value == 1000000
        
        # ScoreRequest 모델 테스트
        score_request = ScoreRequest(
            address="0xaaa...",
            transactions=[tx_data],
            window_hours=24
        )
        
        assert score_request.address == "0xaaa..."
        assert len(score_request.transactions) == 1


if __name__ == "__main__":
    pytest.main([__file__])
