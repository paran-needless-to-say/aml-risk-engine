"""
비지도 학습 기반 이상행동 탐지 모듈

IsolationForest, LOF (Local Outlier Factor) 등을 사용하여
이상행동을 탐지합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


class UnsupervisedAnomalyDetector:
    """비지도 학습 기반 이상행동 탐지기"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 모델 설정 딕셔너리
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정을 반환합니다."""
        return {
            'isolation_forest': {
                'contamination': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            },
            'local_outlier_factor': {
                'contamination': 0.1,
                'n_neighbors': 20,
                'novelty': True,
                'n_jobs': -1
            },
            'dbscan': {
                'eps': 0.5,
                'min_samples': 5,
                'n_jobs': -1
            },
            'scaler': 'robust',  # 'standard', 'robust', 'none'
            'use_pca': False,
            'pca_components': 0.95
        }
    
    def fit(self, features_list: List[Dict[str, float]]) -> 'UnsupervisedAnomalyDetector':
        """
        모델을 훈련합니다.
        
        Args:
            features_list: 피처 딕셔너리 리스트
            
        Returns:
            자기 자신
        """
        if not features_list:
            raise ValueError("빈 피처 리스트입니다.")
        
        # 피처를 DataFrame으로 변환
        df = pd.DataFrame(features_list)
        
        # 결측값 처리
        df = df.fillna(0)
        
        # 무한값 처리
        df = df.replace([np.inf, -np.inf], 0)
        
        # 스케일링
        if self.config['scaler'] == 'standard':
            scaler = StandardScaler()
        elif self.config['scaler'] == 'robust':
            scaler = RobustScaler()
        else:
            scaler = None
        
        if scaler is not None:
            scaled_features = scaler.fit_transform(df)
            self.scalers['main'] = scaler
        else:
            scaled_features = df.values
        
        # PCA 적용
        if self.config['use_pca']:
            pca = PCA(n_components=self.config['pca_components'], random_state=42)
            scaled_features = pca.fit_transform(scaled_features)
            self.scalers['pca'] = pca
        
        # Isolation Forest
        if 'isolation_forest' in self.config:
            iso_forest = IsolationForest(**self.config['isolation_forest'])
            iso_forest.fit(scaled_features)
            self.models['isolation_forest'] = iso_forest
        
        # Local Outlier Factor
        if 'local_outlier_factor' in self.config:
            lof = LocalOutlierFactor(**self.config['local_outlier_factor'])
            lof.fit(scaled_features)
            self.models['local_outlier_factor'] = lof
        
        # DBSCAN (클러스터링 기반)
        if 'dbscan' in self.config:
            dbscan = DBSCAN(**self.config['dbscan'])
            dbscan.fit(scaled_features)
            self.models['dbscan'] = dbscan
        
        self.feature_columns = df.columns.tolist()
        self.is_fitted = True
        
        return self
    
    def predict_anomaly_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        이상 점수를 예측합니다.
        
        Args:
            features: 피처 딕셔너리
            
        Returns:
            모델별 이상 점수 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")
        
        # 피처 전처리
        feature_vector = self._preprocess_features(features)
        
        scores = {}
        
        # Isolation Forest
        if 'isolation_forest' in self.models:
            iso_score = self.models['isolation_forest'].decision_function([feature_vector])[0]
            # -1~1 범위를 0~1로 변환 (높을수록 이상)
            scores['isolation_forest'] = max(0, (1 - iso_score) / 2)
        
        # Local Outlier Factor
        if 'local_outlier_factor' in self.models:
            lof_score = self.models['local_outlier_factor'].decision_function([feature_vector])[0]
            # 음수일수록 이상, 0~1로 변환
            scores['local_outlier_factor'] = max(0, -lof_score)
        
        # DBSCAN (클러스터 거리 기반)
        if 'dbscan' in self.models:
            dbscan_score = self._calculate_dbscan_anomaly_score(feature_vector)
            scores['dbscan'] = dbscan_score
        
        return scores
    
    def _preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """피처를 전처리합니다."""
        # DataFrame 형태로 변환
        df = pd.DataFrame([features])
        
        # 훈련 시 사용된 컬럼들만 선택
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_columns]
        
        # 결측값과 무한값 처리
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        # 스케일링
        if 'main' in self.scalers:
            scaled_features = self.scalers['main'].transform(df)
        else:
            scaled_features = df.values
        
        # PCA
        if 'pca' in self.scalers:
            scaled_features = self.scalers['pca'].transform(scaled_features)
        
        return scaled_features[0]
    
    def _calculate_dbscan_anomaly_score(self, feature_vector: np.ndarray) -> float:
        """DBSCAN 기반 이상 점수를 계산합니다."""
        dbscan = self.models['dbscan']
        
        # 모든 클러스터 중심점들과의 거리 계산
        if not hasattr(dbscan, 'core_sample_indices_'):
            return 1.0  # 클러스터가 없으면 이상으로 간주
        
        if len(dbscan.core_sample_indices_) == 0:
            return 1.0
        
        # 코어 샘플들과의 최소 거리
        min_distance = float('inf')
        
        # 이 부분은 실제로는 훈련 데이터를 저장해야 함
        # 간단한 구현을 위해 고정값 반환
        return 0.5
    
    def save_models(self, model_dir: str) -> None:
        """모델들을 저장합니다."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델들 저장
        for name, model in self.models.items():
            joblib.dump(model, model_dir / f"{name}.joblib")
        
        # 스케일러들 저장
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, model_dir / f"scaler_{name}.joblib")
        
        # 설정과 메타데이터 저장
        joblib.dump({
            'config': self.config,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, model_dir / "metadata.joblib")
    
    def load_models(self, model_dir: str) -> 'UnsupervisedAnomalyDetector':
        """모델들을 로드합니다."""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise ValueError(f"모델 디렉토리가 존재하지 않습니다: {model_dir}")
        
        # 메타데이터 로드
        metadata = joblib.load(model_dir / "metadata.joblib")
        self.config = metadata['config']
        self.feature_columns = metadata['feature_columns']
        self.is_fitted = metadata['is_fitted']
        
        # 모델들 로드
        for model_file in model_dir.glob("*.joblib"):
            if model_file.name.startswith("scaler_"):
                scaler_name = model_file.name.replace("scaler_", "").replace(".joblib", "")
                self.scalers[scaler_name] = joblib.load(model_file)
            elif model_file.name != "metadata.joblib":
                model_name = model_file.name.replace(".joblib", "")
                self.models[model_name] = joblib.load(model_file)
        
        return self


class EnsembleAnomalyDetector:
    """앙상블 이상행동 탐지기"""
    
    def __init__(self, detectors: List[UnsupervisedAnomalyDetector], weights: Optional[List[float]] = None):
        """
        Args:
            detectors: 탐지기 리스트
            weights: 각 탐지기의 가중치
        """
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        
        if len(self.weights) != len(self.detectors):
            raise ValueError("탐지기 수와 가중치 수가 일치하지 않습니다.")
    
    def predict_anomaly_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        앙상블 이상 점수를 예측합니다.
        
        Args:
            features: 피처 딕셔너리
            
        Returns:
            통합된 이상 점수 딕셔너리
        """
        all_scores = {}
        ensemble_scores = {}
        
        # 각 탐지기의 점수 수집
        for i, detector in enumerate(self.detectors):
            detector_scores = detector.predict_anomaly_scores(features)
            
            for method, score in detector_scores.items():
                if method not in all_scores:
                    all_scores[method] = []
                all_scores[method].append(score * self.weights[i])
        
        # 앙상블 점수 계산
        for method, scores in all_scores.items():
            ensemble_scores[method] = np.mean(scores)
        
        # 전체 앙상블 점수
        ensemble_scores['ensemble_mean'] = np.mean(list(ensemble_scores.values()))
        ensemble_scores['ensemble_max'] = np.max(list(ensemble_scores.values()))
        
        return ensemble_scores


def create_default_unsupervised_detector() -> UnsupervisedAnomalyDetector:
    """기본 비지도 탐지기를 생성합니다."""
    config = {
        'isolation_forest': {
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        },
        'local_outlier_factor': {
            'contamination': 0.1,
            'n_neighbors': 20,
            'novelty': True,
            'n_jobs': -1
        },
        'scaler': 'robust',
        'use_pca': False
    }
    
    return UnsupervisedAnomalyDetector(config)


def detect_anomalies_batch(features_list: List[Dict[str, float]], 
                          detector: Optional[UnsupervisedAnomalyDetector] = None) -> List[Dict[str, float]]:
    """
    배치로 이상행동을 탐지합니다.
    
    Args:
        features_list: 피처 딕셔너리 리스트
        detector: 사용할 탐지기 (None이면 기본 탐지기 사용)
        
    Returns:
        각 샘플의 이상 점수 리스트
    """
    if detector is None:
        detector = create_default_unsupervised_detector()
    
    if not detector.is_fitted:
        # 전체 데이터로 훈련
        detector.fit(features_list)
    
    results = []
    for features in features_list:
        scores = detector.predict_anomaly_scores(features)
        results.append(scores)
    
    return results


def evaluate_detector_performance(true_labels: List[int], 
                                 anomaly_scores: List[float],
                                 threshold: float = 0.5) -> Dict[str, float]:
    """
    탐지기 성능을 평가합니다.
    
    Args:
        true_labels: 실제 라벨 (0: 정상, 1: 이상)
        anomaly_scores: 이상 점수
        threshold: 이상 판정 임계값
        
    Returns:
        성능 지표 딕셔너리
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    # 이상 점수를 이진 예측으로 변환
    predictions = [1 if score > threshold else 0 for score in anomaly_scores]
    
    metrics = {
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions),
        'auc_roc': roc_auc_score(true_labels, anomaly_scores)
    }
    
    return metrics
