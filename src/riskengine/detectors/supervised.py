"""
지도 학습 기반 이상행동 탐지 모듈

XGBoost, LightGBM 등을 사용하여 라벨된 데이터로부터
이상행동 패턴을 학습합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path

# XGBoost와 LightGBM import (선택적)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# 대안으로 sklearn의 그래디언트 부스팅 사용
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class SupervisedAnomalyDetector:
    """지도 학습 기반 이상행동 탐지기"""
    
    def __init__(self, model_type: str = 'xgboost', config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_type: 모델 타입 ('xgboost', 'lightgbm', 'random_forest', 'logistic')
            config: 모델 설정 딕셔너리
        """
        self.model_type = model_type
        self.config = config or self._get_default_config(model_type)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.is_fitted = False
        
    def _get_default_config(self, model_type: str) -> Dict[str, Any]:
        """모델별 기본 설정을 반환합니다."""
        configs = {
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            },
            'lightgbm': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'metric': 'auc',
                'verbose': -1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'logistic': {
                'random_state': 42,
                'max_iter': 1000,
                'C': 1.0
            }
        }
        
        return configs.get(model_type, configs['random_forest'])
    
    def _create_model(self) -> Any:
        """모델을 생성합니다."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(**self.config)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(**self.config)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.config)
        elif self.model_type == 'logistic':
            return LogisticRegression(**self.config)
        else:
            # 폴백: GradientBoostingClassifier
            config = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            return GradientBoostingClassifier(**config)
    
    def fit(self, 
            features_list: List[Dict[str, float]], 
            labels: List[int],
            validation_split: float = 0.2) -> 'SupervisedAnomalyDetector':
        """
        모델을 훈련합니다.
        
        Args:
            features_list: 피처 딕셔너리 리스트
            labels: 라벨 리스트 (0: 정상, 1: 이상)
            validation_split: 검증 데이터 비율
            
        Returns:
            자기 자신
        """
        if not features_list or not labels:
            raise ValueError("빈 데이터입니다.")
        
        if len(features_list) != len(labels):
            raise ValueError("피처와 라벨의 개수가 일치하지 않습니다.")
        
        # 피처를 DataFrame으로 변환
        df = pd.DataFrame(features_list)
        
        # 결측값 처리
        df = df.fillna(0)
        
        # 무한값 처리
        df = df.replace([np.inf, -np.inf], 0)
        
        # 피처 컬럼 저장
        self.feature_columns = df.columns.tolist()
        
        # 스케일링
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(df)
        
        # 라벨 인코딩 (필요한 경우)
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 훈련/검증 분할
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                scaled_features, encoded_labels,
                test_size=validation_split,
                random_state=42,
                stratify=encoded_labels
            )
        else:
            X_train, y_train = scaled_features, encoded_labels
            X_val, y_val = None, None
        
        # 모델 생성 및 훈련
        self.model = self._create_model()
        
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # 조기 종료를 위한 검증 세트 사용
            eval_set = [(X_val, y_val)]
            
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model.fit(X_train, y_train, 
                             eval_set=eval_set, 
                             early_stopping_rounds=10,
                             verbose=False)
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model.fit(X_train, y_train,
                             eval_set=eval_set,
                             callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # 검증 점수 출력
        if X_val is not None:
            val_predictions = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_predictions)
            print(f"검증 AUC: {val_auc:.4f}")
        
        return self
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        이상 확률을 예측합니다.
        
        Args:
            features: 피처 딕셔너리
            
        Returns:
            이상 확률 (0~1)
        """
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다. fit()을 먼저 호출하세요.")
        
        # 피처 전처리
        feature_vector = self._preprocess_features(features)
        
        # 예측
        proba = self.model.predict_proba([feature_vector])[0]
        
        # 이상 클래스의 확률 반환
        return proba[1] if len(proba) > 1 else proba[0]
    
    def predict(self, features: Dict[str, float], threshold: float = 0.5) -> int:
        """
        이상 여부를 예측합니다.
        
        Args:
            features: 피처 딕셔너리
            threshold: 이상 판정 임계값
            
        Returns:
            예측 결과 (0: 정상, 1: 이상)
        """
        proba = self.predict_proba(features)
        return 1 if proba > threshold else 0
    
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
        scaled_features = self.scaler.transform(df)
        
        return scaled_features[0]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도를 반환합니다."""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 로지스틱 회귀의 경우
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_columns, importances))
    
    def cross_validate(self, 
                      features_list: List[Dict[str, float]], 
                      labels: List[int],
                      cv_folds: int = 5) -> Dict[str, float]:
        """
        교차 검증을 수행합니다.
        
        Args:
            features_list: 피처 딕셔너리 리스트
            labels: 라벨 리스트
            cv_folds: 교차 검증 폴드 수
            
        Returns:
            교차 검증 결과
        """
        # 데이터 전처리
        df = pd.DataFrame(features_list)
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        
        # 교차 검증
        model = self._create_model()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, scaled_features, labels, 
                               cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return {
            'mean_auc': np.mean(scores),
            'std_auc': np.std(scores),
            'scores': scores.tolist()
        }
    
    def save_model(self, model_path: str) -> None:
        """모델을 저장합니다."""
        if not self.is_fitted:
            raise ValueError("훈련된 모델이 없습니다.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str) -> 'SupervisedAnomalyDetector':
        """모델을 로드합니다."""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        return self


class EnsembleClassifier:
    """앙상블 분류기"""
    
    def __init__(self, models: List[SupervisedAnomalyDetector], weights: Optional[List[float]] = None):
        """
        Args:
            models: 모델 리스트
            weights: 각 모델의 가중치
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("모델 수와 가중치 수가 일치하지 않습니다.")
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        앙상블 이상 확률을 예측합니다.
        
        Args:
            features: 피처 딕셔너리
            
        Returns:
            앙상블 이상 확률
        """
        weighted_probas = []
        
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(features)
            weighted_probas.append(proba * weight)
        
        return np.mean(weighted_probas)
    
    def predict(self, features: Dict[str, float], threshold: float = 0.5) -> int:
        """앙상블 예측을 수행합니다."""
        proba = self.predict_proba(features)
        return 1 if proba > threshold else 0


def create_default_supervised_detector(model_type: str = 'xgboost') -> SupervisedAnomalyDetector:
    """기본 지도 학습 탐지기를 생성합니다."""
    return SupervisedAnomalyDetector(model_type=model_type)


def train_multiple_models(features_list: List[Dict[str, float]], 
                         labels: List[int],
                         model_types: List[str] = None) -> Dict[str, SupervisedAnomalyDetector]:
    """
    여러 모델을 동시에 훈련합니다.
    
    Args:
        features_list: 피처 딕셔너리 리스트
        labels: 라벨 리스트
        model_types: 훈련할 모델 타입들
        
    Returns:
        훈련된 모델들의 딕셔너리
    """
    if model_types is None:
        model_types = ['xgboost', 'lightgbm', 'random_forest', 'logistic']
    
    trained_models = {}
    
    for model_type in model_types:
        print(f"훈련 중: {model_type}")
        
        try:
            detector = SupervisedAnomalyDetector(model_type=model_type)
            detector.fit(features_list, labels)
            trained_models[model_type] = detector
            print(f"{model_type} 훈련 완료")
            
        except Exception as e:
            print(f"{model_type} 훈련 실패: {e}")
            continue
    
    return trained_models


def evaluate_model_performance(model: SupervisedAnomalyDetector,
                             test_features: List[Dict[str, float]],
                             test_labels: List[int]) -> Dict[str, Any]:
    """
    모델 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        test_features: 테스트 피처들
        test_labels: 테스트 라벨들
        
    Returns:
        성능 지표 딕셔너리
    """
    # 예측 수행
    predictions = []
    probabilities = []
    
    for features in test_features:
        pred = model.predict(features)
        prob = model.predict_proba(features)
        predictions.append(pred)
        probabilities.append(prob)
    
    # 성능 지표 계산
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1_score': f1_score(test_labels, predictions),
        'auc_roc': roc_auc_score(test_labels, probabilities)
    }
    
    # 분류 보고서
    metrics['classification_report'] = classification_report(test_labels, predictions)
    
    # 혼동 행렬
    metrics['confusion_matrix'] = confusion_matrix(test_labels, predictions).tolist()
    
    return metrics
