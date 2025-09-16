import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from ..utils.logger import setup_logger
from config.model_config import MODEL_CONFIG, PERFORMANCE_THRESHOLDS, MODEL_PATHS

logger = setup_logger(__name__)

class SpamModelTrainer:
    """Train and evaluate a spam detection model with full metrics."""

    def __init__(self, config: Optional[Dict] = None):
        # Load config or use defaults
        self.config = config or MODEL_CONFIG
        self.performance_thresholds = PERFORMANCE_THRESHOLDS
        self.model_paths = MODEL_PATHS

        # Initialize model
        self.model = self._initialize_model()
        self.is_trained = False
        self.training_metrics = {}
        self.cross_validation_scores = {}

        logger.info("Trainer ready")

    def _initialize_model(self):
        """Initialize the ML model (MultinomialNB for text)."""
        if self.config['algorithm'] == 'MultinomialNB':
            model = MultinomialNB(alpha=self.config['alpha'])
        else:
            raise ValueError(f"Unsupported algorithm: {self.config['algorithm']}")
        logger.info(f"Model initialized: {type(model).__name__}")
        return model

    def train_and_evaluate(self, X, y, feature_names: Optional[list] = None) -> Dict:
        """Full training and evaluation workflow."""
        logger.info("Starting training pipeline")
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Cross-validation
        cv_scores = self._perform_cross_validation(X_train, y_train)
        
        # Train final model
        self._train_final_model(X_train, y_train)
        
        # Evaluate test set
        test_metrics = self._evaluate_on_test_set(X_test, y_test)
        
        # Feature importance
        feature_importance = self._analyze_feature_importance(X_train, y_train, feature_names)
        
        # Validate model against thresholds
        validation_results = self._validate_model_performance(test_metrics)
        
        # Compile results
        results = {
            'cross_validation': cv_scores,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'validation_results': validation_results,
            'training_info': {
                'algorithm': self.config['algorithm'],
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': X.shape[1],
                'train_spam_ratio': y_train.mean(),
                'test_spam_ratio': y_test.mean(),
                'timestamp': datetime.now().isoformat()
            }
        }
        logger.info("Training pipeline finished")
        return results

    def _split_data(self, X, y) -> Tuple:
        """Split data into training and testing sets with stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _perform_cross_validation(self, X_train, y_train) -> Dict:
        """Perform stratified cross-validation for multiple metrics."""
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.config['random_state'])
        metrics = {}
        for metric_name, scorer in [('accuracy','accuracy'),('precision','precision'),('recall','recall'),('f1','f1'),('roc_auc','roc_auc')]:
            scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
            metrics[metric_name] = {'mean': float(scores.mean()), 'std': float(scores.std())}
        logger.info("Cross-validation done")
        return metrics

    def _train_final_model(self, X_train, y_train):
        """Fit the model on the full training set."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Final model trained")

    def _evaluate_on_test_set(self, X_test, y_test) -> Dict:
        """Evaluate model on test data with multiple metrics."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': {'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])},
            'classification_report': classification_report(y_test, y_pred, target_names=['Ham','Spam'], output_dict=True)
        }
        logger.info("Test evaluation complete")
        return metrics

    def _analyze_feature_importance(self, X_train, y_train, feature_names) -> Dict:
        """Analyze features using chi-square test."""
        if not feature_names:
            return {}
        try:
            from sklearn.feature_selection import chi2
            chi2_scores, p_values = chi2(X_train, y_train)
            importance = sorted(
                [(name,float(score),float(p)) for name,score,p in zip(feature_names,chi2_scores,p_values)],
                key=lambda x:x[1], reverse=True
            )
            top_features = importance[:20]
            return {'top_features': top_features, 'total_features': len(feature_names)}
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")
            return {}

    def _validate_model_performance(self, test_metrics) -> Dict:
        """Check if model meets production thresholds."""
        results = {'passed': True, 'checks': {}}
        for metric, threshold in [('accuracy', self.performance_thresholds['min_accuracy']),
                                  ('precision_spam', self.performance_thresholds['min_precision_spam']),
                                  ('recall_spam', self.performance_thresholds['min_recall_spam'])]:
            value = test_metrics.get(metric,0)
            passed = value >= threshold
            results['checks'][metric] = {'actual': value, 'threshold': threshold, 'passed': passed}
            if not passed:
                results['passed'] = False
        return results

    def save_model_and_metrics(self, results: Dict):
        """Save trained model and metrics to files."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        os.makedirs('models', exist_ok=True)
        with open(self.model_paths['model_file'], 'wb') as f:
            pickle.dump(self.model, f)
        metrics_with_info = {'timestamp': datetime.now().isoformat(), 'config': self.config, 'results': results}
        with open(self.model_paths['metrics_file'], 'w') as f:
            json.dump(metrics_with_info, f, indent=2)
        logger.info("Model and metrics saved")
        return metrics_with_info
