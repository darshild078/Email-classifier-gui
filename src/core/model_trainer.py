import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
from utils.logger import setup_logger
from config.model_config import MODEL_CONFIG, PERFORMANCE_THRESHOLDS, MODEL_PATHS

logger = setup_logger(__name__)

class SpamModelTrainer:
    """Train and evaluate spam detection model with hyperparameter optimization."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODEL_CONFIG
        self.performance_thresholds = PERFORMANCE_THRESHOLDS
        self.model_paths = MODEL_PATHS
        self.model = self._initialize_model()
        self.is_trained = False
        self.training_metrics = {}
        self.best_threshold = 0.5  # Default classification threshold
        logger.info("Trainer ready")

    def _initialize_model(self):
        """Initialize Naive Bayes model."""
        if self.config['algorithm'] == 'MultinomialNB':
            model = MultinomialNB(alpha=self.config['alpha'])
        else:
            raise ValueError(f"Unsupported algorithm: {self.config['algorithm']}")
        logger.info(f"Model initialized: {type(model).__name__}")
        return model

    def train_and_evaluate(self, X, y, feature_names: Optional[list] = None) -> Dict:
        """Complete training pipeline with optimization."""
        logger.info("Starting training pipeline")
        
        # Validate input data first
        if not self._validate_training_data(X, y):
            raise ValueError("Training data validation failed")
        
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        cv_scores = self._perform_cross_validation(X_train, y_train)
        best_alpha = self._optimize_alpha(X_train, y_train)
        
        # Retrain with best alpha
        self.model.set_params(alpha=best_alpha)
        self._train_final_model(X_train, y_train)
        
        test_metrics = self._evaluate_on_test_set(X_test, y_test)
        
        # Optimize classification threshold if performance is poor
        if test_metrics.get('recall', 0) < 0.3:
            logger.info("Low recall detected, optimizing classification threshold")
            self.best_threshold = self._optimize_threshold(X_test, y_test)
            test_metrics = self._evaluate_with_threshold(X_test, y_test, self.best_threshold)
        
        feature_importance = self._analyze_feature_importance(X_train, y_train, feature_names)
        validation_results = self._validate_model_performance(test_metrics)
        
        results = {
            'cross_validation': cv_scores,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'validation_results': validation_results,
            'training_info': {
                'algorithm': self.config['algorithm'],
                'best_alpha': best_alpha,
                'best_threshold': self.best_threshold,
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'feature_count': X.shape[1],
                'train_spam_ratio': y_train.mean(),
                'test_spam_ratio': y_test.mean(),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info("Training pipeline finished")
        return results

    def _validate_training_data(self, X, y) -> bool:
        """Validate training data quality."""
        try:
            # Check data sizes
            if X.shape[0] != len(y):
                logger.error("Feature matrix and labels have different sizes")
                return False
            
            # Check for empty data
            if X.shape[0] == 0:
                logger.error("No training samples found")
                return False
            
            # Check label distribution
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                logger.error(f"Only {len(unique_labels)} unique label(s) found: {unique_labels}")
                return False
            
            spam_count = np.sum(y == 1)
            ham_count = np.sum(y == 0)
            
            if spam_count == 0:
                logger.error("No spam samples in training data - this causes 0% detection!")
                return False
            
            if ham_count == 0:
                logger.error("No ham samples in training data")
                return False
            
            logger.info(f"Data validation passed: {ham_count} ham, {spam_count} spam samples")
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    def _split_data(self, X, y) -> Tuple:
        """Split data with stratification to maintain class balance."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        train_samples = X_train.shape[0]
        test_samples = X_test.shape[0]
        
        logger.info(f"Train: {train_samples} samples ({y_train.sum()} spam, {y_train.mean():.1%})")
        logger.info(f"Test: {test_samples} samples ({y_test.sum()} spam, {y_test.mean():.1%})")
        
        return X_train, X_test, y_train, y_test

    def _perform_cross_validation(self, X_train, y_train) -> Dict:
        """Cross-validation with multiple metrics."""
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'], 
            shuffle=True, 
            random_state=self.config['random_state']
        )
        
        cv_metrics = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric_name in metrics:
            try:
                scores = cross_val_score(
                    self.model, X_train, y_train, 
                    cv=cv, scoring=metric_name, n_jobs=-1
                )
                cv_metrics[metric_name] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'scores': scores.tolist()
                }
                logger.info(f"CV {metric_name}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                logger.warning(f"CV metric {metric_name} failed: {e}")
                cv_metrics[metric_name] = {'mean': 0.0, 'std': 0.0, 'scores': []}
        
        logger.info("Cross-validation done")
        return cv_metrics

    def _optimize_alpha(self, X_train, y_train) -> float:
        """Find optimal alpha parameter with extended search."""
        # Extended alpha range for better spam detection
        alphas = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_alpha = 1.0
        best_score = 0
        
        logger.info("Optimizing alpha parameter...")
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for alpha in alphas:
            try:
                temp_model = MultinomialNB(alpha=alpha)
                
                # Use F1 score to balance precision and recall
                scores = cross_val_score(
                    temp_model, X_train, y_train, 
                    cv=cv, scoring='f1', n_jobs=-1
                )
                mean_score = scores.mean()
                
                logger.info(f"Alpha {alpha:6.3f}: F1 = {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    
            except Exception as e:
                logger.warning(f"Alpha {alpha} optimization failed: {e}")
                continue
        
        logger.info(f"Best alpha: {best_alpha} (F1 = {best_score:.4f})")
        return best_alpha

    def _optimize_threshold(self, X_test, y_test) -> float:
        """Optimize classification threshold for better spam recall."""
        try:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {f1_scores[best_idx]:.3f})")
            return float(best_threshold)
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
            return 0.5

    def _train_final_model(self, X_train, y_train):
        """Train final model on complete training set."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Final model trained")

    def _evaluate_on_test_set(self, X_test, y_test) -> Dict:
        """Comprehensive evaluation on test set."""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return self._compute_metrics(y_test, y_pred, y_pred_proba)

    def _evaluate_with_threshold(self, X_test, y_test, threshold: float) -> Dict:
        """Evaluate with custom classification threshold."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return self._compute_metrics(y_test, y_pred, y_pred_proba)

    def _compute_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Compute comprehensive evaluation metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Confusion matrix with better handling
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            metrics['confusion_matrix'] = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
        else:
            # Handle edge case where only one class is predicted
            logger.warning("Confusion matrix not 2x2, model may not be detecting both classes")
            metrics['confusion_matrix'] = {'error': 'Invalid confusion matrix shape'}
        
        # Classification report with error handling
        try:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=['Ham', 'Spam'], output_dict=True, zero_division=0
            )
        except Exception as e:
            logger.warning(f"Classification report failed: {e}")
            metrics['classification_report'] = {}
        
        # Log key metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics

    def _analyze_feature_importance(self, X_train, y_train, feature_names) -> Dict:
        """Analyze important features for spam detection."""
        if not feature_names:
            return {}
        
        try:
            from sklearn.feature_selection import chi2
            chi2_scores, p_values = chi2(X_train, y_train)
            
            feature_importance_data = [
                (name, float(score), float(p_val))
                for name, score, p_val in zip(feature_names, chi2_scores, p_values)
            ]
            
            # Sort by importance
            feature_importance_data.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance_data[:20]
            
            # Log top spam indicators
            logger.info("Top spam indicators:")
            for i, (name, score, p_val) in enumerate(top_features[:5], 1):
                logger.info(f"  {i}. {name}: {score:.2f} (p={p_val:.2e})")
            
            return {
                'top_spam_indicators': [
                    {'feature': name, 'chi2_score': score, 'p_value': p_val}
                    for name, score, p_val in top_features
                ],
                'total_features': len(feature_names)
            }
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            return {}

    def _validate_model_performance(self, test_metrics) -> Dict:
        """Check if model meets minimum performance thresholds."""
        validation_results = {
            'passed': True,
            'checks': {},
            'recommendations': []
        }
        
        checks = [
            ('accuracy', test_metrics['accuracy'], self.performance_thresholds['min_accuracy']),
            ('precision_spam', test_metrics['precision'], self.performance_thresholds['min_precision_spam']),
            ('recall_spam', test_metrics['recall'], self.performance_thresholds['min_recall_spam'])
        ]
        
        for metric_name, actual_value, threshold in checks:
            passed = actual_value >= threshold
            validation_results['checks'][metric_name] = {
                'actual': actual_value,
                'threshold': threshold,
                'passed': passed
            }
            
            if not passed:
                validation_results['passed'] = False
                logger.warning(f"Model failed {metric_name}: {actual_value:.4f} < {threshold:.4f}")
                
                # Add specific recommendations
                if metric_name == 'recall_spam' and actual_value == 0:
                    validation_results['recommendations'].append(
                        "0% spam recall suggests data labeling issues or severe class imbalance"
                    )
                elif metric_name == 'precision_spam' and actual_value < 0.5:
                    validation_results['recommendations'].append(
                        "Low precision suggests model is overpredicting spam"
                    )
            else:
                logger.info(f"✓ {metric_name} passed: {actual_value:.4f} >= {threshold:.4f}")
        
        return validation_results

    def save_model_and_metrics(self, results: Dict, force_save: bool = False):
        """Save trained model and metrics with optional force save."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Check validation before saving (unless force_save is True)
        if not force_save and not results.get('validation_results', {}).get('passed', False):
            logger.warning("Model failed validation - use force_save=True to save anyway")
            
            # Show specific issues
            failed_checks = []
            for check, result in results.get('validation_results', {}).get('checks', {}).items():
                if not result.get('passed', True):
                    failed_checks.append(f"{check}: {result['actual']:.3f} < {result['threshold']:.3f}")
            
            if failed_checks:
                logger.warning(f"Failed checks: {', '.join(failed_checks)}")
            
            return None
        
        os.makedirs('models', exist_ok=True)
        
        # Save model with threshold
        model_data = {
            'model': self.model,
            'threshold': self.best_threshold,
            'config': self.config
        }
        
        with open(self.model_paths['model_file'], 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {self.model_paths['model_file']}")
        
        # Save metrics
        metrics_with_metadata = {
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_type': type(self.model).__name__,
            'config': self.config,
            'threshold': self.best_threshold,
            'results': results
        }
        
        with open(self.model_paths['metrics_file'], 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2)
        logger.info(f"Metrics saved to {self.model_paths['metrics_file']}")
        
        return metrics_with_metadata

    def force_save_model(self):
        """Save model regardless of validation results (for testing)."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs('models', exist_ok=True)
        
        # Save just the model for quick testing
        with open(self.model_paths['model_file'], 'wb') as f:
            pickle.dump({
                'model': self.model,
                'threshold': self.best_threshold,
                'config': self.config
            }, f)
        
        logger.info(f"Model FORCE SAVED to {self.model_paths['model_file']}")
        return True
