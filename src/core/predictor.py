"""
Loads trained spam detection models and makes predictions.
"""

import os
import pickle
import threading
from typing import Dict
from ..utils.logger import setup_logger
from .preprocessor import TextPreprocessor
from .feature_engineer import SpamFeatureEngineer

logger = setup_logger(__name__)


class SpamPredictionEngine:
    """Handles loading models and predicting spam messages."""

    def __init__(self,
                 model_path: str = 'models/spam_classifier.pkl',
                 feature_engineer_path: str = 'models/feature_engineer.pkl'):
        """Set file paths and prepare placeholders."""
        self.model_path = model_path
        self.feature_engineer_path = feature_engineer_path

        self.model = None
        self.feature_engineer = None
        self.preprocessor = None

        self.is_ready = False
        self.last_error = None
        self._lock = threading.Lock()

        logger.info("SpamPredictionEngine created")

    def load_models(self) -> bool:
        """Load ML model, feature engineer, and preprocessor."""
        logger.info("Loading models...")

        try:
            with self._lock:
                if not os.path.exists(self.model_path):
                    self.last_error = f"Missing model: {self.model_path}"
                    logger.error(self.last_error)
                    return False

                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded")

                if not os.path.exists(self.feature_engineer_path):
                    self.last_error = f"Missing feature engineer: {self.feature_engineer_path}"
                    logger.error(self.last_error)
                    return False

                self.feature_engineer = SpamFeatureEngineer.load_feature_engineer(self.feature_engineer_path)
                logger.info("Feature engineer loaded")

                self.preprocessor = TextPreprocessor(
                    remove_urls=True,
                    remove_emails=True,
                    remove_phone_numbers=True,
                    remove_special_chars=True,
                    convert_lowercase=True,
                    remove_stopwords=True,
                    apply_stemming=True,
                    min_word_length=2
                )
                logger.info("Preprocessor ready")

                self.is_ready = True
                logger.info("Engine ready")
                return True

        except Exception as e:
            self.last_error = f"Load error: {str(e)}"
            logger.error(self.last_error)
            return False

    def predict_message(self, message: str) -> Dict:
        """Predict spam/ham for a given message."""
        if not self.is_ready:
            return {'success': False, 'error': 'Engine not ready', 'prediction': None, 'confidence': 0.0}

        if not message or not message.strip():
            return {'success': False, 'error': 'Empty message', 'prediction': None, 'confidence': 0.0}

        try:
            with self._lock:
                processed = self.preprocessor.transform([message])[0]
                if not processed.strip():
                    return {'success': False, 'error': 'No content after cleaning', 'prediction': None, 'confidence': 0.0}

                features = self.feature_engineer.transform([processed])
                pred = self.model.predict(features)[0]
                probs = self.model.predict_proba(features)[0]

                is_spam = (pred == 1)
                confidence = float(max(probs))

                result = {
                    'success': True,
                    'is_spam': is_spam,
                    'prediction': 'SPAM' if is_spam else 'HAM',
                    'confidence': confidence,
                    'spam_probability': float(probs[1]),
                    'ham_probability': float(probs[0]),
                    'original_length': len(message.split()),
                    'processed_length': len(processed.split()),
                    'error': None
                }

                logger.info(f"Prediction: {result['prediction']} ({confidence:.2f})")
                return result

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg, 'prediction': None, 'confidence': 0.0}

    def get_model_info(self) -> Dict:
        """Return info about model and preprocessing."""
        if not self.is_ready:
            return {'error': 'Models not loaded'}

        try:
            return {
                'model_type': type(self.model).__name__,
                'feature_count': len(self.feature_engineer.get_feature_names()),
                'preprocessing_steps': [
                    'remove URLs',
                    'remove emails',
                    'remove phone numbers',
                    'remove special chars',
                    'lowercase',
                    'remove stopwords',
                    'stemming',
                    'TF-IDF features',
                ]
            }
        except Exception as e:
            return {'error': f'Info error: {str(e)}'}

    def health_check(self) -> Dict:
        """Check if files and components are working."""
        status = {'healthy': True, 'last_error': self.last_error, 'components': {}}

        checks = [
            ('model_file', os.path.exists(self.model_path)),
            ('feature_engineer_file', os.path.exists(self.feature_engineer_path)),
            ('model_loaded', self.model is not None),
            ('feature_engineer_loaded', self.feature_engineer is not None),
            ('preprocessor_ready', self.preprocessor is not None),
            ('engine_ready', self.is_ready)
        ]

        for name, ok in checks:
            status['components'][name] = ok
            if not ok:
                status['healthy'] = False

        return status
