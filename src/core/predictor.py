import pickle
import os
from typing import Dict, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SpamPredictionEngine:
    """Spam detection prediction engine with model loading and inference."""

    def __init__(self, model_path: str = 'models/spam_classifier.pkl', 
                 feature_engineer_path: str = 'models/feature_engineer.pkl'):
        self.model_path = model_path
        self.feature_engineer_path = feature_engineer_path
        self.model = None
        self.feature_engineer = None
        self.threshold = 0.5
        self.is_loaded = False

    @property
    def is_ready(self):
        """Alias for is_loaded to maintain GUI compatibility."""
        return self.is_loaded

    def load_models(self) -> bool:
        """Load trained model and feature engineer."""
        try:
            logger.info("Loading spam detection models...")
            
            # Check if files exist
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            if not os.path.exists(self.feature_engineer_path):
                logger.error(f"Feature engineer file not found: {self.feature_engineer_path}")
                return False
            
            # Load model (handle both dict and direct model formats)
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.threshold = model_data.get('threshold', 0.5)
                logger.info(f"Loaded model from dict with threshold: {self.threshold}")
            else:
                self.model = model_data
                self.threshold = 0.5
                logger.info("Loaded direct model object")
            
            if self.model is None:
                logger.error("Failed to extract model from loaded data")
                return False
            
            # Load feature engineer (handle both dict and direct formats)
            with open(self.feature_engineer_path, 'rb') as f:
                fe_data = pickle.load(f)
            
            if isinstance(fe_data, dict):
                self.feature_engineer = fe_data.get('feature_engineer', fe_data)
                logger.info("Loaded feature engineer from dict")
            else:
                self.feature_engineer = fe_data
                logger.info("Loaded direct feature engineer object")
            
            if self.feature_engineer is None:
                logger.error("Failed to extract feature engineer from loaded data")
                return False
            
            self.is_loaded = True
            logger.info("✅ Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Load error: {e}")
            self.is_loaded = False
            return False

    def predict(self, message: str) -> Dict:
        """Predict if message is spam or ham."""
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Models not loaded. Call load_models() first.',
                'prediction': None,
                'confidence': 0.0
            }
        
        if not message or not message.strip():
            return {
                'success': False,
                'error': 'Empty message provided',
                'prediction': None,
                'confidence': 0.0
            }
        
        try:
            logger.info(f"Predicting message: {message[:50]}...")
            
            # Transform message to features
            features = self.feature_engineer.transform([message.strip()])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            # Apply custom threshold if available
            spam_prob = probabilities[1] if len(probabilities) > 1 else 0
            ham_prob = probabilities[0] if len(probabilities) > 1 else 1
            
            # Determine prediction based on threshold
            prediction = 1 if spam_prob >= self.threshold else 0
            result_label = "SPAM" if prediction == 1 else "HAM"
            
            # Use appropriate probability as confidence
            confidence = spam_prob if prediction == 1 else ham_prob
            
            logger.info(f"Prediction: {result_label} (confidence: {confidence:.3f})")
            
            return {
                'success': True,
                'prediction': result_label,
                'confidence': float(confidence),
                'spam_probability': float(spam_prob),
                'ham_probability': float(ham_prob),
                'threshold_used': self.threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

    def predict_message(self, message: str) -> Dict:
        """Alias for predict method."""
        return self.predict(message)

    def batch_predict(self, messages: list) -> list:
        """Predict multiple messages at once."""
        results = []
        for message in messages:
            result = self.predict(message)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        if not self.is_loaded:
            return {'loaded': False, 'info': 'Models not loaded'}
        
        try:
            model_type = type(self.model).__name__
            fe_type = type(self.feature_engineer).__name__
            
            return {
                'loaded': True,
                'model_type': model_type,
                'feature_engineer_type': fe_type,
                'threshold': self.threshold,
                'model_path': self.model_path,
                'feature_engineer_path': self.feature_engineer_path
            }
        except Exception as e:
            return {'loaded': True, 'info': f'Error getting info: {e}'}

    def reload_models(self) -> bool:
        """Reload models from disk."""
        self.is_loaded = False
        self.model = None
        self.feature_engineer = None
        return self.load_models()


# Convenience function for simple usage
def predict_spam(message: str, model_path: str = 'models/spam_classifier.pkl', 
                 feature_engineer_path: str = 'models/feature_engineer.pkl') -> Dict:
    """Simple function to predict spam for a single message."""
    engine = SpamPredictionEngine(model_path, feature_engineer_path)
    
    if not engine.load_models():
        return {
            'success': False,
            'error': 'Failed to load models',
            'prediction': None,
            'confidence': 0.0
        }
    
    return engine.predict(message)


# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    print("🧪 Testing SpamPredictionEngine")
    print("=" * 50)
    
    predictor = SpamPredictionEngine()
    
    if predictor.load_models():
        print(f"✅ Models loaded. Ready status: {predictor.is_ready}")
        
        test_messages = [
            "Hi how are you today?",
            "FREE MONEY! Click here to win $1000 NOW!",
            "Meeting at 3pm in conference room B",
            "URGENT! Your account suspended! Verify immediately!",
            "Thanks for dinner last night"
        ]
        
        print("\nTesting predictions:")
        print("-" * 50)
        
        for msg in test_messages:
            result = predictor.predict(msg)
            if result['success']:
                pred = result['prediction']
                conf = result['confidence']
                print(f"{pred:<4} ({conf:.3f}) | {msg}")
            else:
                print(f"ERROR: {result['error']} | {msg}")
                
        print("\n✅ Predictor testing complete!")
        
    else:
        print("❌ Failed to load models. Please run training first:")
        print("   python train_model.py")
