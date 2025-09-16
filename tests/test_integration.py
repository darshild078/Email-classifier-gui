"""
Integration tests for spam detector.
Checks end-to-end pipeline and component interaction.
"""
import unittest
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

class TestSpamDetectorIntegration(unittest.TestCase):
    """Integration tests for full spam detection flow."""
    
    def setUp(self):
        """Prepare sample data."""
        self.test_messages = {
            'ham': [
                "Hi there! How are you doing today?",
                "Meeting at 3pm in conference room B",
                "Thanks for dinner last night, it was great!",
                "Can you pick up milk on your way home?"
            ],
            'spam': [
                "CONGRATULATIONS! You've won $1000! Click here now!",
                "URGENT! Your account will be suspended. Act immediately!",
                "FREE entry to win iPhone! Text WIN to 12345",
                "You have been selected for a special offer! Call now!"
            ]
        }
    
    def test_model_loading(self):
        """Models should load correctly."""
        from core.predictor import SpamPredictionEngine
        engine = SpamPredictionEngine()
        success = engine.load_models()
        self.assertTrue(success, "Model loading should succeed")
        self.assertTrue(engine.is_ready, "Engine should be ready after load")
    
    def test_spam_prediction(self):
        """Spam messages should be flagged."""
        from core.predictor import SpamPredictionEngine
        engine = SpamPredictionEngine(); engine.load_models()
        
        for spam_message in self.test_messages['spam']:
            result = engine.predict_message(spam_message)
            self.assertTrue(result['success'], f"Prediction failed: {spam_message}")
            self.assertTrue(result['is_spam'], f"Missed spam: {spam_message}")
            self.assertGreater(result['confidence'], 0.5, "Low confidence")
    
    def test_ham_prediction(self):
        """Ham messages should be safe."""
        from core.predictor import SpamPredictionEngine
        engine = SpamPredictionEngine(); engine.load_models()
        
        for ham_message in self.test_messages['ham']:
            result = engine.predict_message(ham_message)
            self.assertTrue(result['success'], f"Prediction failed: {ham_message}")
            self.assertFalse(result['is_spam'], f"Wrongly flagged ham: {ham_message}")
            self.assertGreater(result['confidence'], 0.5, "Low confidence")
    
    def test_empty_message_handling(self):
        """Empty/invalid inputs should fail."""
        from core.predictor import SpamPredictionEngine
        engine = SpamPredictionEngine(); engine.load_models()
        
        for test_case in ["", "   ", None]:
            result = engine.predict_message(test_case)
            self.assertFalse(result['success'], f"Should fail for: {test_case}")
    
    def test_model_info(self):
        """Model info should be valid."""
        from core.predictor import SpamPredictionEngine
        engine = SpamPredictionEngine(); engine.load_models()
        info = engine.get_model_info()
        self.assertIn('model_type', info, "Missing model type")
        self.assertIn('feature_count', info, "Missing feature count")
        self.assertGreater(info['feature_count'], 0, "Invalid feature count")

def run_tests():
    """Run integration tests."""
    print("🧪 Running Integration Tests...")
    print("=" * 40)
    
    # Check model existence
    if not os.path.exists('models/spam_classifier.pkl'):
        print("❌ Models not found. Run 'python train_model.py' first")
        return False
    
    # Run suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSpamDetectorIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All integration tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
