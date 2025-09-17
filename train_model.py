import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.data_loader import SpamDataLoader
from core.preprocessor import TextPreprocessor
from core.feature_engineer import SpamFeatureEngineer
from core.model_trainer import SpamModelTrainer
from utils.logger import setup_logger
from config.model_config import TFIDF_CONFIG, MODEL_CONFIG

logger = setup_logger(__name__)

def check_prerequisites() -> bool:
    """Check if dataset, directories, and Python packages are ready."""
    logger.info("Checking prerequisites...")
    missing_items = []

    # Check dataset
    if not os.path.exists('data/raw/spam.csv'):
        missing_items.append("Dataset (data/raw/spam.csv)")

    # Ensure directories exist
    for folder in ['data/processed', 'models', 'logs']:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Created folder: {folder}")

    # Check essential packages
    try:
        import pandas, sklearn, nltk
        logger.info("✓ All required packages available")
    except ImportError as e:
        missing_items.append(f"Python package: {e}")

    if missing_items:
        logger.error("Missing prerequisites:")
        for item in missing_items:
            logger.error(f"  - {item}")
        return False

    logger.info("✓ All prerequisites met")
    return True

def main() -> bool:
    """Run the complete spam detection training pipeline."""
    logger.info("=== SPAM DETECTION MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    try:
        # Step 0: Check prerequisites
        if not check_prerequisites():
            logger.error("[ERROR] Prerequisites not met. Aborting training.")
            return False

        # Step 1: Load data
        logger.info("\n STEP 1: Data Loading")
        data_loader = SpamDataLoader('data/raw/spam.csv')
        df = data_loader.load_dataset()
        if df is None:
            logger.error("[ERROR] Failed to load dataset.")
            return False
        logger.info("Dataset loaded successfully!")

        # Step 2: Preprocess text
        logger.info("\n STEP 2: Text Preprocessing")
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        processed_path = 'data/processed/cleaned_data.csv'
        df_processed.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved: {processed_path}")

        # Step 3: Feature engineering
        logger.info("\n STEP 3: Feature Engineering")
        feature_engineer = SpamFeatureEngineer(
            use_tfidf=True, use_message_features=True, tfidf_config=TFIDF_CONFIG
        )
        X = feature_engineer.fit_transform(df_processed['processed_message'])
        y = df_processed['label'].values
        feature_names = feature_engineer.get_feature_names()
        feature_engineer.save_feature_engineer('models/feature_engineer.pkl')
        logger.info(f"Feature engineering done: {len(feature_names)} features")

        # Step 4: Train & evaluate model
        logger.info("\n STEP 4: Model Training & Evaluation")
        model_trainer = SpamModelTrainer(config=MODEL_CONFIG)
        results = model_trainer.train_and_evaluate(X, y, feature_names)

        # Step 5: Validate and save model
        logger.info("\n STEP 5: Model Validation & Saving")
        if results['validation_results']['passed']:
            logger.info("[OK] Model passed all quality checks!")
            model_metadata = model_trainer.save_model_and_metrics(results)
            test_metrics = results['test_metrics']
            logger.info("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.1%}")
            logger.info(f"  Precision: {test_metrics['precision']:.1%}")
            logger.info(f"  Recall: {test_metrics['recall']:.1%}")
            logger.info(f"  F1-Score: {test_metrics['f1_score']:.1%}")
            logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.1%}")
        else:
            logger.error("[ERROR] Model failed quality validation!")
            for check, result in results['validation_results']['checks'].items():
                if not result['passed']:
                    logger.error(f"  • {check}: {result['actual']:.4f} < {result['threshold']:.4f}")
            return False

        return True

    except Exception as e:
        logger.error(f"[ERROR] Training pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
