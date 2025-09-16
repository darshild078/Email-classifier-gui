"""
Machine Learning Model Configuration.
Purpose: Centralize all ML hyperparameters for easy tuning.
"""

# TF-IDF Vectorizer Settings
TFIDF_CONFIG = {
    'max_features': 3000,        # Vocabulary size limit
    'ngram_range': (1, 2),       # Use unigrams and bigrams  
    'min_df': 2,                 # Ignore rare terms
    'max_df': 0.95,              # Ignore too common terms
    'lowercase': True,           # Case normalization
    'stop_words': 'english'      # Remove stopwords
}

# Model Training Settings
MODEL_CONFIG = {
    'algorithm': 'MultinomialNB',  # Naive Bayes for text
    'alpha': 1.0,                  # Laplace smoothing
    'test_size': 0.2,              # 80/20 train/test split
    'random_state': 42,            # Reproducible results
    'cv_folds': 5                  # Cross-validation folds
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.90,          # Minimum acceptable accuracy
    'min_precision_spam': 0.85,    # Minimize false spam detection
    'min_recall_spam': 0.80        # Catch most spam messages
}

# File Paths
MODEL_PATHS = {
    'model_file': 'models/spam_classifier.pkl',
    'vectorizer_file': 'models/vectorizer.pkl', 
    'metrics_file': 'models/model_metrics.json',
    'training_data': 'data/processed/training_data.csv'
}
