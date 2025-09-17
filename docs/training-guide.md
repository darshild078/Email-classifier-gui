# 🤖 Training Guide

## 1. Complete ML Pipeline - 6-Stage Training Process

### Stage 1: Data Loading & Validation
- Load dataset using **pandas** from CSV/TSV files
- Validate data integrity (check columns, labels, missing values)
- Split into train/test sets with **scikit-learn**

### Stage 2: Text Preprocessing  
- Clean and normalize text using **NLTK** and regex
- Remove noise while preserving spam signals
- Tokenization, stemming, stopword filtering

### Stage 3: Feature Engineering
- Extract TF-IDF features using **scikit-learn**
- Add custom spam-specific features
- Create sparse feature matrices with **numpy**

### Stage 4: Model Training
- Train algorithms with **scikit-learn** (Naive Bayes, SVM, etc.)
- Hyperparameter optimization using GridSearchCV
- Cross-validation for robust evaluation

### Stage 5: Model Evaluation
- Test performance metrics (accuracy, precision, recall, F1)
- Generate confusion matrices and classification reports
- Validate against performance thresholds

### Stage 6: Model Persistence
- Save trained models using **pickle**
- Store metrics and configuration in **json**
- Create reproducible model artifacts

## 2. Technology Stack

### Core ML Libraries
- **scikit-learn**: Machine learning algorithms, metrics, model selection
- **pandas**: Data manipulation, CSV handling, DataFrame operations  
- **numpy**: Numerical operations, array processing, sparse matrices
- **NLTK**: Natural language processing, tokenization, stemming

### Persistence & Configuration
- **pickle**: Model serialization and deserialization
- **json**: Configuration files, metrics storage, hyperparameters

### System Libraries
- **pathlib**: Modern file path handling
- **os**: File system operations
- **logging**: Training progress tracking and debugging

## 3. Configuration Examples with Hyperparameter Tuning

### Model Configuration (`config/model_config.py`)
```python
# Core algorithm settings
MODEL_CONFIG = {
    'algorithm': 'MultinomialNB',     # Primary algorithm
    'alpha': 1.0,                    # Smoothing parameter
    'test_size': 0.2,                # Train/test split ratio
    'cv_folds': 5,                   # Cross-validation folds
    'random_state': 42               # Reproducibility seed
}

# Hyperparameter optimization grid
HYPERPARAMETER_GRID = {
    'MultinomialNB': {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }
}

# TF-IDF feature extraction settings
TFIDF_CONFIG = {
    'max_features': 5000,            # Vocabulary size
    'ngram_range': (1, 2),           # Unigrams + bigrams
    'min_df': 2,                     # Min document frequency
    'max_df': 0.95,                  # Max document frequency
    'stop_words': 'english'          # Remove English stopwords
}

# Performance thresholds for validation
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.95,            # Minimum accuracy requirement
    'min_precision_spam': 0.90,      # Spam detection precision
    'min_recall_spam': 0.85,         # Spam detection recall
    'min_f1_score': 0.87             # Overall F1 score
}
```

### Training Script Example
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

def train_spam_detector():
    # Stage 1: Data Loading
    print("📂 Loading dataset...")
    df = pd.read_csv('data/raw/spam.csv', sep='\t', 
                     names=['label', 'message'], encoding='utf-8')
    
    # Stage 2: Data Preprocessing (simplified)
    print("🔧 Preprocessing text...")
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    # Stage 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Stage 4: Feature Engineering
    print("⚡ Extracting features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_features = tfidf.fit_transform(X_train)
    X_test_features = tfidf.transform(X_test)
    
    # Stage 5: Model Training with Hyperparameter Tuning
    print("🤖 Training model...")
    param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_features, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best alpha: {grid_search.best_params_['alpha']}")
    
    # Stage 6: Evaluation
    print("📊 Evaluating model...")
    y_pred = best_model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {accuracy:.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))
    
    # Save model and artifacts
    with open('models/spam_classifier.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'vectorizer': tfidf}, f)
    
    metrics = {
        'accuracy': float(accuracy),
        'best_alpha': float(grid_search.best_params_['alpha']),
        'cv_score': float(grid_search.best_score_)
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("💾 Model and metrics saved successfully!")

if __name__ == "__main__":
    train_spam_detector()
```

## 4. Alternative Algorithms

### Logistic Regression Implementation
```python
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("🔄 Training Logistic Regression...")
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    y_pred = best_lr.predict(X_test)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    return best_lr
```

### SVM Implementation
```python
from sklearn.svm import SVC

def train_svm_classifier(X_train, y_train, X_test, y_test):
    print("⚙️ Training SVM...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm_model = SVC(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    return best_svm
```

### Random Forest Implementation
```python
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, X_test, y_test):
    print("🌲 Training Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_model = RandomForestClassifier(class_weight='balanced', 
                                    random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    return best_rf
```

## 5. Performance Optimization & Memory Management

### Memory Optimization Techniques
```python
import gc
from scipy.sparse import csr_matrix

def optimize_memory_usage():
    # Use sparse matrices for TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, dtype=np.float32)
    
    # Process data in chunks for large datasets
    chunk_size = 1000
    for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_chunk(chunk)
        
        # Clear intermediate variables
        del chunk
        gc.collect()
    
    # Use memory-efficient data types
    df['label'] = df['label'].astype('category')
    
    # Free unused memory
    del unnecessary_variables
    gc.collect()
```

### Speed Optimization
```python
def optimize_training_speed():
    # Use parallel processing
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=3,           # Reduce CV folds for speed
        n_jobs=-1,      # Use all CPU cores
        verbose=1       # Show progress
    )
    
    # Reduce feature dimensions if needed
    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=3000)  # Top 3000 features
    X_selected = selector.fit_transform(X, y)
    
    # Use incremental learning for large datasets
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    
    # Partial fit in batches
    for batch_X, batch_y in get_batches(X, y, batch_size=1000):
        model.partial_fit(batch_X, batch_y)
```

## 6. Troubleshooting Section

### Common Issues & Solutions

**Issue 1: Low Accuracy (<90%)**
```python
# Solutions:
# 1. Check data quality
print(f"Label distribution: {df['label'].value_counts()}")
print(f"Average message length: {df['message'].str.len().mean()}")

# 2. Increase vocabulary size
TFIDF_CONFIG['max_features'] = 10000

# 3. Try different algorithms
MODEL_CONFIG['algorithm'] = 'LogisticRegression'

# 4. Add more features
# Include custom spam features (URLs, caps, exclamations)
```

**Issue 2: Memory Errors**
```python
# Solutions:
# 1. Reduce feature dimensions
TFIDF_CONFIG['max_features'] = 2000

# 2. Use float32 instead of float64
tfidf = TfidfVectorizer(dtype=np.float32)

# 3. Process in batches
def process_in_batches(data, batch_size=500):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]
```

**Issue 3: Training Too Slow**
```python
# Solutions:
# 1. Reduce hyperparameter search space
HYPERPARAMETER_GRID = {
    'alpha': [0.1, 1.0, 10.0]  # Fewer values to test
}

# 2. Use fewer CV folds
MODEL_CONFIG['cv_folds'] = 3

# 3. Enable parallel processing
grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
```

**Issue 4: Poor Spam Recall**
```python
# Solutions:
# 1. Adjust class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', classes=np.unique(y), y=y
)

model = LogisticRegression(class_weight='balanced')

# 2. Tune decision threshold
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
optimal_threshold = thresholds[np.argmax(recalls >= 0.90)]
```

## 7. Validation Scripts & Debug Mode

### Model Validation Script
```python
#!/usr/bin/env python3
"""Validate trained model performance and artifacts."""

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

def validate_model_artifacts():
    """Check if all required model files exist and are valid."""
    
    required_files = [
        'models/spam_classifier.pkl',
        'models/model_metrics.json'
    ]
    
    print("🔍 Validating model artifacts...")
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing file: {file_path}")
            return False
        
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Valid pickle file: {file_path}")
            
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ Valid JSON file: {file_path}")
        
        except Exception as e:
            print(f"❌ Corrupted file {file_path}: {e}")
            return False
    
    return True

def validate_model_performance():
    """Check if model meets minimum performance thresholds."""
    
    print("📊 Validating model performance...")
    
    with open('models/model_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    thresholds = {
        'accuracy': 0.95,
        'cv_score': 0.90
    }
    
    for metric, threshold in thresholds.items():
        if metric in metrics:
            value = metrics[metric]
            if value >= threshold:
                print(f"✅ {metric}: {value:.3f} >= {threshold}")
            else:
                print(f"❌ {metric}: {value:.3f} < {threshold}")
                return False
        else:
            print(f"⚠️ Missing metric: {metric}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting model validation...")
    
    if validate_model_artifacts() and validate_model_performance():
        print("🎉 All validations passed!")
        exit(0)
    else:
        print("💥 Validation failed!")
        exit(1)
```

### Debug Mode Configuration
```python
import logging
import sys

def setup_debug_logging():
    """Enable detailed debug logging."""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training_debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Enable scikit-learn verbose output
    import warnings
    warnings.simplefilter('always')

def debug_training_pipeline():
    """Run training with extensive debugging."""
    
    setup_debug_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug("🔍 Starting debug training pipeline...")
        
        # Add debug prints throughout training
        logger.debug(f"Dataset shape: {df.shape}")
        logger.debug(f"Feature matrix shape: {X_train.shape}")
        logger.debug(f"Best CV score: {grid_search.best_score_}")
        
        # Save debug information
        debug_info = {
            'dataset_info': {
                'total_samples': len(df),
                'spam_samples': sum(df['label'] == 'spam'),
                'ham_samples': sum(df['label'] == 'ham')
            },
            'feature_info': {
                'feature_count': X_train.shape[1],
                'vocab_size': len(tfidf.vocabulary_)
            },
            'training_info': {
                'best_params': grid_search.best_params_,
                'cv_scores': grid_search.cv_results_
            }
        }
        
        with open('logs/debug_info.json', 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        
        logger.debug("✅ Debug training completed")
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise

# Usage: python train_model.py --debug
if '--debug' in sys.argv:
    debug_training_pipeline()
```

### Performance Monitoring
```python
import time
import psutil
import os

def monitor_training_performance():
    """Monitor system resources during training."""
    
    process = psutil.Process(os.getpid())
    
    def log_system_stats():
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        print(f"📈 Memory usage: {memory_mb:.1f} MB")
        print(f"🖥️ CPU usage: {cpu_percent:.1f}%")
    
    # Monitor at different training stages
    print("📊 System stats at training start:")
    log_system_stats()
    
    start_time = time.time()
    
    # Your training code here
    train_model()
    
    training_time = time.time() - start_time
    
    print(f"⏱️ Total training time: {training_time:.1f} seconds")
    print("📊 System stats at training end:")
    log_system_stats()
```

---

**This comprehensive training guide covers all aspects of model development using the specified technology stack. Each section includes practical code examples and real-world solutions for building robust spam detection systems.**