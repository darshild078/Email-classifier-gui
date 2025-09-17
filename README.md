# 🚀 Email Spam Detector

An AI-powered email and SMS spam detection system with advanced natural language processing, machine learning, and a modern GUI interface.

## ✨ Features

- **🧠 Smart AI Detection**: Multinomial Naive Bayes with hyperparameter optimization
- **🔧 Advanced Text Preprocessing**: URL/email/phone preservation, stemming, stopword filtering
- **⚡ Feature Engineering**: TF-IDF vectorization + custom spam-specific features  
- **🖥️ Modern GUI Interface**: Professional Tkinter-based interface with threading
- **📊 Instant Demo**: Pre-loaded sample dataset for immediate testing
- **🔄 Extensible Architecture**: Modular design for easy customization and extension

## 🛠️ Technologies & Libraries

### Core ML & Data Processing
- **Python 3.8+**: Core programming language
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **NLTK**: Natural language processing and text preprocessing
- **pickle**: Model serialization and persistence

### GUI & Interface
- **tkinter**: Native Python GUI framework with ttk styling
- **threading**: Asynchronous operations and background processing
- **pathlib**: Modern file path handling

### Text Processing & NLP
- **Regular Expressions (re)**: Pattern matching and text cleaning
- **NLTK Stopwords**: English stopword filtering with custom preservation
- **NLTK Stemming**: Porter stemmer for word normalization
- **NLTK Tokenization**: Word and sentence tokenization

### Development & Utilities  
- **logging**: Comprehensive application logging
- **os & sys**: System operations and path management
- **traceback**: Error handling and debugging
- **json**: Configuration and metrics storage
- **subprocess**: External process execution

## 📋 Requirements

### Python Version
- **Python 3.8 or higher** (recommended: Python 3.9+)

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/email-classifier.git
cd email-classifier
pip install -r requirements.txt
```

### 2. Download NLTK Data (First Time Only)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 3. Run the GUI (Instant Demo)
```bash
python run_app.py
```

**Demo Features:**
- 📧 **Ham Example**: Load random legitimate message samples
- 🚨 **Spam Example**: Load random spam message samples  
- 🔍 **Analyze Message**: Get AI-powered spam predictions
- ⚡ **Real-time Results**: Instant analysis with confidence scores

### 4. Train Your Own Model (Optional)
```bash
# 1. Download SMS Spam Collection dataset
# 2. Place spam.csv in data/raw/
# 3. Run training pipeline
python train_model.py
# 4. Restart GUI to use your model
python run_app.py
```

## 📁 Project Architecture

```
Email-Classifier/
├── 📁 config/              # Model configuration files
│   └── model_config.py     # ML hyperparameters & settings
├── 📁 data/
│   ├── 📁 raw/             # Original datasets (spam.csv)
│   ├── 📁 processed/       # Cleaned & preprocessed data
│   └── 📁 samples/         # Demo sample messages (20 included)
├── 📁 models/              # Trained model artifacts
│   ├── spam_classifier.pkl # Trained Naive Bayes model
│   ├── feature_engineer.pkl # TF-IDF + feature pipeline
│   └── model_metrics.json  # Training performance metrics
├── 📁 src/
│   ├── 📁 core/            # ML pipeline components
│   │   ├── data_loader.py   # Dataset loading & validation
│   │   ├── preprocessor.py  # Text cleaning & preprocessing
│   │   ├── feature_engineer.py # Feature extraction (TF-IDF + custom)
│   │   ├── model_trainer.py # Model training & hyperparameter tuning
│   │   └── predictor.py     # Real-time spam prediction engine
│   ├── 📁 gui/             # User interface components
│   │   ├── main_window.py   # Main application window
│   │   ├── components.py    # Reusable UI components
│   │   └── styles.py        # Modern theme & styling
│   └── 📁 utils/           # Utilities & logging
│       └── logger.py        # Application logging setup
├── 📁 tests/               # Test scripts & integration tests
├── 📁 docs/                # Project documentation
├── 📄 run_app.py           # 🎯 GUI application launcher
├── 📄 train_model.py       # 🤖 Model training script
├── 📄 requirements.txt     # Python dependencies
└── 📄 README.md           # This file
```

## 🎯 Machine Learning Pipeline

### 1. Data Preprocessing (`src/core/preprocessor.py`)
- **URL/Email/Phone Preservation**: Converts to meaningful tokens
- **Text Normalization**: Lowercase, punctuation handling
- **Stemming**: Porter stemmer for word reduction
- **Stopword Filtering**: English stopwords with spam-word preservation
- **Noise Removal**: Clean whitespace and invalid characters

### 2. Feature Engineering (`src/core/feature_engineer.py`)
- **TF-IDF Vectorization**: Unigrams + bigrams (5000 features)
- **Custom Spam Features**: 
  - Money/currency mentions
  - Urgent language detection
  - URL/email/phone counts
  - Exclamation patterns
  - Message length metrics

### 3. Model Training (`src/core/model_trainer.py`)
- **Algorithm**: Multinomial Naive Bayes
- **Hyperparameter Tuning**: Grid search for optimal alpha
- **Cross-Validation**: 5-fold stratified validation
- **Performance Metrics**: Accuracy, precision, recall, F1-score

### 4. Prediction Engine (`src/core/predictor.py`)
- **Real-time Inference**: Fast text-to-prediction pipeline
- **Confidence Scoring**: Probability estimates for predictions
- **Error Handling**: Robust error management and fallbacks

## 🎨 GUI Features

### Modern Interface Design
- **Professional Theme**: Custom color scheme and typography
- **Responsive Layout**: Resizable panels and components
- **Status Indicators**: Real-time model loading and analysis status
- **Threading**: Non-blocking UI with background processing

### User Experience
- **Instant Demo**: Sample buttons for immediate testing
- **Real-time Analysis**: Live spam detection with confidence scores
- **Clear Results Display**: Color-coded predictions with detailed metrics
- **Error Handling**: User-friendly error messages and guidance

## 📊 Performance

### Model Accuracy
- **Training Accuracy**: ~98%+ on SMS Spam Collection
- **Cross-Validation**: Consistent performance across folds
- **Real-time Speed**: <100ms inference time
- **Memory Efficient**: Optimized for desktop deployment

### Preprocessing Features
- **Signal Preservation**: Maintains spam-indicative patterns
- **Robust Cleaning**: Handles various text formats and encodings
- **Feature Engineering**: 5000+ TF-IDF features + 15 custom indicators

## 🔧 Customization

### Model Configuration (`config/model_config.py`)
```python
MODEL_CONFIG = {
    'algorithm': 'MultinomialNB',  # Change to 'LogisticRegression', 'SVM'
    'alpha': 1.0,                  # Smoothing parameter  
    'cv_folds': 5,                 # Cross-validation folds
    'test_size': 0.2,              # Train/test split ratio
}

TFIDF_CONFIG = {
    'max_features': 5000,          # Vocabulary size
    'ngram_range': (1, 2),         # Unigrams + bigrams
    'min_df': 2,                   # Minimum document frequency
    'max_df': 0.95,                # Maximum document frequency
}
```

### Adding New Features
Extend `src/core/feature_engineer.py`:
```python
def _extract_custom_features(self, messages):
    # Add your custom spam indicators
    features = []
    for message in messages:
        crypto_mentions = len(re.findall(r'bitcoin|crypto', message.lower()))
        features.append([crypto_mentions])
    return features
```

### GUI Customization
Modify themes in `src/gui/styles.py`:
```python
COLORS = {
    'primary': '#your_color',      # Change primary theme color
    'success': '#your_color',      # Change success indicators
    'danger': '#your_color',       # Change warning colors
}
```

## 🧪 Testing

### Run Demo Tests
```bash
# Test with sample data
python run_app.py
# Click "Ham Example" or "Spam Example" buttons
# Click "Analyze Message" to see predictions
```

### Train with Custom Data
```bash
# Place your dataset in data/raw/spam.csv
# Format: label,message (where label is 'ham' or 'spam')
python train_model.py
```

### Integration Testing
```bash
python -m pytest tests/ -v
```

## 📚 Documentation

Detailed guides available in `docs/`:
- **Dataset Guide**: How to obtain and prepare training data
- **Training Guide**: Model training and hyperparameter tuning  
- **Customization Guide**: Extending and modifying the system
- **API Reference**: Code documentation and examples

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
git clone https://github.com/yourusername/email-classifier.git
cd email-classifier
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository**: SMS Spam Collection dataset
- **NLTK Team**: Natural language processing tools
- **scikit-learn**: Machine learning library and algorithms
- **Python Community**: Amazing ecosystem and libraries

## 📞 Support & Contact

- 📧 **Email**: darshild078@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/darshild078/Email-classifier-gui/issues)
- 📖 **Documentation**: See `docs/` folder
- 💬 **Discussions**: [GitHub Discussions](https://github.com/darshild078/Email-classifier-gui/discussions)

## 🌟 Star History

⭐ **Star this repository if you find it useful!** ⭐

It helps others discover the project and motivates continued development.

---

**Built with ❤️ using Python, scikit-learn, and modern software engineering practices.**