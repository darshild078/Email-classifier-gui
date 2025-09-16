"""
Feature engineering for spam detection.
Transforms messages into numeric features for ML.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, hstack
import pickle
import os
from ..utils.logger import setup_logger
from config.model_config import TFIDF_CONFIG

logger = setup_logger(__name__)

class SpamFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert text into features using TF-IDF and message patterns."""

    def __init__(self, use_tfidf=True, use_message_features=True, tfidf_config=None):
        # Store options
        self.use_tfidf = use_tfidf
        self.use_message_features = use_message_features
        self.tfidf_config = tfidf_config or TFIDF_CONFIG
        self.tfidf_vectorizer = None
        self.is_fitted = False

        if self.use_tfidf:
            self._initialize_tfidf()
        logger.info(f"Feature engineer ready: TF-IDF={use_tfidf}, Extra features={use_message_features}")

    def _initialize_tfidf(self):
        # Set up TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_config['max_features'],
            ngram_range=self.tfidf_config['ngram_range'],
            min_df=self.tfidf_config['min_df'],
            max_df=self.tfidf_config['max_df'],
            lowercase=self.tfidf_config['lowercase'],
            stop_words=self.tfidf_config['stop_words'],
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        logger.info("TF-IDF vectorizer initialized")

    def fit(self, X, y=None):
        # Learn vocabulary from messages
        if self.use_tfidf and self.tfidf_vectorizer:
            self.tfidf_vectorizer.fit(X)
            logger.info(f"TF-IDF fitted with {len(self.tfidf_vectorizer.vocabulary_)} words")
        self.is_fitted = True
        return self

    def transform(self, X):
        # Convert messages to feature matrix
        if not self.is_fitted:
            raise ValueError("Call fit() before transform()")
        
        features_list = []

        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(X)
            features_list.append(tfidf_features)
        
        if self.use_message_features:
            message_features = self._extract_message_features(X)
            features_list.append(message_features)
        
        # Combine TF-IDF and extra features
        final_features = hstack(features_list, format='csr') if len(features_list) > 1 else features_list[0]
        logger.info(f"Feature matrix shape: {final_features.shape}")
        return final_features

    def _extract_message_features(self, messages):
        # Create extra features like length, capitals, digits, punctuation, spam keywords
        features = []
        spam_keywords = [
            'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'limited', 
            'offer', 'click', 'call', 'text', 'stop', 'claim', 'congratulations',
            'guaranteed', 'risk-free', 'act now', 'hurry', 'expires'
        ]
        for message in messages:
            msg = str(message).lower() if message else ""
            length = len(msg)
            words = msg.split()
            word_count = len(words)
            capital_ratio = sum(1 for c in str(message) if c.isupper()) / length if length else 0
            digit_ratio = sum(1 for c in msg if c.isdigit()) / length if length else 0
            punctuation_ratio = sum(1 for c in msg if c in '!?.,;:') / length if length else 0
            spam_keyword_count = sum(1 for k in spam_keywords if k in msg)
            spam_keyword_ratio = spam_keyword_count / word_count if word_count else 0
            exclamation_count = msg.count('!')
            question_count = msg.count('?')
            has_url = 1 if 'http' in msg or 'www' in msg else 0
            has_phone = 1 if any(c.isdigit() for c in msg.replace(' ', '')) else 0
            avg_word_length = sum(len(w) for w in words)/word_count if word_count else 0

            features.append([
                length, word_count, capital_ratio, digit_ratio, punctuation_ratio,
                spam_keyword_count, spam_keyword_ratio, exclamation_count,
                question_count, has_url, has_phone, avg_word_length
            ])
        return csr_matrix(np.array(features, dtype=np.float32))

    def get_feature_names(self):
        # Return all feature names
        names = []
        if self.use_tfidf and self.tfidf_vectorizer:
            names.extend(self.tfidf_vectorizer.get_feature_names_out().tolist())
        if self.use_message_features:
            names.extend([
                'message_length', 'word_count', 'capital_ratio', 'digit_ratio',
                'punctuation_ratio', 'spam_keyword_count', 'spam_keyword_ratio',
                'exclamation_count', 'question_count', 'has_url', 'has_phone',
                'avg_word_length'
            ])
        return names

    def save_feature_engineer(self, filepath):
        # Save object for later use
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Feature engineer saved: {filepath}")

    @classmethod
    def load_feature_engineer(cls, filepath):
        # Load saved object
        with open(filepath, 'rb') as f:
            fe = pickle.load(f)
        logger.info(f"Feature engineer loaded: {filepath}")
        return fe
