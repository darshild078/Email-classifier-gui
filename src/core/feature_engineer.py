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
from utils.logger import setup_logger
from config.model_config import TFIDF_CONFIG

logger = setup_logger(__name__)

class SpamFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert text into features using TF-IDF and message patterns."""

    def __init__(self, use_tfidf=True, use_message_features=True, use_spam_features=True, tfidf_config=None):
        # Store options
        self.use_tfidf = use_tfidf
        self.use_message_features = use_message_features
        self.use_spam_features = use_spam_features  # New: dedicated spam features
        self.tfidf_config = tfidf_config or TFIDF_CONFIG
        self.tfidf_vectorizer = None
        self.is_fitted = False

        if self.use_tfidf:
            self._initialize_tfidf()
        logger.info(f"Feature engineer ready: TF-IDF={use_tfidf}, Extra features={use_message_features}, Spam features={use_spam_features}")

    def _initialize_tfidf(self):
        # Enhanced TF-IDF setup for spam detection
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_config.get('max_features', 5000),  # Increased from 3000
            ngram_range=self.tfidf_config.get('ngram_range', (1, 2)),
            min_df=self.tfidf_config.get('min_df', 2),
            max_df=self.tfidf_config.get('max_df', 0.95),
            lowercase=self.tfidf_config.get('lowercase', True),
            stop_words=self.tfidf_config.get('stop_words', None),  # Handle in preprocessing
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,  # Log scaling helps with spam detection
            norm='l2'
        )
        logger.info("TF-IDF vectorizer initialized")

    def fit(self, X, y=None):
        # Learn vocabulary from messages
        if self.use_tfidf and self.tfidf_vectorizer:
            self.tfidf_vectorizer.fit(X)
            vocab_size = len(self.tfidf_vectorizer.vocabulary_)
            logger.info(f"TF-IDF fitted with {vocab_size} words")
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
            
        if self.use_spam_features:
            spam_features = self._extract_spam_features(X)
            features_list.append(spam_features)
        
        # Combine all feature types
        if len(features_list) == 1:
            final_features = features_list[0]
        else:
            final_features = hstack(features_list, format='csr')
            
        logger.info(f"Feature matrix shape: {final_features.shape}")
        return final_features

    def _extract_message_features(self, messages):
        # Basic message-level features
        features = []
        
        for message in messages:
            msg = str(message).lower() if message else ""
            original_msg = str(message) if message else ""
            length = len(msg)
            words = msg.split()
            word_count = len(words)
            
            # Character ratios
            capital_ratio = sum(1 for c in original_msg if c.isupper()) / length if length else 0
            digit_ratio = sum(1 for c in msg if c.isdigit()) / length if length else 0
            punctuation_ratio = sum(1 for c in msg if c in '!?.,;:') / length if length else 0
            
            # Word statistics
            avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
            max_word_length = max(len(w) for w in words) if words else 0
            
            # Punctuation counts
            exclamation_count = msg.count('!')
            question_count = msg.count('?')
            
            features.append([
                length, word_count, capital_ratio, digit_ratio, punctuation_ratio,
                avg_word_length, max_word_length, exclamation_count, question_count
            ])
            
        return csr_matrix(np.array(features, dtype=np.float32))

    def _extract_spam_features(self, messages):
        # Enhanced spam-specific features
        features = []
        
        # Expanded spam keywords with weights
        high_spam_words = [
            'free', 'win', 'winner', 'cash', 'prize', 'money', 'urgent', 'limited',
            'offer', 'click', 'call', 'text', 'stop', 'claim', 'congratulations',
            'guaranteed', 'risk', 'act', 'now', 'hurry', 'expires', 'bonus',
            'discount', 'save', 'cheap', 'deal', 'credit', 'loan', 'debt'
        ]
        
        medium_spam_words = [
            'amazing', 'incredible', 'fantastic', 'special', 'exclusive',
            'selected', 'chosen', 'lucky', 'opportunity', 'chance'
        ]
        
        for message in messages:
            msg = str(message).lower() if message else ""
            words = msg.split()
            word_count = len(words)
            
            # Spam keyword features
            high_spam_count = sum(1 for word in high_spam_words if word in msg)
            medium_spam_count = sum(1 for word in medium_spam_words if word in msg)
            total_spam_words = high_spam_count + medium_spam_count
            
            spam_word_ratio = total_spam_words / word_count if word_count else 0
            high_spam_ratio = high_spam_count / word_count if word_count else 0
            
            # Preprocessing placeholder detection (from enhanced preprocessor)
            has_url_marker = 1 if 'hasurl' in msg else 0
            has_email_marker = 1 if 'hasemail' in msg else 0
            has_phone_marker = 1 if 'hasphone' in msg else 0
            has_money_marker = 1 if 'money' in msg else 0
            has_urgent_marker = 1 if 'urgent' in msg else 0
            
            # Pattern-based features
            all_caps_words = sum(1 for word in str(message).split() if word.isupper() and len(word) > 1)
            caps_word_ratio = all_caps_words / word_count if word_count else 0
            
            # Special character patterns
            repeated_chars = len([c for c in msg if msg.count(c) > 3])
            multiple_exclamations = 1 if '!!' in msg else 0
            multiple_questions = 1 if '??' in msg else 0
            
            # Urgency indicators
            urgency_words = ['now', 'today', 'immediately', 'asap', 'quickly', 'fast']
            urgency_count = sum(1 for word in urgency_words if word in msg)
            
            # Time-sensitive words
            time_words = ['limited', 'expires', 'deadline', 'until', 'before', 'ends']
            time_pressure_count = sum(1 for word in time_words if word in msg)
            
            features.append([
                high_spam_count, medium_spam_count, spam_word_ratio, high_spam_ratio,
                has_url_marker, has_email_marker, has_phone_marker, has_money_marker,
                has_urgent_marker, caps_word_ratio, repeated_chars, multiple_exclamations,
                multiple_questions, urgency_count, time_pressure_count
            ])
            
        return csr_matrix(np.array(features, dtype=np.float32))

    def get_feature_names(self):
        # Return all feature names for analysis
        names = []
        
        if self.use_tfidf and self.tfidf_vectorizer:
            names.extend(self.tfidf_vectorizer.get_feature_names_out().tolist())
            
        if self.use_message_features:
            names.extend([
                'message_length', 'word_count', 'capital_ratio', 'digit_ratio',
                'punctuation_ratio', 'avg_word_length', 'max_word_length',
                'exclamation_count', 'question_count'
            ])
            
        if self.use_spam_features:
            names.extend([
                'high_spam_count', 'medium_spam_count', 'spam_word_ratio', 'high_spam_ratio',
                'has_url_marker', 'has_email_marker', 'has_phone_marker', 'has_money_marker',
                'has_urgent_marker', 'caps_word_ratio', 'repeated_chars', 'multiple_exclamations',
                'multiple_questions', 'urgency_count', 'time_pressure_count'
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
