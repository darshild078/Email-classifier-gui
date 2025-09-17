import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """
    Cleans raw text messages and prepares them for ML models.
    """

    def __init__(self, remove_stopwords=True, apply_stemming=True, min_word_length=2, *args, **kwargs):
        # Accept all possible parameters for backward compatibility
        self.keep_words = {"not", "no", "now", "free", "win", "your", "you", "call", "click"}
        self.stop_words = set(stopwords.words("english")) - self.keep_words if remove_stopwords else set()
        self.stemmer = PorterStemmer() if apply_stemming else None
        self.min_word_length = min_word_length
        
        # Log any extra parameters for debugging (optional)
        if kwargs:
            print(f"TextPreprocessor received extra params: {list(kwargs.keys())}")

    def _clean_message(self, text: str) -> str:
        """Clean a single message step by step."""
        if not isinstance(text, str):
            return ""

        # Replace URLs and emails with meaningful placeholders
        text = re.sub(r"http\S+|www\.\S+", " hasurl ", text)
        text = re.sub(r"\S+@\S+", " hasemail ", text)
        
        # Handle phone numbers
        text = re.sub(r"\b\d{10,}\b", " hasphone ", text)
        text = re.sub(r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b", " hasphone ", text)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", " hasphone ", text)

        text = text.lower()

        # Preserve money symbols as meaningful tokens
        text = re.sub(r"[$£€¥]", " money ", text)
        text = re.sub(r"[!]{2,}", " urgent ", text)
        text = re.sub(r"[?]{2,}", " question ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = word_tokenize(text)
        words = [w for w in words if len(w) >= self.min_word_length]

        if self.stop_words:
            words = [w for w in words if w not in self.stop_words]

        if self.stemmer:
            words = [self.stemmer.stem(w) for w in words]

        return " ".join(words)

    def transform(self, messages):
        """Clean a list/series of messages."""
        if isinstance(messages, pd.Series):
            messages = messages.tolist()
        return [self._clean_message(m) for m in messages]

    def preprocess_dataframe(self, df, msg_col="message", label_col="label"):
        """Clean a dataset and return ready-to-use DataFrame."""
        df = df.copy()
        df["processed_message"] = self.transform(df[msg_col])

        # Robust label encoding with validation
        if df[label_col].dtype != 'int64':
            print(f"Converting labels from {df[label_col].unique()} to numeric")
            df[label_col] = df[label_col].astype(str).str.lower().str.strip()
            df[label_col] = df[label_col].map({"ham": 0, "spam": 1})
            
            missing_labels = df[label_col].isna().sum()
            if missing_labels > 0:
                print(f"Warning: {missing_labels} labels could not be mapped, removing them")
                df = df.dropna(subset=[label_col])
            
            df[label_col] = df[label_col].astype(int)

        initial_count = len(df)
        df = df[df["processed_message"].str.strip() != ""]
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} empty messages after preprocessing")
            
        label_counts = df[label_col].value_counts()
        print(f"Final dataset: {len(df)} messages")
        print(f"Label distribution: {label_counts.to_dict()}")
        
        if len(label_counts) < 2:
            print("Warning: Only one label type remaining after preprocessing!")
            
        return df[["processed_message", label_col]]
