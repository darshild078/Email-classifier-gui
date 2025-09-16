import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """
    Cleans raw text messages and prepares them for ML models.
    """

    def __init__(self, remove_stopwords=True, apply_stemming=True, min_word_length=2):
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if apply_stemming else None
        self.min_word_length = min_word_length

    def _clean_message(self, text: str) -> str:
        """Clean a single message step by step."""
        if not isinstance(text, str):
            return ""

        # remove urls, emails, phone numbers
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\b\d{10,}\b", "", text)

        # lowercase everything
        text = text.lower()

        # keep only alphabets and spaces
        text = re.sub(r"[^a-z\s]", " ", text)

        # split into words
        words = word_tokenize(text)

        # filter small words
        words = [w for w in words if len(w) >= self.min_word_length]

        # remove stopwords
        if self.stop_words:
            words = [w for w in words if w not in self.stop_words]

        # stem words
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

        # convert labels: ham=0, spam=1
        df[label_col] = df[label_col].map({"ham": 0, "spam": 1})

        # remove empty rows
        df = df[df["processed_message"].str.strip() != ""]
        return df[["processed_message", label_col]]
