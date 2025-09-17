import pandas as pd
import os
from typing import Optional

class SpamDataLoader:
    """Load and prepare spam detection dataset with proper label encoding."""

    def __init__(self, data_path: str = "data/raw/spam.csv"):
        self.data_path = data_path

    def load_dataset(self) -> Optional[pd.DataFrame]:
        """Load dataset and return DataFrame with encoded labels."""
        try:
            if not os.path.exists(self.data_path):
                print(f"Dataset file not found: {self.data_path}")
                return None
            
            df = pd.read_csv(self.data_path, encoding="latin-1")
            df = self._standardize_columns(df)
            if df is None:
                return None

            df = self._encode_labels(df)
            if df is None:
                return None

            if not self._validate_labels(df):
                return None

            df = self._clean_data(df)
            print(f"Dataset loaded: {df.shape}")
            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Convert different column formats to standard 'label' and 'message'."""
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
        elif "label" in df.columns and "message" in df.columns:
            df = df[["label", "message"]]
        else:
            print(f"Unsupported columns: {df.columns.tolist()}")
            return None
        return df

    def _encode_labels(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Map text labels to numeric values: ham=0, spam=1."""
        print("Encoding labels...")
        original_dist = df['label'].value_counts()
        print(f"Original: {original_dist.to_dict()}")
        
        # Convert to lowercase and map to numbers
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        label_mapping = {'ham': 0, 'spam': 1, '0': 0, '1': 1}
        df['label'] = df['label'].map(label_mapping)
        
        # Remove unmapped labels
        missing_count = df['label'].isna().sum()
        if missing_count > 0:
            print(f"Removing {missing_count} invalid labels")
            df = df.dropna(subset=['label'])
        
        df['label'] = df['label'].astype(int)
        final_dist = df['label'].value_counts().sort_index()
        print(f"Final: {final_dist.to_dict()}")
        return df

    def _validate_labels(self, df: pd.DataFrame) -> bool:
        """Check that we have both spam (1) and ham (0) labels."""
        unique_labels = set(df['label'].unique())
        if unique_labels != {0, 1}:
            print(f"Expected labels [0,1], found: {unique_labels}")
            return False
        
        spam_count = (df['label'] == 1).sum()
        ham_count = (df['label'] == 0).sum()
        
        if spam_count == 0:
            print("No spam messages found - this causes 0% detection!")
            return False
        if ham_count == 0:
            print("No ham messages found!")
            return False
            
        print(f"Labels valid: {ham_count} ham, {spam_count} spam")
        return True

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty messages and duplicates."""
        initial_size = len(df)
        df = df.dropna(subset=['message'])
        df = df[df['message'].str.strip().str.len() > 0]
        df = df.drop_duplicates(subset=['message'], keep='first')
        
        removed = initial_size - len(df)
        if removed > 0:
            print(f"Cleaned: removed {removed} problematic rows")
        return df

    def dataset_info(self, df: pd.DataFrame) -> None:
        """Print dataset summary."""
        if df is None or df.empty:
            print("Dataset is empty")
            return
            
        print("Dataset Summary")
        print("---------------")
        print(f"Total messages: {len(df)}")
        
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "Ham" if label == 0 else "Spam"
            pct = (count / len(df)) * 100
            print(f"{label_name}: {count} ({pct:.1f}%)")
        
        avg_length = df['message'].str.len().mean()
        print(f"Average message length: {avg_length:.1f} chars")

    def validate_dataset_quality(self, df: pd.DataFrame) -> bool:
        """Check if dataset is suitable for training."""
        if df is None or df.empty:
            return False
            
        if len(df) < 100:
            print("Dataset too small (need 100+ messages)")
            return False
            
        spam_count = (df['label'] == 1).sum()
        if spam_count == 0:
            print("No spam messages - cannot train detector")
            return False
            
        return True


if __name__ == "__main__":
    loader = SpamDataLoader("data/raw/spam.csv")
    df = loader.load_dataset()
    
    if df is not None:
        loader.dataset_info(df)
        if loader.validate_dataset_quality(df):
            print("Dataset ready for training!")
        else:
            print("Dataset has quality issues")
    else:
        print("Failed to load dataset")
        print("Download SMS Spam Collection and save as data/raw/spam.csv")
