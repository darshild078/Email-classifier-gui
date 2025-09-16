import pandas as pd

class SpamDataLoader:
    """
    It makes sure the dataset has only two columns: 'label' and 'message'.
    """

    def __init__(self, data_path="spam.csv"):
        self.data_path = data_path

    def load_dataset(self):
        """Load CSV and return a DataFrame with standardized columns."""
        try:
            df = pd.read_csv(self.data_path, encoding="latin-1")

            # Fix column names depending on dataset style
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            elif "label" in df.columns and "message" in df.columns:
                df = df[["label", "message"]]
            else:
                raise ValueError("Dataset must have either ['v1','v2'] or ['label','message'].")

            return df

        except Exception as e:
            print("Could not load dataset:", e)
            return pd.DataFrame()

    def dataset_info(self, df):
        """Print a quick summary of the dataset."""
        print("Dataset Summary")
        print("----------------")
        print("Total messages:", len(df))
        print("Labels count:\n", df["label"].value_counts())
        print("Average message length:", round(df["message"].str.len().mean(), 2))
