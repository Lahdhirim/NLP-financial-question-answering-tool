from src.preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
import copy
from src.utils.schema import DataSchema

class CleanDataPreprocessor(BasePreprocessor):
    """Preprocessor to clean the data by removing rows with NaN values and keeping only relevant columns."""

    def keep_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        assert DataSchema.QUESTION in data.columns, "Missing 'question' column in input data"
        assert DataSchema.CONTEXT in data.columns, "Missing 'context' column in input data"
        assert DataSchema.ANSWER in data.columns, "Missing 'answer' column in input data"
        return data[[DataSchema.QUESTION, DataSchema.CONTEXT, DataSchema.ANSWER]]
        
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        clean_data = copy.deepcopy(data)

        # Keep only relevant columns
        clean_data = self.keep_columns(clean_data)

        # Remove rows with NaN values
        clean_data = clean_data.dropna()

        return clean_data