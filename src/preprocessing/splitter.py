from src.preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split 
from typing import Tuple

class SplitDataPreprocessor(BasePreprocessor):
        
    
    def transform(self, data: pd.DataFrame, test_size: float, validation_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the data into training, validation, and test sets."""
        
        # Split the data into training, validation, and test sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=validation_size, random_state=42)
        
        return train_data, val_data, test_data