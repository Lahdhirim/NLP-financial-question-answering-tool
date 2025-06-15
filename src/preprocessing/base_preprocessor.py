from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    """Abstract base class for preprocessing steps implementations"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to the input data."""
        pass