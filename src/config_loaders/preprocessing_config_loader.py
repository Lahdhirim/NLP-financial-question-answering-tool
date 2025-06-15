import json
from pydantic import BaseModel, Field

class PreprocessingConfig(BaseModel):
    input_data: str = Field(..., description="Path to the input data file")
    test_size: float = Field(0.2, description="Proportion of the dataset to include in the test split")
    validation_size: float = Field(0.2, description="Proportion of the dataset to include in the validation split")
    training_data_path: str = Field(..., description="Path to save the training data file")
    validation_data_path: str = Field(..., description="Path to save the validation data file")
    test_data_path: str = Field(..., description="Path to save the test data file")

def preprocessing_config_loader(config_path: str) -> PreprocessingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return PreprocessingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find preprocessing config file: {config_path}")