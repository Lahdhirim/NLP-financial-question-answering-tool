import json
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    tokenizer_pretrained_model: str = Field(default="t5-base", description="Pretrained model name for the tokenizer")
    max_input_length: Optional[int] = Field(..., description="Maximum token length for input question + context")
    max_answer_length: Optional[int] = Field(..., description="Maximum token length for the answer")
    batch_size: int = Field(..., description="Batch size required for Dataloader")

class TrainingConfig(BaseModel):
    training_data_path: str = Field(..., description="Path to load the training data file")
    validation_data_path: str = Field(..., description="Path to load the validation data file")
    test_data_path: str = Field(..., description="Path to load the test data file")
    model: ModelConfig = Field(..., description="Model-related configuration")

def training_config_loader(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return TrainingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find training config file: {config_path}")