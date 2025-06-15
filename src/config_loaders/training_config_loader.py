import json
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    input_data: str = Field(..., description="Path to the input data file")

def training_config_loader(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return TrainingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find training config file: {config_path}")