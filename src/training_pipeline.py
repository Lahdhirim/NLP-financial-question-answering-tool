from config_loaders.training_config_loader import TrainingConfig
from colorama import Fore, Style
import pandas as pd
from src.utils.schema import DataSchema

class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def run(self):

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")


        # Load and split the data
        print(f"{Fore.YELLOW}Loading data from: {self.config.input_data}{Style.RESET_ALL}")
        try:
            data = pd.read_csv(self.config.input_data)
            assert DataSchema.QUESTION in data.columns, "Missing 'question' column in input data"
            assert DataSchema.CONTEXT in data.columns, "Missing 'context' column in input data"
            assert DataSchema.ANSWER in data.columns, "Missing 'answer' column in input data"
            data = data[[DataSchema.QUESTION, DataSchema.CONTEXT, DataSchema.ANSWER]]
            data = data.dropna()
            print(f"{Fore.GREEN}Data loaded successfully!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Data shape: {data.shape}{Style.RESET_ALL}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find input data file: {self.config.input_data}")
            