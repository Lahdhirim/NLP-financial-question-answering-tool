from src.config_loaders.preprocessing_config_loader import PreprocessingConfig
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from colorama import Fore, Style
from src.preprocessing.clean_data import CleanDataPreprocessor
from src.preprocessing.splitter import SplitDataPreprocessor
from src.utils.schema import DataSchema

class PreprocessingPipeline(BasePipeline):
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
    
    def run(self):
        
        print(f"{Fore.GREEN}Starting preprocessing pipeline...{Style.RESET_ALL}")

        # Load the input data
        print(f"{Fore.YELLOW}Loading data from: {self.config.input_data}{Style.RESET_ALL}")
        data = load_csv_data(data_path=self.config.input_data)
        print(f"{Fore.CYAN}Data shape before preprocessing: {data.shape}{Style.RESET_ALL}")

        # Clean the data
        print(f"{Fore.YELLOW}Cleaning data...{Style.RESET_ALL}")
        clean_data_preprocessor = CleanDataPreprocessor()
        clean_data = clean_data_preprocessor.transform(data)
        print(f"{Fore.CYAN}Data shape after cleaning: {clean_data.shape}{Style.RESET_ALL}")

        # Statistics about the length of the question, context, and answer
        print(f"{Fore.CYAN}Maximum lengths - Question: {clean_data[DataSchema.QUESTION].str.len().max()}, Context: {clean_data[DataSchema.CONTEXT].str.len().max()}, Answer: {clean_data[DataSchema.ANSWER].str.len().max()}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Minimum lengths - Question: {clean_data[DataSchema.QUESTION].str.len().min()}, Context: {clean_data[DataSchema.CONTEXT].str.len().min()}, Answer: {clean_data[DataSchema.ANSWER].str.len().min()}{Style.RESET_ALL}")

        # Split the data into training, validation, and test sets
        print(f"{Fore.YELLOW}Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
        splitter = SplitDataPreprocessor()
        train_data, val_data, test_data = splitter.transform(
            data=clean_data,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size
        )
        print(f"{Fore.CYAN}Data shapes after splitting - Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}{Style.RESET_ALL}")

        # Save the split data to the specified paths
        print(f"{Fore.YELLOW}Saving data to specified paths...{Style.RESET_ALL}")
        train_data.to_csv(self.config.training_data_path, index=False)
        val_data.to_csv(self.config.validation_data_path, index=False)
        test_data.to_csv(self.config.test_data_path, index=False)

        print(f"{Fore.GREEN}Preprocessing pipeline completed successfully!{Style.RESET_ALL}")

