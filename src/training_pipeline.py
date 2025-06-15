from src.config_loaders.training_config_loader import TrainingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from src.modeling.dataset import FinancialQADataset
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, RandomSampler

class TrainingPipeline(BasePipeline):    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
    
    def run(self):

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")


        # Load the data
        print(f"{Fore.YELLOW}Loading data to specified paths...{Style.RESET_ALL}")
        train_data = load_csv_data(data_path=self.config.training_data_path)
        validation_data = load_csv_data(data_path=self.config.validation_data_path)
        test_data = load_csv_data(data_path=self.config.test_data_path)

        # Create the dataset objects
        tokenizer = T5TokenizerFast.from_pretrained(self.config.model.tokenizer_pretrained_model)
        print(f"{Fore.YELLOW}Creating dataset objects...{Style.RESET_ALL}")
        train_dataset = FinancialQADataset(
            tokenizer=tokenizer,
            data=train_data,
            max_input_length=self.config.model.max_input_length,
            max_answer_length=self.config.model.max_answer_length
        )
        validation_dataset = FinancialQADataset(
            tokenizer=tokenizer,
            data=validation_data,
            max_input_length=self.config.model.max_input_length,
            max_answer_length=self.config.model.max_answer_length
        )
        test_dataset = FinancialQADataset(
            tokenizer=tokenizer,
            data=test_data,
            max_input_length=self.config.model.max_input_length,
            max_answer_length=self.config.model.max_answer_length
        )

        # Create the DataLoader objects
        print(f"{Fore.YELLOW}Creating DataLoader objects...{Style.RESET_ALL}")
        train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.config.model.batch_size
        )
        validation_loader = DataLoader(
            validation_dataset,
            sampler=RandomSampler(validation_dataset),
            batch_size=self.config.model.batch_size
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=self.config.model.batch_size
        )
        print("Example of train batch: ", next(iter(train_loader)))
        breakpoint()
        
            