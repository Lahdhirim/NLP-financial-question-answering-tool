from src.config_loaders.testing_config_loader import TestingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data, load_trained_model, choose_device
from src.modeling.dataset import FinancialQADataset
from torch.utils.data import DataLoader, RandomSampler
from src.utils.schema import ModelSchema, MetricSchema
import torch
from src.evaluators.metrics import MetricEvaluator


class TestingPipeline(BasePipeline):    
    def __init__(self, config: TestingConfig):
        super().__init__(config)
    
    def run(self):

        print(f"{Fore.GREEN}Starting testing pipeline...{Style.RESET_ALL}")

        # Load the data
        print(f"{Fore.YELLOW}Loading data from specified paths...{Style.RESET_ALL}")
        test_data = load_csv_data(data_path=self.config.test_data_path)

        # Load the model
        print(f"{Fore.YELLOW}Loading trained model from {self.config.trained_model_path}{Style.RESET_ALL}")
        model, tokenizer, max_input_length, max_answer_length = load_trained_model(save_path=self.config.trained_model_path)
        
        # Create the test dataset object
        test_dataset = FinancialQADataset(
            tokenizer=tokenizer,
            data=test_data,
            max_input_length=max_input_length,
            max_answer_length=max_answer_length
        )

        # Create the test DataLoader object
        print(f"{Fore.YELLOW}Creating Test DataLoader object...{Style.RESET_ALL}")
        test_loader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=self.config.batch_size
        )

        # Evaluate the model
        model.eval()
        device = choose_device(enable_gpu=self.config.enable_gpu)
        model.to(device)

        predictions = []
        ground_truths = []
        for batch in test_loader:
            input_ids = batch[ModelSchema.INPUT_IDS].to(device)
            attention_mask = batch[ModelSchema.ENCODER_MASK].to(device)
            labels = batch[ModelSchema.LABELS].to(device)
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_answer_length)
                predictions.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
                ground_truths.extend([tokenizer.decode(label, skip_special_tokens=True) for label in labels])
        
        evaluator = MetricEvaluator()
        rouge_scores = evaluator.compute_rouge(predictions, ground_truths)
        bleu_scores = evaluator.compute_bleu(predictions, ground_truths)
        print(f"{Fore.CYAN}ROUGE-1: {rouge_scores[MetricSchema.ROUGE1]:.4f}, ROUGE-2: {rouge_scores[MetricSchema.ROUGE2]:.4f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}BLEU-1: {bleu_scores[MetricSchema.BLEU1]:.4f}, BLEU-2: {bleu_scores[MetricSchema.BLEU2]:.4f}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
            