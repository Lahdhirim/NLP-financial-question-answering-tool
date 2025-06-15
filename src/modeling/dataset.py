from torch.utils.data import Dataset
import torch
from src.utils.schema import DataSchema, ModelSchema

class FinancialQADataset(Dataset):

    def __init__(self, tokenizer, data, max_input_length=256, max_answer_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_answer_length = max_answer_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question = self.data.iloc[idx][DataSchema.QUESTION]
        context = self.data.iloc[idx][DataSchema.CONTEXT]
        answer = self.data.iloc[idx][DataSchema.ANSWER]
        
        input_tokenized = self.tokenizer(question, context, max_length=self.max_input_length, padding="max_length", truncation=True)
        answer_tokenized  = self.tokenizer(answer, max_length=self.max_answer_length, padding="max_length", truncation=True)
        
        return {
            ModelSchema.INPUT_IDS: torch.tensor(input_tokenized["input_ids"], dtype=torch.long),
            ModelSchema.ENCODER_MASK: torch.tensor(input_tokenized["attention_mask"], dtype=torch.long),
            ModelSchema.LABELS: torch.tensor(answer_tokenized["input_ids"], dtype=torch.long),
            ModelSchema.DECODER_MASK: torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)}
    