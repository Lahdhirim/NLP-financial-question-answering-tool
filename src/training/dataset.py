from torch.utils.data import Dataset
import torch

class FinancialQADataset(Dataset):

    def __init__(self, tokenizer, data, max_length_question=256, max_length_answer=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length_question = max_length_question
        self.max_length_answer = max_length_answer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question = self.data.iloc[idx]["question"]
        context = self.data.iloc[idx]["context"]
        answer = self.data.iloc[idx]["answer"]
        
        question_tokenized = self.tokenizer(question, context, max_length=self.max_length_question, padding="max_length", truncation=True)
        answer_tokenized  = self.tokenizer(answer, max_length=self.max_length_answer, padding="max_length", truncation=True)
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "encoder_attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(answer_tokenized["input_ids"], dtype=torch.long),
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)}
    