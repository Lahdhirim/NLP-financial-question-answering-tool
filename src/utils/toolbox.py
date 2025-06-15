import pandas as pd
from colorama import Fore, Style
import torch
import matplotlib.pyplot as plt
from src.utils.schema import CheckpointSchema
from typing import Tuple
from transformers import PreTrainedTokenizerBase, T5TokenizerFast, T5ForConditionalGeneration

def load_csv_data(data_path: str) -> pd.DataFrame:
        try :
            data = pd.read_csv(data_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(
                Fore.RED + f"Could not find the CSV file at {data_path}. Please check the path and try again." + Style.RESET_ALL
            )

def save_model(
        model, 
        model_name: str, 
        tokenizer_pretrained_model: str, 
        save_path: str, 
        epoch: int, 
        val_loss: float,
        max_input_length: int,
        max_answer_length: int
) -> None:
    
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        CheckpointSchema.EPOCH: epoch,
        CheckpointSchema.VAL_LOSS: val_loss,
        CheckpointSchema.STATE_DICT: model_to_save.state_dict(),
        CheckpointSchema.MODEL_NAME: model_name,
        CheckpointSchema.TOKENIZER_PRETRAINED_MODEL: tokenizer_pretrained_model,
        CheckpointSchema.MAX_INPUT_LENGTH: max_input_length,
        CheckpointSchema.MAX_ANSWER_LENGTH: max_answer_length
    }
    torch.save(checkpoint, save_path)
    print(Fore.MAGENTA + f"Model saved at {save_path} with validation loss: {val_loss:.4f}." + Style.RESET_ALL)
    return None

def plot_training_and_validation_losses(train_losses: list, val_losses: list, save_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(Fore.MAGENTA + f"Graph saved at {save_path}." + Style.RESET_ALL)
    return None

def load_trained_model(save_path: str) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase, int, int]:
    try:
        checkpoint = torch.load(save_path)

        model = T5ForConditionalGeneration.from_pretrained(checkpoint[CheckpointSchema.MODEL_NAME], return_dict=True)
        model.load_state_dict(checkpoint[CheckpointSchema.STATE_DICT])
        
        tokenizer = T5TokenizerFast.from_pretrained(checkpoint[CheckpointSchema.TOKENIZER_PRETRAINED_MODEL])

        max_input_length = checkpoint[CheckpointSchema.MAX_INPUT_LENGTH]
        max_answer_length = checkpoint[CheckpointSchema.MAX_ANSWER_LENGTH]

        print(Fore.MAGENTA + f"Model loaded from {save_path} with validation loss: {checkpoint[CheckpointSchema.VAL_LOSS]:.4f}." + Style.RESET_ALL)
        return model, tokenizer, max_input_length, max_answer_length
    
    except FileNotFoundError:
        raise FileNotFoundError(
            Fore.RED + f"Could not find the model at {save_path}. Please check the path and try again." + Style.RESET_ALL
        )
    except KeyError as e:
        raise KeyError(Fore.RED + f"Missing key in checkpoint: {e}" + Style.RESET_ALL)


def choose_device(enable_gpu):
    if enable_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"{Fore.CYAN}Using device: {device}{Style.RESET_ALL}")
    return device



     
