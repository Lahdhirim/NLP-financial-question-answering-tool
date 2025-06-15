import pandas as pd
from colorama import Fore, Style
import torch
import matplotlib.pyplot as plt

def load_csv_data(data_path: str) -> pd.DataFrame:
        try :
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise print(Fore.RED + f"Could not find the CSV file at {data_path}. Please check the path and try again." + Style.RESET_ALL)
        return data

def save_model(model, save_path: str, epoch: int, val_loss: float) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "state_dict": model_to_save.state_dict(),
        "val_loss": val_loss
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
