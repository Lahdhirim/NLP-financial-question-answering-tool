from torch import nn
from torch.optim import Adam
from transformers import T5ForConditionalGeneration
from colorama import Fore, Style
from src.utils.toolbox import choose_device

class ModelBuilder:
    def __init__(
        self,
        model_name: str = "t5-base",
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
        enable_gpu: bool = True,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.enable_gpu = enable_gpu
  
    def _print_trainable_parameters(self, model: nn.Module) -> None:
        print(f"{Fore.CYAN}Trainable parameters:{Style.RESET_ALL}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")

    def build_model(self) -> nn.Module:
        # Load model
        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            return_dict=True
        )
        print(f"{Fore.CYAN}T5 Model loaded.{Style.RESET_ALL}")

        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print(f"{Fore.CYAN}Encoder parameters frozen.{Style.RESET_ALL}")
        self._print_trainable_parameters(model)

        return model

    def build_optimizer(self, model: nn.Module) -> Adam:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=self.learning_rate)
        return optimizer
    
    def initialize(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model=model)
        device = choose_device(enable_gpu=self.enable_gpu)
        return model, optimizer, device








    

