from src.config_loaders.training_config_loader import TrainingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data, save_model, plot_training_and_validation_losses
from src.modeling.dataset import FinancialQADataset
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, RandomSampler
from src.modeling.model import ModelBuilder
from src.utils.schema import ModelSchema
import torch

class TrainingPipeline(BasePipeline):    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
    
    def run(self):

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")

        # Load the data
        print(f"{Fore.YELLOW}Loading data from specified paths...{Style.RESET_ALL}")
        train_data = load_csv_data(data_path=self.config.training_data_path)
        validation_data = load_csv_data(data_path=self.config.validation_data_path)

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

        # Create the model, optimizer and select device
        print(f"{Fore.YELLOW}Creating model and optimizer...{Style.RESET_ALL}")
        model_builder = ModelBuilder(model_name=self.config.model.model_name,
                                     learning_rate=self.config.model.learning_rate,
                                     freeze_encoder=self.config.model.freeze_encoder,
                                     enable_gpu=self.config.model.enable_gpu)
        model, optimizer, device = model_builder.initialize()
        model.to(device)

        # Train the model
        print(f"{Fore.YELLOW}Starting training loop...{Style.RESET_ALL}")

        lowest_val_loss = None
        train_losses = []
        val_losses = []

        n_epochs = self.config.n_epochs
        for epoch in range(n_epochs):
            print(f"{Fore.YELLOW}Epoch {epoch}...{Style.RESET_ALL}")

            # Train the model on the training set
            train_loss = 0
            model.train()
            for batch in train_loader:
                input_ids = batch[ModelSchema.INPUT_IDS].to(device)
                attention_mask = batch[ModelSchema.ENCODER_MASK].to(device)
                labels = batch[ModelSchema.LABELS].to(device)
                decoder_attention_mask = batch[ModelSchema.DECODER_MASK].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels, 
                                decoder_attention_mask=decoder_attention_mask)
                outputs.loss.backward()
                optimizer.step()
                train_loss += outputs.loss.item()
            
            mean_train_loss = train_loss/len(train_loader) # Average train loss per batch
            train_losses.append(mean_train_loss)
            print(f"{Fore.CYAN}Train Loss: {mean_train_loss:.4f}{Style.RESET_ALL}")

            # Evaluate the model on validation set
            val_loss = 0
            model.eval()
            for batch in validation_loader:
                input_ids = batch[ModelSchema.INPUT_IDS].to(device)
                attention_mask = batch[ModelSchema.ENCODER_MASK].to(device)
                labels = batch[ModelSchema.LABELS].to(device)
                decoder_attention_mask = batch[ModelSchema.DECODER_MASK].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
                    val_loss += outputs.loss.item()
            
            mean_val_loss = val_loss/len(validation_loader) # Average validation loss per batch
            val_losses.append(mean_val_loss)
            print(f"{Fore.CYAN}Validation Loss: {mean_val_loss:.4f}{Style.RESET_ALL}")

            # Save the model with the lowest validation loss
            if lowest_val_loss is None or mean_val_loss < lowest_val_loss:
                save_model(model=model,
                           model_name=self.config.model.model_name,
                           tokenizer_pretrained_model=self.config.model.tokenizer_pretrained_model,
                           save_path=self.config.best_model_path, 
                           epoch=epoch,
                           val_loss=mean_val_loss,
                           max_input_length=self.config.model.max_input_length,
                           max_answer_length=self.config.model.max_answer_length
                           )


                lowest_val_loss = mean_val_loss
            
            # Save the training and validation loss curve after each epoch (to keep track of progress)
            if len(train_losses) >= 2:
                print(f"{Fore.YELLOW}Plotting losses...{Style.RESET_ALL}")
                plot_training_and_validation_losses(train_losses=train_losses,
                                                val_losses=val_losses,
                                                save_path=self.config.losses_curve_path)
            
        print(f"{Fore.GREEN}Training completed with the best validation loss: {lowest_val_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")