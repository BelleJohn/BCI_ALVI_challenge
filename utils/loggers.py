"""
Logger for training metrics using Weights & Biases (wandb).
Code for logging the training process can be embedded directly in the train.py file, but separating it into a standalone module can improve readability and extensibility. Although it may take more time upfront to design the program structure, this approach can ultimately save time in the long run.
"""

from utils.train import TrainConfig, run_train_model
import wandb
import time


# Wrap the original run_train_model function to capture and log metrics
class BasicLogger:
    def __init__(self, model, datasets, config, device):
        self.model = model
        self.datasets = datasets
        self.config = config
        self.device = device
        self.start_time = time.time()
        
    def run(self):
        train_dataset, val_dataset = self.datasets
        
        # Create a hook for monitoring training loss (might change this method)
        def forward_hook(module, input, output):
            loss = output[0].item()  # See def forward() in hvatnet.py to know the item. Here is loss(0) and pred(1)

            #output is loss
            wandb.log({
                "train_loss": loss,
            })
            return output
        
        # Register the forward hook
        forward_hook_handle = self.model.register_forward_hook(forward_hook)
        
        try:
            # Run the original training function
            best_val_loss = run_train_model(self.model, self.datasets, self.config, self.device)
            
            # Log the final metrics
            wandb.log({
                "best_val_loss": best_val_loss,
                "training_time(seconds)": round(time.time() - self.start_time, 2)
            })
            
            return best_val_loss
        finally:
            # Remove the hook
            forward_hook_handle.remove()
            wandb.finish()