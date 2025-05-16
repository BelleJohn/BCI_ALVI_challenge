"""
Logger for training metrics using Weights & Biases (wandb).
Code for logging the training process can be embedded directly in the train.py file, but separating it into a standalone module can improve readability and extensibility. Although it may take more time upfront to design the program structure, this approach can ultimately save time in the long run.
"""

from utils.train import TrainConfig, run_train_model
import wandb
import time
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score

# Wrap the original run_train_model function to capture and log metrics
class BasicLogger:
    def __init__(self, model, datasets, config, device):
        self.model = model
        self.datasets = datasets
        self.config = config
        self.device = device
        self.start_time = time.time()
        self.step = 0
        
    def run(self):
        train_dataset, val_dataset = self.datasets
        
        # Create a hook for monitoring training loss and metrics
        def forward_hook(module, input, output):
            self.step += 1
            loss = output[0].item()  # Original L1 loss
            pred = output[1]  
            target = input[1]
            
            # Basic metrics - directly with tensors
            mse = F.mse_loss(pred, target).item()
            mae = F.l1_loss(pred, target).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            # Calculate max error
            max_error = torch.max(torch.abs(pred - target)).item()
            
            # Log basic metrics
            metrics = {
                "train_loss": loss,
                "train_mse": mse,
                "train_mae": mae,
                "train_rmse": rmse,
                "train_max_error": max_error,
                "step": self.step
            }
            
            # Every 100 steps, calculate more expensive metrics
            if self.step % 100 == 0:
                # Move tensors to CPU and convert to numpy for sklearn metrics
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                
                # Calculate R² score and explained variance
                # Reshape to 2D for sklearn compatibility
                pred_2d = pred_np.reshape(-1, pred_np.shape[-1])
                target_2d = target_np.reshape(-1, target_np.shape[-1])
                
                try:
                    r2 = r2_score(target_2d, pred_2d)
                    exp_var = explained_variance_score(target_2d, pred_2d)
                    
                    metrics.update({
                        "train_r2_score": r2,
                        "train_explained_variance": exp_var
                    })
                    
                    # Per-joint metrics (for first few joints)
                    for i in range(pred_np.shape[1]):  #min(5, pred_np.shape[1])# Track first 5 joints
                        joint_mse = F.mse_loss(pred[:, i, :], target[:, i, :]).item()
                        metrics[f"train_joint_{i}_mse"] = joint_mse
                except:
                    # Skip if there's an error in calculation
                    pass
                
                # Perform validation and log metrics
                self.log_validation_metrics()
            
            wandb.log(metrics)
            return output
        
        # Register the forward hook
        forward_hook_handle = self.model.register_forward_hook(forward_hook)
        
        try:
            # Run the original training function
            best_val_loss = run_train_model(self.model, self.datasets, self.config, self.device)
            
            # Final validation
            final_metrics = self.log_validation_metrics(final=True)
            
            # Log the final metrics
            wandb.log({
                "best_val_loss": best_val_loss,
                "training_time_seconds": round(time.time() - self.start_time, 2),
                **final_metrics
            })
            
            return best_val_loss
        finally:
            # Remove the hook
            forward_hook_handle.remove()
            wandb.finish()
            
    def log_validation_metrics(self, final=False):
        """Calculate and log validation metrics."""
        self.model.eval()
        val_dataset = self.datasets[1]
        
        all_preds = []
        all_targets = []
        val_losses = []
        
        # Process a subset of validation data for regular checks
        max_samples = 10 if not final else len(val_dataset)
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataset):
                if i >= max_samples:
                    break
                
                # Convert NumPy arrays to tensors if needed
                if isinstance(inputs, np.ndarray):
                    inputs = torch.FloatTensor(inputs)
                if isinstance(targets, np.ndarray):
                    targets = torch.FloatTensor(targets)
                
                # Check dimensions and reshape if needed
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(0)  # Add batch dimension [channels, time] -> [1, channels, time]
                
                if targets.dim() == 2:
                    targets = targets.unsqueeze(0)  # Add batch dimension
                
                # Now move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                try:
                    # Forward pass
                    val_loss, predictions = self.model(inputs, targets)
                    
                    # Store results
                    val_losses.append(val_loss.item())
                    all_preds.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                except RuntimeError as e:
                    print(f"Error in validation batch {i}: {e}")
                    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                    continue
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        val_mse = F.mse_loss(all_preds, all_targets).item()
        val_mae = F.l1_loss(all_preds, all_targets).item()
        val_rmse = torch.sqrt(torch.tensor(val_mse)).item()
        val_max_error = torch.max(torch.abs(all_preds - all_targets)).item()
        
        metrics = {
            "val_loss": np.mean(val_losses),
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_max_error": val_max_error
        }
        
        # Calculate advanced metrics
        try:
            # Reshape for sklearn
            preds_np = all_preds.numpy().reshape(-1, all_preds.shape[-1])
            targets_np = all_targets.numpy().reshape(-1, all_targets.shape[-1])
            
            val_r2 = r2_score(targets_np, preds_np)
            val_exp_var = explained_variance_score(targets_np, preds_np)
            
            metrics.update({
                "val_r2_score": val_r2,
                "val_explained_variance": val_exp_var
            })
            
            # Per-joint metrics for the first few joints
            for i in range(min(5, all_preds.shape[1])):
                joint_mse = F.mse_loss(all_preds[:, i, :], all_targets[:, i, :]).item()
                metrics[f"val_joint_{i}_mse"] = joint_mse
        except:
            # Skip if there's an error
            pass
        
        # Only log if not the final evaluation
        if not final:
            wandb.log(metrics)
        
        self.model.train()
        return metrics