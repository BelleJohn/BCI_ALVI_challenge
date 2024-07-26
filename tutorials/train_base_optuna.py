import os
import sys
import torch
import wandb
import optuna
from optuna.trial import TrialState
from dataclasses import dataclass
from pathlib import Path
import math
from safetensors.torch import save_model, load_model
from loguru import logger

sys.path.insert(1, os.path.realpath(os.path.pardir))

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset
from utils import hvatnet

DATA_PATH = "/media/lutetia/Extreme SSD/EMG_Yun/bci-initiative-alvi-hci-challenge/dataset_v2_blocks/dataset_v2_blocks"

def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")
    return n_total, n_trainable

# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    n_res_blocks = trial.suggest_int('n_res_blocks', 1, 5)
    n_blocks_per_layer = trial.suggest_int('n_blocks_per_layer', 1, 5)
    n_filters = trial.suggest_int('n_filters', 16, 128)
    kernel_size = trial.suggest_int('kernel_size', 3, 7)
    dilation = trial.suggest_int('dilation', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128)

    # Model configuration
    model_config = hvatnet.Config(
        n_electrodes=8, n_channels_out=20,
        n_res_blocks=n_res_blocks, n_blocks_per_layer=n_blocks_per_layer,
        n_filters=n_filters, kernel_size=kernel_size,
        strides=(2, 2, 2), dilation=dilation, 
        small_strides=(2, 2),
        dropout_rate=dropout_rate
    )
    model = hvatnet.HVATNetv3(model_config)
    count_parameters(model)

    train_config = TrainConfig(
        exp_name='optuna_run',
        p_augs=0.3,
        batch_size=batch_size,
        eval_interval=150,
        num_workers=0,
        learning_rate=learning_rate
    )

    # Data preparation
    transform = get_default_transform(train_config.p_augs)
    data_paths = dict(datasets=[DATA_PATH],
                      hand_type=['left', 'right'],
                      human_type=['health', 'amputant'],
                      test_dataset_list=['fedya_tropin_standart_elbow_left'])
    data_config = creating_dataset.DataConfig(**data_paths)
    train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model and evaluate
    val_loss = run_train_model(model, (train_dataset, test_dataset), train_config, device, trial)

    return val_loss

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=3600)  # Adjust n_trials and timeout as needed

print('Best hyperparameters:', study.best_params)

# # Retrieve and use the best model
# best_trial = study.best_trial
# best_params = best_trial.params

# best_model_config = hvatnet.Config(
#     n_electrodes=8, n_channels_out=20,
#     n_res_blocks=best_params['n_res_blocks'], n_blocks_per_layer=best_params['n_blocks_per_layer'],
#     n_filters=best_params['n_filters'], kernel_size=best_params['kernel_size'],
#     strides=(2, 2, 2), dilation=best_params['dilation'], 
#     small_strides=(2, 2),
#     dropout_rate=best_params['dropout_rate']
# )
# best_model = hvatnet.HVATNetv3(best_model_config)

# # Continue with training the best model or using it for inference
# transform = get_default_transform(train_config.p_augs)
# data_paths = dict(datasets=[DATA_PATH],
#                   hand_type=['left', 'right'],
#                   human_type=['health', 'amputant'],
#                   test_dataset_list=['fedya_tropin_standart_elbow_left'])
# data_config = creating_dataset.DataConfig(**data_paths)
# train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# run_train_model(best_model, (train_dataset, test_dataset), train_config, device)
