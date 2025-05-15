"""
This script is used to train both the baseline model and the modified models on the dataset, with each run focusing on a single change—for example, preprocessing, neural network modifications, or postprocessing. It is essentially the same as the 02_train_baseline.ipynb notebook, except it can be run directly from the terminal, without needing to execute one cell at a time. If a particular modification leads to better training results, I will use Optuna to tune the hyperparameters in order to obtain the optimized model (see train_optuna.py).

Note: 
- To activate the virtual environment in Windows, use the following command in the terminal:
.venv\Scripts\Activate.ps1
or
.\venv\Scripts\activate

- The command to run this script in the terminal is:
-> cd tutorials
-> python train_a_model.py
"""

import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import torch
import wandb

from utils.train import TrainConfig, run_train_model
from utils.augmentations import get_default_transform
from utils import creating_dataset
from utils.loggers import BasicLogger

# this is the implementation of the custom baseline model
from utils import hvatnet

import toml
toml_file = toml.load("../config.toml")

# Define configuration
# Note: On Windows, due to multiprocessing restrictions, please set num_workers to 0.
# On Linux, you can set it to 2 or higher for better performance.
train_config = TrainConfig(exp_name='NumWorker_2', p_augs=0.3, batch_size=64, eval_interval=150, num_workers = 0)

# Initialize Weights & Biases
wandb.init(
    entity=toml_file['wandb']['entity'], # Entity name
    project="BCI_ALVI_challenge",  # Project name
    name=train_config.exp_name,    # Run name
    config={
        "model": "HVATNetv3_baseline",
        "learning_rate": train_config.learning_rate,
        "batch_size": train_config.batch_size,
        "max_steps": train_config.max_steps,
        "augmentation_prob": train_config.p_augs,
        "weight_decay": train_config.weight_decay,
        "grad_clip": train_config.grad_clip,
        "num_workers": train_config.num_workers,
    }
)

# Load DATA_PATH from config.toml
DATA_PATH = toml_file['paths']['DATA_PATH']

## Data preparation
transform = get_default_transform(train_config.p_augs)
data_paths = dict(datasets=[DATA_PATH],
                    hand_type = ['left', 'right'], # [left, 'right']
                    human_type = ['health', 'amputant'], # [amputant, 'health']
                    test_dataset_list = ['fedya_tropin_standart_elbow_left'])
data_config = creating_dataset.DataConfig(**data_paths)
train_dataset, test_dataset = creating_dataset.get_datasets(data_config, transform=transform)

# Model configuration
model_config = hvatnet.Config(n_electrodes=8, n_channels_out=20,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
model = hvatnet.HVATNetv3(model_config)

# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# best_val_loss = run_train_model(model, (train_dataset, test_dataset), train_config, device)
# wandb.log({"best_val_loss": best_val_loss})
# wandb.finish()

wrapper = BasicLogger(model, (train_dataset, test_dataset), train_config, device)
best_val_loss = wrapper.run()
