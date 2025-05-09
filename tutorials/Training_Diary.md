# Training Diary

This document tracks the progress, experiments, and observations during the training of the baseline model.

---

## Date: 2025.05.08

### Objective:
- Collect results from the baseline model.

### Configuration:
- **Model Architecture:** HVATNetv3
- **Hyperparameters:**
  - Batch Size: 64
  - Learning Rate: 0.001
  - Augmentation Probability: 0
  - Number of Workers: 0
  - Other Parameters: eval_interval=150
- **Dataset:**
  - Training Dataset: dataset_v2_blocks
  - Test Dataset: fedya_tropin_standart_elbow_left
  - Data Transformations: Default Transformations

### Results:
- **Training Loss:** [e.g., Final loss value or loss curve observations]
- **Validation Accuracy:** [e.g., Accuracy or other metrics]
- **Model Parameters:**
  - Total Parameters: 4.23M
  - Trainable Parameters: 4.23M
  **Time taken:** around 2.6 hr

### Observations:
- [Record any observations during training, e.g., overfitting, underfitting, convergence issues, etc.]

### Post-Processing:
- **Techniques Used:** None for baseline model
- **Effect on Results:** None for baseline model

### Next Steps:
- The original file does not include tracking for Training Loss or Validation Accuracy. It is recommended to implement a function to log and monitor these metrics throughout training like Weight & Bias.
- There is currently no function to record training time. Adding such a function will help evaluate the efficiency of model changes.

---
