# Training Diary

This document tracks the progress, experiments, and observations during the training of the baseline model.

---
## Date: 2025.05.19-2025.05.20, 2025.06.04, 
[Link of training](https://wandb.ai/belle/BCI_ALVI_challenge/workspace?nw=nwuserbelle)

### Objective:
- Study preprocessing of EMG signals
- Apply and test augmentations

### Configuration:
- **Model Architecture:** HVATNetv3
- **Hyperparameters:**
  - Batch Size: 64
  - Learning Rate: 0.001
  - Augmentation Probability: 0
  - Number of Workers: 0 (due to Windows system)
  - Other Parameters: eval_interval=150
- **Dataset:**
  - Training Dataset: dataset_v2_blocks
  - Test Dataset: fedya_tropin_standart_elbow_left
  - Data Transformations: Default Transformations

### Results:
- **Training Loss:** L1 loss 
- **Validation Metrics:** MSE, MAE, RMSE

- **Model Parameters:**
  - Total Parameters: 4.23M
  - Trainable Parameters: 4.23M
  - Time taken: 18.6 mins

### Observations/Note:
- Preprocessing focuses on cleaning, aligning, and normalizing.
- Augmentation focuses on creating variations for better generalization.

Analysis of Wavelet Augmentation Impact on Model Performance
- Results Comparison
| Metric | Without Augmentation | With Augmentation | Improvement |
|--------|:--------------------:|:-----------------:|:-----------:|
| val_loss | 0.3336 | 0.3314 | 0.64% ↓ |
| val_mae | 0.3336 | 0.3314 | 0.64% ↓ |
| val_max_error | 2.2293 | 2.1796 | 2.23% ↓ |
| val_mse | 0.2441 | 0.2391 | 2.03% ↓ |
| val_r2_score | 0.1889 | 0.2053 | 8.68% ↑ |
| val_rmse | 0.4941 | 0.4890 | 1.02% ↓ |

- Conclusion
The data augmentation strategy has yielded consistent improvements across all evaluation metrics:

1. General Performance Improvement: All error metrics decreased with augmentation, demonstrating that the model generalizes better to unseen data.

2. Better Outlier Handling: The 2.23% reduction in max_error is particularly noteworthy, suggesting that augmentation helps the model better handle extreme cases.

3. Improved Explanatory Power: The R² score increased by 8.68%, which is the most substantial improvement. This indicates that your augmented model captures significantly more of the variance in the target variable.

4. Balanced Augmentation Strategy: The improvements suggest your chosen augmentations (Gaussian noise, spatial rotation, and wavelet noise injection) strike a good balance - they introduce enough variability to improve generalization without distorting the underlying signal patterns.

5. Validation of Wavelet Approach: The inclusion of WaveletNoiseInjection appears to be beneficial, supporting the theoretical advantage of adding structured noise that respects the time-frequency characteristics of EMG signals.

These results validate the augmentation approach and suggest that data augmentation is an effective strategy for improving EMG-based motion prediction models.

### Modification:
- **Techniques Used:** None for baseline model
- **Effect on Results:** None for baseline model

### Next Steps:
- Study and test post-processing

### Reference:

1. Reaz, M. B. I., Hussain, M. S., & Mohd-Yasin, F. (2006). Techniques of EMG signal analysis: detection, processing, classification and applications. *Biological Procedures Online, 8*, 11–35. [DOI](https://link.springer.com/article/10.1251/bpo115)


---

## Date: 2025.05.12-2025.05.16
[Link of training](https://wandb.ai/belle/BCI_ALVI_challenge/workspace?nw=nwuserbelle)

### Objective:
- Track training process on Weight & Bias.
- Decide the method to evaluate model performance
- Track training time

### Configuration:
- **Model Architecture:** HVATNetv3
- **Hyperparameters:**
  - Batch Size: 64
  - Learning Rate: 0.001
  - Augmentation Probability: 0
  - Number of Workers: 0 (due to Windows system)
  - Other Parameters: eval_interval=150
- **Dataset:**
  - Training Dataset: dataset_v2_blocks
  - Test Dataset: fedya_tropin_standart_elbow_left
  - Data Transformations: Default Transformations

### Results:
- **Training Loss:** L1 loss 
- **Validation Metrics:** MSE, MAE, RMSE
    1. MSE (Mean Squared Error)
    - Standard regression metric
    - Emphasizes larger errors (due to squaring)
    - Lower is better
    2. MAE (Mean Absolute Error)
    - More interpretable than MSE
    - Represents average error magnitude in original units
    - Less sensitive to outliers than MSE
    - Lower is better
    3. RMSE (Root Mean Square Error)
    - Same units as your target variable
    - Combines MSE's sensitivity to outliers with MAE's interpretability
    - Lower is better
- **Model Parameters:**
  - Total Parameters: 4.23M
  - Trainable Parameters: 4.23M
  - Time taken: 18.6 mins

### Observations:
- [Record any observations during training, e.g., overfitting, underfitting, convergence issues, etc.]

### Modification:
- **Techniques Used:** None for baseline model
- **Effect on Results:** None for baseline model

### Next Steps:
- Test different augmentation methods

### Reference:
The evaluation metrics reference:

1. Akhtar, A., Aghasadeghi, N., Hargrove, L., & Bretl, T. (2017). Estimation of distal arm joint angles from EMG and shoulder orientation for transhumeral prostheses. *Journal of Electromyography and Kinesiology, 35*, 86-94. [DOI](https://europepmc.org/article/MED/28624687)

2. Guo, S., Yang, Z., & Liu, Y. (2019). EMG-based Continuous Prediction of the Upper Limb Elbow Joint Angle Using GRNN. *2019 IEEE International Conference on Mechatronics and Automation (ICMA)*. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/8816401/)

3. Coker, J., Chen, H., Schall, M. C., Gallagher, S., & Zabala, M. (2021). EMG and Joint Angle-Based Machine Learning to Predict Future Joint Angles at the Knee. *Sensors, 21*(11), 3622. [DOI](https://www.mdpi.com/1424-8220/21/11/3622)

4. Kumar, R., Muthukrishnan, S. P., Kumar, L., & Roy, S. (2023). Predicting Multi-Joint Kinematics of the Upper Limb from EMG Signals Across Varied Loads with a Physics-Informed Neural Network. *arXiv preprint arXiv:2312.09418*. [arXiv](https://arxiv.org/abs/2312.09418)

5. Ma, X., Liu, Y., Song, Q., & Wang, C. (2020). Continuous Estimation of Knee Joint Angle Based on Surface Electromyography Using a Long Short-Term Memory Neural Network and Time-Advanced Feature. *Sensors, 20*(17), 4966. [DOI](https://www.mdpi.com/1424-8220/20/17/4966)

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

### Modification:
- **Techniques Used:** None for baseline model
- **Effect on Results:** None for baseline model

### Next Steps:
- The original file does not include tracking for Training Loss or Validation Accuracy. It is recommended to implement a function to log and monitor these metrics throughout training like Weight & Bias.
- There is currently no function to record training time. Adding such a function will help evaluate the efficiency of model changes.

---
