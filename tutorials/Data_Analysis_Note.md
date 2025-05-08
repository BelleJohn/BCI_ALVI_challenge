
The code demo is in the 01_load_dataset.ipynb.

There are two ways to start explore data: one is to find a suitable method from the same domain of literature, and the other is to use traditional and simple methods. In the file, traditional methods are used, like mean, standard deviation, and so on.

---

## Mean in EMG Signal Analysis can show:

1. **Baseline Detection**
   - Helps identify the resting electrical activity level of muscles
   - Allows you to distinguish between active and inactive states
   - Provides a reference point for signal normalization

2. **Signal Quality Assessment**
   - Unusual mean values might indicate poor electrode contact
   - Unexpected offsets could suggest instrumentation bias or electrical noise
   - Consistent means across electrodes often indicate proper recording setup

3. **Electrode Positioning Insights**
   - Different mean values across electrodes can reveal which sensors are placed over more active muscle groups
   - Helps understand the spatial distribution of muscle activity

4. **Subject Comparison**
   - Comparing means between healthy and amputant subjects reveals fundamental differences in muscle activation patterns
   - Can highlight compensatory strategies used by amputant subjects

5. **Feature Engineering**
   - The mean serves as a simple yet informative feature for machine learning models
   - Often combined with other statistics to create robust EMG feature sets

6. **Signal Preprocessing**
   - Removing the mean (centering) is a common preprocessing step before further analysis
   - Helps eliminate DC offset before frequency analysis

In this challenge of mapping EMG signals to hand movements, the mean values provide insights into overall muscle engagement patterns that contribute to different hand gestures and positions.

---

## Standard Deviation in EMG Signal Analysis can show:

1. **Activity Intensity Measurement**
   - Higher standard deviation indicates more vigorous muscle contractions
   - Low standard deviation suggests minimal muscle activity or resting state
   - Helps distinguish between active movement and static holding positions

2. **Signal Quality Indicator**
   - Unusually low standard deviation might indicate poor electrode contact
   - Abnormally high values could suggest electrical noise or movement artifacts
   - Helps identify potential recording problems

3. **Movement Pattern Characterization**
   - Different hand gestures create distinctive variability patterns
   - Precise movements show different standard deviation profiles than gross movements
   - Can help distinguish between different types of hand activities

4. **Muscle Fatigue Detection**
   - Progressive changes in standard deviation over time can indicate fatigue
   - Decreasing variability often accompanies muscle exhaustion
   - Important for understanding signal evolution during extended use

5. **Feature Importance**
   - Standard deviation is a powerful statistical feature for machine learning models
   - Often more informative than mean values for pattern recognition tasks
   - Crucial for creating robust EMG-based control systems

6. **Subject Differences**
   - Comparing standard deviations between healthy and amputant subjects reveals differences in motor control strategies
   - Individual variations in standard deviation patterns may require personalized model calibration

---

## Maximum and Minimum in EMG Signal Analysis can show:

### Maximum Values Reveal:

1. **Peak Muscle Activity**
   - Identifies the highest level of muscle recruitment during movements
   - Helps understand the maximum force generation capability
   - Shows which gestures require the most intense muscle activation

2. **Electrode Positioning Effectiveness**
   - Large differences in maximum values between electrodes can indicate optimal/suboptimal placement
   - Higher maximums typically indicate electrodes positioned directly over active muscle groups

3. **Subject-Specific Capabilities**
   - Differences in maximum values between subjects (healthy vs. amputant) reveal strength disparities
   - Lower maximum values in amputant subjects might indicate muscle atrophy
   - Higher values might suggest compensatory overactivation

4. **Signal Quality**
   - Unusually high maximum values could indicate artifacts or noise
   - Consistent maximum ranges across recording sessions suggest reliable data collection

### Minimum Values Provide:

1. **Baseline Information**
   - Establishes the resting electrical activity of muscles
   - Helps identify the noise floor of your recording system
   - Reveals potential DC offset that might need correction

2. **Muscle Relaxation Assessment**
   - Higher minimum values suggest inability to fully relax muscles
   - In amputant subjects, elevated minimums may indicate phantom limb tension

3. **Signal Preprocessing Requirements**
   - Large offsets from zero indicate the need for baseline correction
   - Helps determine appropriate filtering strategies

### The Range (Max-Min) Indicates:

1. **Dynamic Range of Movement**
   - Wider ranges suggest more varied muscle recruitment patterns
   - Narrower ranges might indicate limited motor control or compensatory strategies

2. **Signal Normalization Strategy**
   - Helps establish appropriate normalization parameters for comparing across subjects
   - Essential for creating standardized features for machine learning models

Note: In experiments, I usually superimpose the signals obtained from the same subject, the same action, and the same electrode and use plots to show their maximum and minimum values. Sometimes abnormal signals are discovered and the data is removed. 

---

## EMG Electrode Statistics Boxplots in EMG Signal Analysis can show:

### Distribution of Mean EMG Values (left plot):

1. **Consistent Baseline Activity**: The mean values across electrodes show relatively similar median levels, indicating a consistent baseline of muscle electrical activity detected by most electrodes.

2. **Electrode Uniformity**: The similar box sizes suggest that the baseline activity captured by each electrode is relatively consistent across different samples/recordings.

3. **Outlier Patterns**: Several electrodes show outliers (dots above/below whiskers), suggesting occasional recordings with unusually high/low activity levels.

4. **Stability Across Samples**: The relatively tight distributions indicate that each electrode measures consistent mean activity levels across different hand movement samples.

### Distribution of Standard Deviations (right plot):

1. **Variable Signal Dynamics**: Unlike the means, standard deviations vary considerably between electrodes, with electrodes 1, 2, 7, and 8 showing higher variability.

2. **Electrode Sensitivity Differences**: Electrode 5 consistently shows the lowest standard deviation, suggesting it captures less dynamic muscle activity or is positioned over less active muscle groups.

3. **Signal Quality Indicators**: The wider spread in some electrodes may indicate their placement over more dynamic muscle groups that contribute significantly to hand movements.

4. **Processing Implications**: Electrodes with higher standard deviations likely contain more information about changing muscle states, making them potentially more valuable for predicting hand movements.

---

## Multiple Electrodes Plots in EMG Signal Analysis can show:

The image shows 8 EMG (electromyography) signal plots from different electrodes placed around the arm, representing the electrical activity of muscles during movement. Several key observations:

### Signal Characteristics
- **Spike Patterns**: All electrodes show distinct spike patterns representing muscle activation events
- **Varying Amplitudes**: Different electrodes capture muscle activity with varying intensities
- **Burst Patterns**: Brief periods of high activity followed by lower activity, indicating dynamic muscle contractions

### Electrode Differences
- **Spatial Variation**: Each electrode captures a unique pattern based on its position relative to different muscle groups
- **Signal Quality**: Some electrodes show clearer spike patterns than others, suggesting varied contact quality or proximity to active muscles
- **Different Muscle Groups**: The variations between plots likely represent different muscle recruitment patterns

### Relevance to this Project
These EMG signals form the input data for the hand movement prediction model. The patterns in these signals contain the information the model needs to decode:

1. Which muscles are activating
2. How strongly they're contracting
3. The timing sequences of muscle activation

---

## Joint Angle Visualizations in EMG Signal Analysis can show:

### Joint Angle Time Series

1. **Temporal Coordination**: The time series show synchronized patterns across multiple joints, revealing how joints work together to produce coherent hand gestures.

2. **Different Response Characteristics**: Some joints show rapid transitions (sharp peaks) while others display more gradual changes, indicating varying roles in different movements.

3. **Movement Phases**: Several plots show distinct phases of movement - initialization, peak movement, and return to baseline - which could correspond to complete gesture cycles.

4. **Coupled Joint Relationships**: Similar curve shapes appearing in multiple plots (especially in middle and bottom rows) suggest mechanical coupling between joints.

5. **Movement Amplitude Variations**: The vertical range in each plot indicates the functional range of motion for each joint, with some joints showing much greater angular displacement than others.

### Joint Angle Distributions (Histograms)

1. **Distinct Movement Patterns**: Each joint shows unique distribution patterns, indicating specialized functional roles in hand movements.

2. **Multimodal Distributions**: Many joints show multiple peaks in their histograms, suggesting they tend to rest at specific preferred angles rather than moving continuously through their range.

3. **Range Constraints**: Some joints (particularly in the bottom row) show sharp cutoffs in their distributions, indicating hard biomechanical constraints on their movement range.

4. **Varying Activity Levels**: Joints in the top row show broader distributions, suggesting they participate in more diverse movements, while some in the bottom row show more concentrated distributions, indicating more specialized use.

5. **Correlation Potential**: Similar distribution patterns between certain joints might indicate coordinated movement groups that tend to move together.

---

## PCA Visualization in EMG Signal Analysis can show:
(This is suggested by Claude, the AI tool.)
I usually use PCA for dimensionality reduction before feeding the data into a model for training, rather than for visualization or gaining insights. However, I think it's worthwhile to include this method in my notes—perhaps my future self will discover new insights from it.

### Distribution Patterns

1. **Continuous Data Structure**: The points form a single, continuous cloud rather than discrete clusters, indicating that EMG patterns transition smoothly between different movement states rather than having abrupt categorical differences.

2. **Central Tendency with Outliers**: There's a dense concentration of points in the center representing "typical" muscle activation patterns, with more scattered points at the periphery showing less common or more extreme activation patterns.

3. **Directional Spread**: The cloud shows an elongated shape along the first principal component (x-axis), suggesting that the primary source of variation in the EMG signals occurs along this dimension.

### Technical Implications

1. **Dimensionality Reduction Success**: The visualization demonstrates that the 8-dimensional EMG data (from 8 electrodes) can be effectively projected onto a lower-dimensional space while preserving meaningful structure.

2. **Feature Engineering Potential**: The spread pattern suggests that using these principal components as features for a machine learning model could be effective, as they capture significant variance in the data.

3. **No Distinct Movement Classes**: The lack of clearly separated clusters suggests that predicting discrete hand gesture classes might be challenging - the hand movements represented in the EMG signals appear to exist on a continuum rather than as discrete states.

4. **Data Consistency**: The relatively uniform density gradient from center to periphery suggests good quality data collection without major artifacts or recording issues.

This PCA visualization provides a foundation for understanding the EMG signal space before developing models to map it to hand movement predictions.
