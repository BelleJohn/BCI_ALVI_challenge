# augmentations
import audiomentations as A
from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy as np

def make_electrode_shifting(data, min_angle, max_angle):
    """
    Rotates electrode signals within a specified angular range and interpolates the resulting data.
    
    Parameters:
    - data (numpy.ndarray): Electrode data of shape (n_sensors, n_samples).
    - min_angle (float): Minimum rotation angle in degrees.
    - max_angle (float): Maximum rotation angle in degrees.

    Returns:
    - numpy.ndarray: Interpolated data after rotation, same shape as input.
    """
    
    n_sensors, _ = data.shape
    original_angles = np.linspace(0, 360, num=n_sensors, endpoint=False)
    

    signs = np.random.choice([-1, 1], size=n_sensors)
    random_angles = np.random.uniform(low=min_angle, high=max_angle, size=n_sensors)

    delta_array = signs * random_angles
    new_angles = original_angles + delta_array

    # Vectorized correction to ensure angles are within the 0-360 degree range
    new_angles = (new_angles + 360) % 360

    
    # Vectorized computation of angular distances for all new angles to all original angles
    distances = np.abs(new_angles[:, np.newaxis] - original_angles)
    distances = np.minimum(distances, 360 - distances) / (360 / n_sensors)

    weights = 1 - distances  # Invert distances to get weights
    weights = np.clip(weights, 0, 1)

    rotated_data = np.dot(weights, data)
    
    return rotated_data

class SpatialRotation(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True 

    def __init__(self, min_angle, max_angle, p=0.5):

        super().__init__(p)
        self.min_angle = min_angle
        self.max_angle = max_angle

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

    def apply(self, samples, sample_rate):
        # print('sample_shape_in_SpatialRotation',samples.shape) # (n_sensors, n_samples) (8, 256)
        result = make_electrode_shifting(samples,
                                         self.min_angle,
                                         self.max_angle)
        result = result.astype(samples.dtype)
        return result

class TimeMaskingMultichannel(BaseWaveformTransform):

    supports_multichannel = True 

    def __init__(self, max_mask_time, p=0.5):
        super().__init__(p)
        self.max_mask_time = max_mask_time

    def apply(self, samples, sample_rate):
        num_channels, num_samples = samples.shape
        mask_time = np.random.randint(0, self.max_mask_time)
        start_time = np.random.randint(0, num_samples - mask_time)
        samples[:, start_time:start_time + mask_time] = 0
        return samples

class FrequencyMaskingMultichannel(BaseWaveformTransform):
    supports_multichannel = True 
    def __init__(self, max_mask_freq, p=0.5):
        super().__init__(p)
        self.max_mask_freq = max_mask_freq

    def apply(self, samples, sample_rate):
        num_channels, num_samples = samples.shape
        mask_freq = np.random.randint(0, self.max_mask_freq)
        start_freq = np.random.randint(0, num_channels - mask_freq)
        samples[start_freq:start_freq + mask_freq, :] = 0
        return samples

import pywt

class WaveletNoiseInjection(BaseWaveformTransform):
    supports_multichannel = True 
    """
    Apply wavelet transform to the signal, add noise to the wavelet coefficients,
    and then reconstruct the signal using the inverse wavelet transform.
    """

    def __init__(self, noise_level=0.1, p=0.5):
        super().__init__(p)
        self.noise_level = noise_level

    def apply(self, samples, sample_rate):
        # Apply wavelet transform
        coeffs = pywt.wavedec(samples, 'db1', mode='symmetric')
        # Add Gaussian noise to coefficients
        noisy_coeffs = [coeff + np.random.normal(0, self.noise_level, coeff.shape) for coeff in coeffs]
        # Reconstruct the signal
        augmented_samples = pywt.waverec(noisy_coeffs, 'db1', mode='symmetric')
        # Ensure the shape is consistent
        if augmented_samples.shape != samples.shape:
            augmented_samples = augmented_samples[:samples.shape[0], :samples.shape[1]]
        return augmented_samples.astype(samples.dtype)
    

def get_default_transform(p=0.0):
    if p == 0:
        return None
    
    transform = A.Compose([
        A.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.1, p=p),
        SpatialRotation(min_angle=1, max_angle=10, p=p),
        # TimeMaskingMultichannel(max_mask_time=30, p=p),
        # FrequencyMaskingMultichannel(max_mask_freq=3, p=p),
        # WaveletNoiseInjection(noise_level=0.1, p=p)
    ])
    return transform