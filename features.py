"""
EEG Feature Extraction Module.
Computes multi-band Welch Power Spectral Density (PSD):
- Delta band power (0.5 - 4 Hz)
- Theta band power (4 - 8 Hz)
- Alpha / Mu band power (8 - 12 Hz)
- Beta band power (13 - 30 Hz)
And signal statistics (mean amplitude, standard deviation, variance, and power ratios).
"""

from typing import Dict, List, Optional, Tuple, Union
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from config import ALPHA_BAND, BETA_BAND, DELTA_BAND, THETA_BAND


def extract_band_power_features(
    epochs_data: np.ndarray,
    sfreq: float,
    channel_names: List[str],
    delta_range: Tuple[float, float] = DELTA_BAND,
    theta_range: Tuple[float, float] = THETA_BAND,
    alpha_range: Tuple[float, float] = ALPHA_BAND,
    beta_range: Tuple[float, float] = BETA_BAND,
    include_statistics: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts log band power features and signal statistics from epoched EEG data.
    Input shape: (n_epochs, n_channels, n_times)
    
    Returns:
        X: numpy array of shape (n_epochs, n_features)
        feature_names: human-readable labels for each feature column.
    """
    if isinstance(epochs_data, tuple):
        epochs_data = epochs_data[0]
    epochs_data = np.asarray(epochs_data)

    if epochs_data.ndim == 2:
        # Single epoch of shape (n_channels, n_times) -> expand to (1, n_channels, n_times)
        epochs_data = np.expand_dims(epochs_data, axis=0)

    n_epochs, n_channels, n_times = epochs_data.shape
    
    # Compute PSD using Welch's method with safe segment length
    nperseg = min(128, n_times)
    freqs, psd = welch(epochs_data, fs=sfreq, nperseg=nperseg, axis=-1)
    
    # Frequency masks
    delta_mask = (freqs >= delta_range[0]) & (freqs <= delta_range[1])
    theta_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    beta_mask = (freqs >= beta_range[0]) & (freqs <= beta_range[1])
    
    # Numerical stability epsilon
    eps = 1e-12
    
    # Mean PSD per frequency band (log10 transformed)
    delta_power = np.log10(np.mean(psd[:, :, delta_mask], axis=-1) + eps) # (n_epochs, n_channels)
    theta_power = np.log10(np.mean(psd[:, :, theta_mask], axis=-1) + eps)
    alpha_power = np.log10(np.mean(psd[:, :, alpha_mask], axis=-1) + eps)
    beta_power = np.log10(np.mean(psd[:, :, beta_mask], axis=-1) + eps)
    
    # Band power ratios
    alpha_beta_ratio = alpha_power / (beta_power + eps)
    theta_beta_ratio = theta_power / (beta_power + eps)
    
    feature_blocks = [
        delta_power,
        theta_power,
        alpha_power,
        beta_power,
        alpha_beta_ratio,
        theta_beta_ratio
    ]
    
    feature_names = []
    for ch in channel_names:
        feature_names.append(f"{ch}_Delta_Power (0.5-4Hz)")
    for ch in channel_names:
        feature_names.append(f"{ch}_Theta_Power (4-8Hz)")
    for ch in channel_names:
        feature_names.append(f"{ch}_Alpha_Power (8-12Hz)")
    for ch in channel_names:
        feature_names.append(f"{ch}_Beta_Power (13-30Hz)")
    for ch in channel_names:
        feature_names.append(f"{ch}_Alpha_Beta_Ratio")
    for ch in channel_names:
        feature_names.append(f"{ch}_Theta_Beta_Ratio")
        
    if include_statistics:
        # Time-domain signal statistics
        mean_val = np.mean(epochs_data, axis=-1) # (n_epochs, n_channels)
        std_val = np.std(epochs_data, axis=-1)
        var_val = np.log10(np.var(epochs_data, axis=-1) + eps)
        
        feature_blocks.extend([mean_val, std_val, var_val])
        for ch in channel_names:
            feature_names.append(f"{ch}_Mean")
        for ch in channel_names:
            feature_names.append(f"{ch}_StdDev")
        for ch in channel_names:
            feature_names.append(f"{ch}_LogVar")

    X = np.concatenate(feature_blocks, axis=1)
    return X, feature_names


def extract_single_trial_features_table(
    single_epoch_data: np.ndarray,
    sfreq: float,
    channel_names: List[str]
) -> pd.DataFrame:
    """
    Extracts features for a single trial and returns a structured DataFrame
    with rows for each channel and columns for Delta, Theta, Alpha, Beta, Mean, and StdDev.
    Input shape: (n_channels, n_times)
    """
    if single_epoch_data.ndim == 3:
        single_epoch_data = single_epoch_data[0]
        
    n_channels, n_times = single_epoch_data.shape
    nperseg = min(128, n_times)
    freqs, psd = welch(single_epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
    
    eps = 1e-12
    delta_mask = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    theta_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])

    rows = []
    for i, ch in enumerate(channel_names):
        d_p = float(np.log10(np.mean(psd[i, delta_mask]) + eps))
        t_p = float(np.log10(np.mean(psd[i, theta_mask]) + eps))
        a_p = float(np.log10(np.mean(psd[i, alpha_mask]) + eps))
        b_p = float(np.log10(np.mean(psd[i, beta_mask]) + eps))
        mean_v = float(np.mean(single_epoch_data[i]) * 1e6) # uV
        std_v = float(np.std(single_epoch_data[i]) * 1e6) # uV
        
        rows.append({
            "Channel": ch,
            "Delta (0.5–4Hz)": round(d_p, 3),
            "Theta (4–8Hz)": round(t_p, 3),
            "Alpha/Mu (8–12Hz)": round(a_p, 3),
            "Beta (13–30Hz)": round(b_p, 3),
            "Mean (uV)": round(mean_v, 2),
            "StdDev (uV)": round(std_v, 2)
        })

    return pd.DataFrame(rows)


def compute_spectral_density_curve(
    raw_or_epoch_data: np.ndarray,
    sfreq: float,
    fmin: float = 0.5,
    fmax: float = 40.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D Power Spectral Density curve for a given channel signal.
    Returns: (frequencies, psd_db)
    """
    freqs, psd = welch(raw_or_epoch_data, fs=sfreq, nperseg=min(256, len(raw_or_epoch_data)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    psd_db = 10 * np.log10(psd[mask] + 1e-14)
    return freqs[mask], psd_db
