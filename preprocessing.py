"""
EEG Signal Preprocessing Module
Implements reproducible zero-phase filtering (bandpass and notch), channel validation,
artifact / invalid segment rejection, and multi-run dataset preparation.
"""

from typing import Dict, List, Optional, Tuple, Union
import mne
import numpy as np
from scipy.signal import butter, filtfilt

from config import MOTOR_CHANNELS, PASSBAND, SFREQ_DEFAULT
from data_loader import load_real_eeg_run


def validate_recording_and_channels(
    raw: mne.io.Raw,
    requested_channels: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Validates channels against raw recording.
    Returns: (valid_channels, missing_channels)
    """
    if raw is None or not hasattr(raw, 'ch_names'):
        raise ValueError("Invalid EEG recording object provided.")

    existing_chs = set(raw.ch_names)
    valid_chs = [ch for ch in requested_channels if ch in existing_chs]
    missing_chs = [ch for ch in requested_channels if ch not in existing_chs]

    return valid_chs, missing_chs


def preprocess_raw(
    raw: mne.io.Raw,
    l_freq: float = PASSBAND[0],
    h_freq: float = PASSBAND[1],
    notch_freq: Optional[float] = 60.0,
    copy: bool = True
) -> mne.io.Raw:
    """
    Applies reproducible zero-phase FIR bandpass filtering (isolating Mu and Beta sensorimotor bands)
    and optional notch filtering.
    """
    if raw is None:
        raise ValueError("Cannot preprocess a None raw recording.")

    raw_proc = raw.copy() if copy else raw
    
    # Apply zero-phase FIR bandpass filter
    raw_proc.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design='firwin',
        skip_by_annotation='edge',
        verbose=False
    )
    
    # Apply optional notch filter if powerline noise is present and within Nyquist limit
    if notch_freq is not None and notch_freq < (raw_proc.info['sfreq'] / 2.0):
        try:
            raw_proc.notch_filter(
                freqs=notch_freq,
                fir_design='firwin',
                verbose=False
            )
        except Exception:
            pass

    return raw_proc


def scipy_bandpass_filter(
    signal_1d: np.ndarray,
    sfreq: float = SFREQ_DEFAULT,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    order: int = 4
) -> np.ndarray:
    """
    Reproducible zero-phase Butterworth bandpass filter using SciPy filtfilt.
    Useful for standalone single-channel filtering outside MNE.
    """
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal_1d)


def create_epochs_from_events(
    raw: mne.io.Raw,
    tmin: float = 0.5,
    tmax: float = 3.5,
    selected_channels: Optional[List[str]] = None,
    baseline: Optional[Tuple[float, float]] = None
) -> mne.Epochs:
    """
    Extracts fixed-interval epochs around event annotations.
    Standard motor imagery analysis typically uses [0.5s, 3.5s] relative to cue onset
    to ignore visual evoked potentials and reaction delays.
    """
    if raw is None:
        raise ValueError("Raw object cannot be None.")

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if len(events) == 0:
        raise ValueError("No events or annotations found in this recording.")

    raw_picked = raw.copy()
    if selected_channels is not None:
        valid_chs, _ = validate_recording_and_channels(raw_picked, selected_channels)
        if valid_chs:
            raw_picked.pick(valid_chs)
        else:
            raise ValueError(f"None of the requested channels {selected_channels} exist in recording.")

    epochs = mne.Epochs(
        raw_picked,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False
    )
    return epochs


def get_single_trial_signal(
    raw: mne.io.Raw,
    onset_sec: float,
    duration_sec: float,
    channel: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts a 1D time segment for a specific channel and event.
    Returns: (time_array, amplitude_microvolts, sampling_frequency)
    """
    if raw is None:
        raise ValueError("Raw object cannot be None.")

    sfreq = raw.info['sfreq']
    start_sample = max(0, int(onset_sec * sfreq))
    stop_sample = min(raw.n_times, int((onset_sec + duration_sec) * sfreq))

    if channel not in raw.ch_names:
        # Fallback to the first available channel gracefully
        channel = raw.ch_names[0]

    ch_idx = raw.ch_names.index(channel)
    data, times = raw[ch_idx, start_sample:stop_sample]
    
    # Convert from Volts (MNE default) to microvolts (uV)
    amplitude_uv = data[0] * 1e6
    time_rel = times - times[0] if len(times) > 0 else np.array([])
    
    return time_rel, amplitude_uv, sfreq


def prepare_labelled_dataset_segments(
    recordings_info: List[Dict],
    task_mode: str = "2-class",
    selected_channels: Optional[List[str]] = None,
    tmin: float = 0.5,
    tmax: float = 3.5,
    reject_flatline_uv: float = 0.1
) -> Dict:
    """
    Loads recordings, applies correct run-specific event interpretation,
    extracts fixed-duration segments, excludes invalid/incomplete trials,
    and returns arrays ready for feature extraction and machine learning.
    
    Args:
        recordings_info: list of dicts, e.g. [{"path": Path, "run_id": 4, "subject": "S001"}, ...]
        task_mode: "2-class" (Left vs Right), "3-class" (Rest vs Left vs Right),
                   or "4-class" (Rest vs Left vs Right vs Feet)
    """
    channels = selected_channels or MOTOR_CHANNELS
    
    valid_data_list = []
    valid_labels_list = []
    valid_groups_list = []
    valid_subjects_list = []
    
    total_segments_checked = 0
    excluded_segments_count = 0
    
    if task_mode == "2-class":
        class_names = ["Left Fist Imagery", "Right Fist Imagery"]
    elif task_mode == "3-class":
        class_names = ["Rest", "Left Fist Imagery", "Right Fist Imagery"]
    else:
        class_names = ["Rest", "Left Fist Imagery", "Right Fist Imagery", "Both Feet Imagery"]

    for rec in recordings_info:
        r_path = rec["path"]
        r_id = rec["run_id"]
        sub_id = rec.get("subject", "S001")
        
        raw, labelled_events = load_real_eeg_run(r_path, run_id=r_id)
        raw_filt = preprocess_raw(raw, l_freq=1.0, h_freq=35.0, copy=True)
        
        # Pick channels safely
        valid_chs, _ = validate_recording_and_channels(raw_filt, channels)
        raw_picked = raw_filt.pick(valid_chs)
        sfreq = raw_picked.info['sfreq']
        expected_samples = int((tmax - tmin) * sfreq)

        for ev in labelled_events:
            total_segments_checked += 1
            onset = ev["onset_sec"]
            code = ev["code"] # "T0", "T1", "T2"
            
            # Map event code to class based on run number and task_mode
            target_class_id = None
            if task_mode == "2-class":
                if r_id in [3, 4, 7, 8, 11, 12]:
                    if code == "T1":
                        target_class_id = 0 # Left
                    elif code == "T2":
                        target_class_id = 1 # Right
            elif task_mode == "3-class":
                if r_id in [3, 4, 7, 8, 11, 12]:
                    if code == "T0":
                        target_class_id = 0 # Rest
                    elif code == "T1":
                        target_class_id = 1 # Left
                    elif code == "T2":
                        target_class_id = 2 # Right
            else: # 4-class
                if r_id in [3, 4, 7, 8, 11, 12]:
                    if code == "T0":
                        target_class_id = 0 # Rest
                    elif code == "T1":
                        target_class_id = 1 # Left
                    elif code == "T2":
                        target_class_id = 2 # Right
                elif r_id in [5, 6, 9, 10, 13, 14]:
                    if code == "T0":
                        target_class_id = 0 # Rest
                    elif code == "T2":
                        target_class_id = 3 # Both Feet

            if target_class_id is None:
                # Segment not part of the active task subset
                continue

            # Slice segment
            start_samp = int((onset + tmin) * sfreq)
            stop_samp = start_samp + expected_samples
            
            # Boundary check: exclude incomplete segments
            if stop_samp > raw_picked.n_times or start_samp < 0:
                excluded_segments_count += 1
                continue

            seg_data, _ = raw_picked[:, start_samp:stop_samp]
            
            # Artifact check: exclude flatline or NaN segments
            if np.isnan(seg_data).any() or np.ptp(seg_data) * 1e6 < reject_flatline_uv:
                excluded_segments_count += 1
                continue

            valid_data_list.append(seg_data)
            valid_labels_list.append(target_class_id)
            valid_groups_list.append(r_id)
            valid_subjects_list.append(sub_id)

    if len(valid_data_list) == 0:
        raise ValueError("No valid EEG segments could be extracted with the given configuration.")

    X_epochs = np.array(valid_data_list) # (n_epochs, n_channels, n_times)
    y_labels = np.array(valid_labels_list, dtype=int)
    groups = np.array(valid_groups_list)
    subjects = np.array(valid_subjects_list)

    # Calculate class counts
    class_counts = {}
    for idx, c_name in enumerate(class_names):
        count = int(np.sum(y_labels == idx))
        class_counts[c_name] = count

    return {
        "X_epochs": X_epochs,
        "y": y_labels,
        "groups": groups,
        "subjects": subjects,
        "channels": valid_chs,
        "class_names": class_names,
        "class_counts": class_counts,
        "total_segments_checked": total_segments_checked,
        "excluded_segments_count": excluded_segments_count,
        "sfreq": sfreq
    }
