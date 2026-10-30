"""
Data Loader Module for PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).
Supports real EDF+ recordings, annotation extraction, detailed metadata reporting,
manual download instructions, and clearly-separated synthetic demonstration mode.
"""

from collections import Counter
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from config import MOTOR_CHANNELS, RUN_INFO, SFREQ_DEFAULT


def get_default_mne_path(subject: int = 1) -> Path:
    """Returns the default directory where MNE downloads PhysioNet EEGBCI files."""
    home = Path.home()
    return home / "mne_data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}"


def list_available_runs(subject: int = 1, custom_dir: Optional[str] = None) -> Dict[int, Path]:
    """
    Scans local storage for downloaded PhysioNet runs for a given subject.
    Returns a dict mapping run_id -> Path.
    """
    search_dir = Path(custom_dir) if custom_dir else get_default_mne_path(subject)
    available = {}
    if not search_dir.exists():
        return available

    for run_id in range(1, 15):
        filename = f"S{subject:03d}R{run_id:02d}.edf"
        file_path = search_dir / filename
        if file_path.exists():
            available[run_id] = file_path
    return available


def download_physionet_data(subject: int = 1, runs: Optional[List[int]] = None, verbose: bool = False) -> List[Path]:
    """
    Downloads the requested runs from PhysioNet via MNE EEGBCI module.
    Returns list of local file paths.
    """
    if runs is None:
        # Default to Motor Imagery runs (4, 6, 8, 10, 12, 14) + Baseline (1)
        runs = [1, 4, 6, 8, 10, 12, 14]
    
    try:
        paths = mne.datasets.eegbci.load_data(subject, runs, update_path=True, verbose=verbose)
        return [Path(p) for p in paths]
    except Exception as e:
        raise RuntimeError(f"Failed to download PhysioNet EEG data: {e}")


def get_manual_download_instructions(subject: int = 1) -> str:
    """Returns step-by-step instructions for manually downloading recordings if automated download fails."""
    target_dir = get_default_mne_path(subject)
    instructions = f"""
### Manual Download Instructions for PhysioNet EEGMMIDB
If automated downloading via MNE is blocked or offline:
1. Open your browser and navigate to the official PhysioNet page:
   https://physionet.org/content/eegmmidb/1.0.0/S{subject:03d}/
2. Download the desired EDF files (e.g. `S{subject:03d}R01.edf`, `S{subject:03d}R04.edf`, `S{subject:03d}R08.edf`, `S{subject:03d}R12.edf`).
3. Place the downloaded `.edf` files into this directory on your machine:
   `{target_dir}`
4. Alternatively, place them in a `data/` folder inside the project directory.
5. Refresh the application; the recordings will be detected automatically.
"""
    return instructions


def load_real_eeg_run(file_path: Union[str, Path], run_id: int) -> Tuple[mne.io.Raw, List[Dict]]:
    """
    Loads an EDF/EDF+ recording, standardizes channel names, sets 10-05 montage,
    and returns the Raw object along with a list of extracted labelled event segments.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Recording file not found: {file_path}")

    raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    
    # Try setting montage
    try:
        montage = mne.channels.make_standard_montage('colin27_1005')
        raw.set_montage(montage, on_missing='ignore')
    except Exception:
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
        except Exception:
            pass

    # Extract annotations and map codes to actual physiological task descriptions
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    run_meta = RUN_INFO.get(run_id, {"type": "imagery", "classes": ["Rest", "Task 1", "Task 2"]})
    
    # Map T0, T1, T2 according to dataset specification
    label_map = {}
    if run_id in [3, 4, 7, 8, 11, 12]:
        label_map = {
            "T0": "Rest",
            "T1": "Left Fist (Exec)" if "execution" in run_meta["type"] else "Left Fist Imagery",
            "T2": "Right Fist (Exec)" if "execution" in run_meta["type"] else "Right Fist Imagery"
        }
    elif run_id in [5, 6, 9, 10, 13, 14]:
        label_map = {
            "T0": "Rest",
            "T1": "Both Fists (Exec)" if "execution" in run_meta["type"] else "Both Fists Imagery",
            "T2": "Both Feet (Exec)" if "execution" in run_meta["type"] else "Both Feet Imagery"
        }
    else: # Baseline runs 1 and 2
        label_map = {"T0": "Rest"}

    labelled_events = []
    annots = raw.annotations
    if annots and len(annots) > 0:
        for i in range(len(annots)):
            desc = annots.description[i]
            onset = float(annots.onset[i])
            duration = float(annots.duration[i])
            human_label = label_map.get(desc, desc)
            labelled_events.append({
                "index": i + 1,
                "onset_sec": round(onset, 2),
                "duration_sec": round(duration, 2),
                "code": desc,
                "label": human_label,
                "is_synthetic": False
            })
    else:
        # If baseline with no event annotations, create discrete Rest epochs
        duration = raw.times[-1]
        for start in np.arange(0, duration - 4.0, 4.0):
            labelled_events.append({
                "index": len(labelled_events) + 1,
                "onset_sec": round(float(start), 2),
                "duration_sec": 4.0,
                "code": "T0",
                "label": "Rest",
                "is_synthetic": False
            })

    return raw, labelled_events


def get_recording_metadata(
    raw: mne.io.Raw,
    run_id: int,
    file_path: Optional[Union[str, Path]] = None,
    labelled_events: Optional[List[Dict]] = None
) -> Dict:
    """
    Extracts comprehensive metadata about the recording file, subject, run type,
    sampling rate, channels, and event distribution.
    """
    run_meta = RUN_INFO.get(run_id, {"name": f"Run {run_id}", "type": "Unknown"})
    fname = Path(file_path).name if file_path else f"Run_{run_id}.edf"
    
    event_counts = {}
    if labelled_events:
        counts = Counter([ev["label"] for ev in labelled_events])
        event_counts = dict(counts)

    return {
        "filename": fname,
        "participant": fname.split("R")[0] if "R" in fname else "S001",
        "run_id": run_id,
        "run_name": run_meta.get("name", f"Run {run_id}"),
        "run_type": run_meta.get("type", "Unknown").capitalize(),
        "sampling_frequency_hz": raw.info['sfreq'],
        "duration_seconds": round(float(raw.times[-1]), 2),
        "total_channels": raw.info['nchan'],
        "channels_list": list(raw.ch_names),
        "total_events": len(labelled_events) if labelled_events else 0,
        "event_distribution": event_counts
    }


def generate_synthetic_demo_run(
    duration_sec: float = 120.0,
    sfreq: float = SFREQ_DEFAULT,
    random_seed: int = 42
) -> Tuple[mne.io.Raw, List[Dict]]:
    """
    Generates a realistic synthetic EEG recording with 64 channels, pink noise background,
    and simulated sensorimotor Mu/Beta rhythm event-related desynchronization (ERD)
    for Left Fist, Right Fist, Both Feet, and Rest.
    
    EVERY returned object and label is explicitly marked as SYNTHETIC DEMO.
    """
    np.random.seed(random_seed)
    n_samples = int(duration_sec * sfreq)
    times = np.arange(n_samples) / sfreq
    
    # 64 standard channel names
    channels = [
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
    ][:64]

    # Generate baseline 1/f pink noise (scaled to microvolts range ~10-50 uV)
    white = np.random.randn(len(channels), n_samples)
    eeg_data = mne.filter.filter_data(white, sfreq, l_freq=1.0, h_freq=45.0, verbose=False) * 15e-6

    # Create trial schedule: 4.0s trials alternating with 4.0s rest
    trial_tasks = ["Left Fist Imagery", "Right Fist Imagery", "Both Feet Imagery", "Both Fists Imagery"]
    labelled_events = []
    
    current_time = 2.0
    trial_idx = 1
    
    onset_list = []
    duration_list = []
    description_list = []

    c3_idx = channels.index('C3') if 'C3' in channels else 0
    c4_idx = channels.index('C4') if 'C4' in channels else 1
    cz_idx = channels.index('Cz') if 'Cz' in channels else 2

    # Add baseline Mu rhythm (~10 Hz) and Beta rhythm (~20 Hz)
    base_mu = np.sin(2 * np.pi * 10.0 * times) * 20e-6
    base_beta = np.sin(2 * np.pi * 20.0 * times) * 8e-6
    for ch_i in range(len(channels)):
        eeg_data[ch_i] += base_mu * np.random.uniform(0.6, 1.0)
        eeg_data[ch_i] += base_beta * np.random.uniform(0.5, 0.9)

    while current_time + 4.0 < duration_sec:
        task = trial_tasks[(trial_idx - 1) % len(trial_tasks)]
        t_mask = (times >= current_time) & (times < current_time + 4.0)

        # Simulate ERD (power decrease) during imagery:
        if task == "Left Fist Imagery":
            eeg_data[c4_idx, t_mask] *= 0.35
            eeg_data[c3_idx, t_mask] *= 1.15
            code = "T1"
        elif task == "Right Fist Imagery":
            eeg_data[c3_idx, t_mask] *= 0.35
            eeg_data[c4_idx, t_mask] *= 1.15
            code = "T2"
        elif task == "Both Feet Imagery":
            eeg_data[cz_idx, t_mask] *= 0.30
            code = "T2"
        elif task == "Both Fists Imagery":
            eeg_data[c3_idx, t_mask] *= 0.40
            eeg_data[c4_idx, t_mask] *= 0.40
            code = "T1"

        onset_list.append(current_time)
        duration_list.append(4.0)
        description_list.append(code)

        labelled_events.append({
            "index": trial_idx,
            "onset_sec": round(current_time, 2),
            "duration_sec": 4.0,
            "code": code,
            "label": f"[SYNTHETIC DEMO] {task}",
            "is_synthetic": True
        })

        # Add intervening rest segment
        current_time += 4.0
        trial_idx += 1
        if current_time + 4.0 < duration_sec:
            onset_list.append(current_time)
            duration_list.append(4.0)
            description_list.append("T0")
            labelled_events.append({
                "index": trial_idx,
                "onset_sec": round(current_time, 2),
                "duration_sec": 4.0,
                "code": "T0",
                "label": "[SYNTHETIC DEMO] Rest",
                "is_synthetic": True
            })
            current_time += 4.0
            trial_idx += 1

    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    annotations = mne.Annotations(onset=onset_list, duration=duration_list, description=description_list)
    raw.set_annotations(annotations)
    
    try:
        montage = mne.channels.make_standard_montage('colin27_1005')
        raw.set_montage(montage, on_missing='ignore')
    except Exception:
        pass

    return raw, labelled_events
