"""
Neuroscript Backend REST API Service
FastAPI-powered backend exposing endpoints for EEG data loading, preprocessing,
signal streaming, ML inference, and Mind-to-Text command generation.
"""

from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from config import CLASSIFIER_PASSBAND, DEFAULT_MAPPING, MOTOR_CHANNELS, RUN_INFO, SCIENTIFIC_DISCLAIMER
from communicator import MindToTextSystem
from keyboard import VirtualBCIKeyboard, get_word_suggestions
from data_loader import (
    generate_synthetic_demo_run,
    get_recording_metadata,
    list_available_runs,
    load_real_eeg_run,
)
from features import extract_band_power_features
from model_trainer import EEGClassifierSystem
from preprocessing import create_epochs_from_events, get_single_trial_signal, preprocess_raw

app = FastAPI(
    title="Neuroscript EEG BCI Backend API",
    description="REST API for PhysioNet EEG Motor Imagery Dataset loading, preprocessing, and Mind-to-Text classification.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for models and session communicator
_cached_runs: Dict[int, object] = {}
_cached_events: Dict[int, List[Dict]] = {}
_cached_classifier: Optional[EEGClassifierSystem] = None
_communicator = MindToTextSystem()
_virtual_keyboard = VirtualBCIKeyboard(confidence_threshold=0.60, cooldown_seconds=0.2)


def _get_run(run_id: int):
    """Retrieves or loads a run into cache."""
    if run_id in _cached_runs:
        return _cached_runs[run_id], _cached_events[run_id]

    available = list_available_runs(subject=1)
    if run_id in available:
        raw, events = load_real_eeg_run(available[run_id], run_id=run_id)
        _cached_runs[run_id] = raw
        _cached_events[run_id] = events
        return raw, events
    else:
        # Fallback to synthetic demo
        raw, events = generate_synthetic_demo_run(duration_sec=90.0, random_seed=run_id)
        _cached_runs[run_id] = raw
        _cached_events[run_id] = events
        return raw, events


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check and service status."""
    return {
        "status": "healthy",
        "service": "Neuroscript EEG Backend",
        "api_version": "1.0.0",
        "dataset": "PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)",
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER.strip()
    }


@app.get("/api/recordings")
def get_available_recordings(subject: int = Query(1, description="Participant ID")):
    """Lists all available PhysioNet recordings detected locally."""
    available = list_available_runs(subject=subject)
    runs_info = []
    for r_id in range(1, 15):
        is_local = r_id in available
        r_meta = RUN_INFO.get(r_id, {})
        runs_info.append({
            "run_id": r_id,
            "name": r_meta.get("name", f"Run {r_id}"),
            "type": r_meta.get("type", "Unknown"),
            "is_downloaded": is_local,
            "classes": r_meta.get("classes", [])
        })
    return {"subject": subject, "recordings": runs_info}


@app.get("/api/recording/{run_id}/metadata")
def get_run_metadata(run_id: int):
    """Returns detailed metadata and channel layout for a specific run."""
    try:
        raw, events = _get_run(run_id)
        available = list_available_runs(subject=1)
        path = available.get(run_id, f"S001R{run_id:02d}.edf")
        meta = get_recording_metadata(raw, run_id, file_path=path, labelled_events=events)
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recording/{run_id}/events")
def get_run_events(run_id: int):
    """Returns the list of labelled task segments (Rest, Left Fist, Right Fist, Both Feet) for the recording."""
    try:
        raw, events = _get_run(run_id)
        return {"run_id": run_id, "total_events": len(events), "events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recording/{run_id}/signal")
def get_signal_segment(
    run_id: int,
    channel: str = Query("C3", description="EEG channel name (e.g. C3, Cz, C4)"),
    onset_sec: float = Query(0.0, description="Start time in seconds"),
    duration_sec: float = Query(4.0, description="Duration in seconds"),
    apply_filter: bool = Query(True, description="Apply 8-30 Hz Mu/Beta bandpass filter")
):
    """Returns time-series signal data (time array and amplitude in uV) for a specific channel and window."""
    try:
        raw, _ = _get_run(run_id)
        if apply_filter:
            raw_proc = preprocess_raw(raw, l_freq=8.0, h_freq=30.0, copy=True)
        else:
            raw_proc = raw

        t_rel, amp_uv, sfreq = get_single_trial_signal(raw_proc, onset_sec, duration_sec, channel)
        return {
            "run_id": run_id,
            "channel": channel,
            "sfreq": sfreq,
            "onset_sec": onset_sec,
            "duration_sec": duration_sec,
            "sample_count": len(amp_uv),
            "time_seconds": t_rel.tolist(),
            "amplitude_uv": amp_uv.tolist(),
            "peak_to_peak_uv": float(np.ptp(amp_uv)) if len(amp_uv) > 0 else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    run_id: int = 4
    trial_index: int = 1
    selected_channels: Optional[List[str]] = None


@app.post("/api/predict")
def predict_trial(req: PredictRequest):
    """Classifies an EEG trial using the calibrated Random Forest model and maps to a communication command."""
    global _cached_classifier
    try:
        raw, events = _get_run(req.run_id)
        chs = req.selected_channels or MOTOR_CHANNELS
        
        # Load trained model from disk if not loaded in cache
        if _cached_classifier is None:
            saved_model_path = Path("models/trained_rf_model.joblib")
            if saved_model_path.exists():
                clf = EEGClassifierSystem()
                clf.load_model(str(saved_model_path))
                _cached_classifier = clf
            else:
                raw_filt = preprocess_raw(raw, l_freq=CLASSIFIER_PASSBAND[0], h_freq=CLASSIFIER_PASSBAND[1], copy=True)
                ep = create_epochs_from_events(raw_filt, tmin=0.5, tmax=3.5, selected_channels=chs)
                lbls = ep.events[:, 2]
                mask = np.isin(lbls, [2, 3])
                feat, _ = extract_band_power_features(ep[mask].get_data(), sfreq=ep.info['sfreq'], channel_names=chs)
                y = np.where(lbls[mask] == 2, 0, 1)
                clf = EEGClassifierSystem(n_estimators=100)
                clf.fit_and_evaluate(feat, y, groups=np.zeros(len(y)), split_method="group_kfold", class_names=["Left Fist Imagery", "Right Fist Imagery"])
                _cached_classifier = clf

        # Find target event
        target_idx = max(0, min(req.trial_index - 1, len(events) - 1)) if events else 0
        target_ev = events[target_idx] if events and target_idx < len(events) else {"label": "Unknown", "onset_sec": 0.0, "duration_sec": 4.0}
        
        # Extract continuous signal slice with identical passband
        raw_filt = preprocess_raw(raw, l_freq=CLASSIFIER_PASSBAND[0], h_freq=CLASSIFIER_PASSBAND[1], copy=True)
        raw_picked = raw_filt.copy().pick(chs)
        sfreq = raw_picked.info['sfreq']
        start_samp = int(round((target_ev.get("onset_sec", 0.0) + 0.5) * sfreq))
        stop_samp = start_samp + int(round(3.0 * sfreq))
        stop_samp = min(stop_samp, raw_picked.n_times)
        
        trial_data = raw_picked.get_data()[:, start_samp:stop_samp]
        
        pred_id, pred_label, conf, probs = _cached_classifier.predict_trial_segment(
            trial_data=trial_data,
            sfreq=sfreq,
            channel_names=chs
        )
        
        entry = _communicator.record_prediction(
            predicted_class=pred_label,
            confidence=conf,
            actual_class=target_ev.get("label", "Unknown"),
            source="PhysioNet Real EEG",
            run_id=req.run_id,
            trial_id=req.trial_index
        )
        return {
            "prediction": entry,
            "calibrated_confidence": conf,
            "all_class_probabilities": probs.tolist(),
            "sentence_buffer": _communicator.get_sentence_text()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_classification_history():
    """Returns the full session classification history."""
    df = _communicator.get_history_dataframe()
    return {"count": len(df), "history": df.to_dict(orient="records")}


@app.delete("/api/history")
def clear_history():
    """Clears the session classification history and sentence buffer."""
    _communicator.clear_history()
@app.get("/api/keyboard/state")
def get_keyboard_state():
    """Returns the current Virtual BCI Keyboard state, accumulated text, and word suggestions."""
    return {
        "accumulated_text": _virtual_keyboard.accumulated_text,
        "highlighted_key": _virtual_keyboard.current_key,
        "word_suggestions": get_word_suggestions(_virtual_keyboard.accumulated_text),
        "confidence_threshold": _virtual_keyboard.confidence_threshold,
        "cooldown_seconds": _virtual_keyboard.cooldown_seconds,
        "recent_history": _virtual_keyboard.action_history[-10:] if _virtual_keyboard.action_history else []
    }


class KeyboardActionRequest(BaseModel):
    action: str  # "MOVE_LEFT", "MOVE_RIGHT", "NEXT_ROW", "SELECT", "SPACE", "DELETE", "CLEAR", or key name
    source: str = "Manual API Click"


@app.post("/api/keyboard/action")
def trigger_keyboard_action(req: KeyboardActionRequest):
    """Executes a manual or test action on the Virtual BCI Keyboard."""
    action = req.action.upper()
    if action == "MOVE_LEFT":
        _virtual_keyboard.navigate_left()
    elif action == "MOVE_RIGHT":
        _virtual_keyboard.navigate_right()
    elif action == "NEXT_ROW":
        _virtual_keyboard.navigate_next_row()
    elif action == "SELECT":
        _virtual_keyboard.select_current_key(source=req.source)
    elif action == "DELETE":
        _virtual_keyboard.backspace(source=req.source)
    elif action == "CLEAR":
        _virtual_keyboard.clear(source=req.source)
    elif action == "SPACE":
        _virtual_keyboard.type_letter_direct("SPACE", source=req.source)
    else:
        # Check if single letter or suggestion word
        if len(action) == 1:
            _virtual_keyboard.type_letter_direct(action, source=req.source)
        else:
            _virtual_keyboard.apply_word_suggestion(action, source=req.source)

    return get_keyboard_state()


@app.post("/api/keyboard/dispatch_eeg")
def dispatch_eeg_to_keyboard(req: PredictRequest):
    """Classifies an EEG trial and dispatches the action to the Virtual BCI Keyboard."""
    pred_res = predict_trial(req)
    pred_entry = pred_res["prediction"]
    
    seg_id = f"R{req.run_id}_T{req.trial_index}_{pred_entry['actual_label']}"
    kb_entry = _virtual_keyboard.process_eeg_prediction(
        predicted_class=pred_entry["predicted_task"],
        confidence=pred_res["calibrated_confidence"],
        segment_id=seg_id,
        actual_label=pred_entry["actual_label"],
        is_synthetic=False
    )
    
    return {
        "eeg_prediction": pred_entry,
        "keyboard_result": kb_entry,
        "keyboard_state": get_keyboard_state()
    }


@app.delete("/api/keyboard")
def clear_keyboard():
    """Clears accumulated text and audit log on the Virtual BCI Keyboard."""
    _virtual_keyboard.clear(source="API Reset")
    _virtual_keyboard.action_history.clear()
    return {"message": "Virtual Keyboard reset successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
