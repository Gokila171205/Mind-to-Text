"""
Automated Pipeline Verification Test.
Tests dataset loading, preprocessing, feature extraction, leakage-free training,
evaluation metrics, communication mapping, and synthetic demo fallback.
"""

import sys
from pathlib import Path

import numpy as np

from config import DEFAULT_MAPPING, MOTOR_CHANNELS
from data_loader import (
    generate_synthetic_demo_run,
    list_available_runs,
    load_real_eeg_run,
)
from features import compute_spectral_density_curve, extract_band_power_features
from model_trainer import EEGClassifierSystem
from communicator import MindToTextSystem
from preprocessing import create_epochs_from_events, preprocess_raw


def run_pipeline_test():
    print("=" * 60)
    print("NEUROSCRIPT PIPELINE VERIFICATION TEST")
    print("=" * 60)

    # 1. Test Dataset Availability
    available_runs = list_available_runs(subject=1)
    print(f"\n[1] Available PhysioNet S001 runs detected: {list(available_runs.keys())}")
    
    if len(available_runs) == 0:
        print("No PhysioNet files found. Testing synthetic demo generator...")
        raw, events = generate_synthetic_demo_run(duration_sec=30.0)
        assert len(events) > 0, "Synthetic events failed to generate!"
        print(f"Synthetic demo generated successfully with {len(events)} events.")
        return

    # 2. Test Real EEG Loading for Run 4 (Left vs Right imagery)
    r4_path = available_runs[4]
    raw_r4, events_r4 = load_real_eeg_run(r4_path, run_id=4)
    print(f"[2] Loaded real EEG Run 4: {raw_r4.info['nchan']} channels, {raw_r4.times[-1]:.1f}s")
    print(f"    Events count: {len(events_r4)}, Sample event: {events_r4[1]}")
    assert raw_r4.info['nchan'] == 64, "Expected 64 channels in PhysioNet EEGBCI"
    assert len(events_r4) == 30, "Expected 30 events in Run 4"

    # 3. Test Preprocessing (Bandpass Filtering)
    raw_filt = preprocess_raw(raw_r4, l_freq=8.0, h_freq=30.0)
    print(f"[3] Filtered EEG (8-30 Hz Mu/Beta band): highpass={raw_filt.info['highpass']}Hz, lowpass={raw_filt.info['lowpass']}Hz")

    # 4. Test Epoching
    epochs = create_epochs_from_events(raw_filt, tmin=0.5, tmax=3.5, selected_channels=MOTOR_CHANNELS)
    print(f"[4] Epoched data: {len(epochs)} epochs, shape={epochs.get_data().shape}")
    assert epochs.get_data().shape[1] == len(MOTOR_CHANNELS), "Channel pick mismatch"

    # 5. Test Feature Extraction
    data = epochs.get_data()
    X, feat_names = extract_band_power_features(data, sfreq=epochs.info['sfreq'], channel_names=MOTOR_CHANNELS)
    print(f"[5] Extracted features: shape={X.shape}, n_features={len(feat_names)}")
    assert not np.isnan(X).any(), "NaN values found in extracted features"

    # 6. Test Multi-Run Aggregation & Leakage-Free Classification
    runs_to_load = [r for r in [4, 8, 12] if r in available_runs]
    print(f"\n[6] Preparing Multi-Run Dataset for Motor Imagery (Runs: {runs_to_load})...")
    
    all_X = []
    all_y = []
    all_groups = []
    
    for r_id in runs_to_load:
        raw_r, evs_r = load_real_eeg_run(available_runs[r_id], run_id=r_id)
        raw_r_filt = preprocess_raw(raw_r, l_freq=8.0, h_freq=30.0)
        epochs_r = create_epochs_from_events(raw_r_filt, tmin=0.5, tmax=3.5, selected_channels=MOTOR_CHANNELS)
        
        # Only Left (T1: 2) and Right (T2: 3)
        evs = epochs_r.events
        mask = np.isin(evs[:, 2], [2, 3])
        if mask.sum() > 0:
            X_sub, _ = extract_band_power_features(epochs_r[mask].get_data(), sfreq=epochs_r.info['sfreq'], channel_names=MOTOR_CHANNELS)
            all_X.append(X_sub)
            all_y.append(evs[mask, 2])
            all_groups.append(np.full(mask.sum(), r_id))

    X_mat = np.concatenate(all_X, axis=0)
    y_mat = np.concatenate(all_y, axis=0)
    groups_mat = np.concatenate(all_groups, axis=0)

    print(f"    Total Trials: {len(y_mat)} (Left vs Right imagery across runs)")

    clf_system = EEGClassifierSystem(n_estimators=100, random_state=42)
    eval_results = clf_system.fit_and_evaluate(
        X=X_mat,
        y=y_mat,
        groups=groups_mat,
        split_method="run_holdout",
        test_run_id=12,
        feature_names=feat_names
    )

    print("\n[7] Evaluation Results (Strict Run-Based Split):")
    print(f"    Method: {eval_results['method']}")
    print(f"    Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"    Balanced Accuracy: {eval_results['balanced_accuracy']:.4f}")
    print(f"    F1 Score: {eval_results['f1_score']:.4f}")
    print(f"    Confusion Matrix:\n{eval_results['confusion_matrix']}")
    print(f"    Data Leakage Note: {eval_results['leakage_note'][:120]}...")

    # 8. Test Single Trial Prediction & Confidence
    pred_class, conf, probs = clf_system.predict_single(X_mat[0])
    print(f"\n[8] Single Trial Inference:")
    print(f"    Predicted ID: {pred_class}, Calibrated Confidence: {conf:.3f}, Class Probs: {probs}")

    # 9. Test Mind-to-Text Communicator & TTS
    comm = MindToTextSystem()
    task_label = "Left Fist Imagery" if pred_class == 2 else "Right Fist Imagery"
    entry = comm.record_prediction(
        predicted_class=task_label,
        confidence=conf,
        actual_class="Left Fist Imagery",
        source="PhysioNet Real EEG S001",
        run_id=12,
        trial_id=1
    )
    print(f"\n[9] Mind-to-Text Mapping:")
    print(f"    Classification: '{entry['predicted_task']}' -> Mapped Command: '{entry['mapped_command']}'")
    print(f"    Sentence Buffer: '{comm.get_sentence_text()}'")
    
    # Add a second command
    comm.record_prediction(predicted_class="Right Fist Imagery", confidence=0.72)
    print(f"    Updated Sentence Buffer: '{comm.get_sentence_text()}'")
    
    # Verify history
    df_history = comm.get_history_dataframe()
    print(f"    History Log Entries: {len(df_history)}")

    # 10. Test Synthetic Demo Mode explicitly
    raw_synth, evs_synth = generate_synthetic_demo_run(duration_sec=30.0)
    print(f"\n[10] Synthetic Mode Verification:")
    print(f"     Synthetic Events Count: {len(evs_synth)}")
    print(f"     Label check: '{evs_synth[0]['label']}' (is_synthetic={evs_synth[0]['is_synthetic']})")
    assert evs_synth[0]["is_synthetic"] is True, "Synthetic flag must be True"

    print("\n" + "=" * 60)
    print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()
