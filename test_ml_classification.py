"""
Automated Verification for Step 2: EEG Feature Extraction and Machine Learning Classification.
Tests dataset preparation, multi-band feature extraction (Delta, Theta, Alpha, Beta),
Random Forest training with run-based and participant-based splits, model persistence,
and real single-trial inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import MOTOR_CHANNELS
from data_loader import list_available_runs
from features import (
    extract_band_power_features,
    extract_single_trial_features_table,
)
from model_trainer import EEGClassifierSystem
from preprocessing import prepare_labelled_dataset_segments


def test_step2_feature_extraction_and_ml():
    print("=" * 65)
    print("TESTING STEP 2: EEG FEATURE EXTRACTION & MACHINE LEARNING")
    print("=" * 65)

    sub1_runs = list_available_runs(subject=1)
    sub2_runs = list_available_runs(subject=2)
    print(f"\n[1] Available Participant Recordings:")
    print(f"    - S001 runs: {sorted(list(sub1_runs.keys()))}")
    print(f"    - S002 runs: {sorted(list(sub2_runs.keys()))}")

    # Build recording list for Subject 1 (Runs 4, 8, 12) and Subject 2 (Runs 4, 8)
    recordings_info = []
    for r_id in [4, 8, 12]:
        if r_id in sub1_runs:
            recordings_info.append({"path": sub1_runs[r_id], "run_id": r_id, "subject": "S001"})
    for r_id in [4, 8]:
        if r_id in sub2_runs:
            recordings_info.append({"path": sub2_runs[r_id], "run_id": r_id, "subject": "S002"})

    print(f"\n[2] Loaded {len(recordings_info)} total recording runs across participants.")

    # 1. EEG Data Preparation & Invalid Segment Exclusion
    seg_prep = prepare_labelled_dataset_segments(
        recordings_info=recordings_info,
        task_mode="2-class",
        selected_channels=MOTOR_CHANNELS,
        tmin=0.5,
        tmax=3.5
    )
    print(f"\n[3] Labelled Segment Preparation:")
    print(f"    - Total inspected segments: {seg_prep['total_segments_checked']}")
    print(f"    - Excluded invalid/boundary segments: {seg_prep['excluded_segments_count']}")
    print(f"    - Valid segments by class: {seg_prep['class_counts']}")
    print(f"    - Extracted epochs shape: {seg_prep['X_epochs'].shape}")
    assert seg_prep["X_epochs"].shape[0] > 0, "No valid segments extracted!"

    # 2. Multi-Band Feature Extraction (Delta, Theta, Alpha, Beta, Mean, StdDev)
    X_features, feat_names = extract_band_power_features(
        epochs_data=seg_prep["X_epochs"],
        sfreq=seg_prep["sfreq"],
        channel_names=seg_prep["channels"],
        include_statistics=True
    )
    print(f"\n[4] Multi-Band Feature Extraction:")
    print(f"    - Feature matrix shape: {X_features.shape}")
    print(f"    - Total feature dimensions: {len(feat_names)}")
    print(f"    - Sample features: {feat_names[:6]}")
    assert not np.isnan(X_features).any(), "Found NaN in feature matrix"
    # Verify Delta, Theta, Alpha, Beta are present in feature names
    assert any("Delta" in f for f in feat_names), "Delta band power missing"
    assert any("Theta" in f for f in feat_names), "Theta band power missing"
    assert any("Alpha" in f for f in feat_names), "Alpha band power missing"
    assert any("Beta" in f for f in feat_names), "Beta band power missing"
    assert any("Mean" in f for f in feat_names), "Mean statistic missing"
    assert any("StdDev" in f for f in feat_names), "StdDev statistic missing"

    # Test single-trial features table
    df_single_table = extract_single_trial_features_table(
        seg_prep["X_epochs"][0],
        sfreq=seg_prep["sfreq"],
        channel_names=seg_prep["channels"]
    )
    print(f"    - Single-trial feature table preview:\n{df_single_table.head(3)}")

    # 3. Model Training & Within-Subject Run-Based Holdout Evaluation (S001)
    s1_mask = (seg_prep["subjects"] == "S001")
    X_s1 = X_features[s1_mask]
    y_s1 = seg_prep["y"][s1_mask]
    groups_s1 = seg_prep["groups"][s1_mask]

    clf_run_holdout = EEGClassifierSystem(n_estimators=100, random_state=42)
    eval_run = clf_run_holdout.fit_and_evaluate(
        X=X_s1,
        y=y_s1,
        groups=groups_s1,
        split_method="run_holdout",
        test_run_id=12,
        feature_names=feat_names,
        class_names=seg_prep["class_names"]
    )

    print(f"\n[5] Within-Subject Run-Based Holdout Results (Train Runs 4 & 8, Test Run 12):")
    print(f"    - Accuracy: {eval_run['accuracy'] * 100:.1f}%")
    print(f"    - Balanced Accuracy: {eval_run['balanced_accuracy'] * 100:.1f}%")
    print(f"    - Weighted F1 Score: {eval_run['f1_score']:.3f}")
    print(f"    - Train Samples: {eval_run['n_train_samples']}, Test Samples: {eval_run['n_test_samples']}")
    print(f"    - Confusion Matrix:\n{eval_run['confusion_matrix']}")
    print(f"    - Class-wise performance:\n{pd.DataFrame(eval_run['class_performance'])}")

    # 4. Model Training & Cross-Subject Participant-Independent Evaluation (Train S001, Test S002)
    clf_subject_holdout = EEGClassifierSystem(n_estimators=100, random_state=42)
    eval_sub = clf_subject_holdout.fit_and_evaluate(
        X=X_features,
        y=seg_prep["y"],
        groups=seg_prep["groups"],
        subject_ids=seg_prep["subjects"],
        split_method="subject_holdout",
        test_subject_id="S002",
        feature_names=feat_names,
        class_names=seg_prep["class_names"]
    )

    print(f"\n[6] Cross-Subject Participant-Independent Results (Train S001, Test S002):")
    print(f"    - Evaluation Method: {eval_sub['method']}")
    print(f"    - Cross-Subject Test Accuracy: {eval_sub['accuracy'] * 100:.1f}%")
    print(f"    - Train Samples (S001): {eval_sub['n_train_samples']}, Test Samples (S002): {eval_sub['n_test_samples']}")
    print(f"    - Confusion Matrix:\n{eval_sub['confusion_matrix']}")

    # 5. Model Persistence (Save & Load)
    save_path = clf_run_holdout.save_model(model_dir="models", filename="trained_rf_model.joblib")
    print(f"\n[7] Model Persistence:")
    print(f"    - Model saved to: {save_path}")
    assert Path(save_path).exists(), "Model file was not created"
    assert Path("models/model_metadata.json").exists(), "Metadata file was not created"

    # Load back
    loaded_clf = EEGClassifierSystem.load_saved_model(model_dir="models")
    print(f"    - Successfully loaded model back from disk.")
    assert loaded_clf.model is not None

    # 6. Single-Trial Prediction with Calibrated Confidence
    pred_class, conf, probs = loaded_clf.predict_single(X_features[0])
    print(f"\n[8] Single Trial Inference:")
    print(f"    - Predicted Class Index: {pred_class} ({seg_prep['class_names'][pred_class]})")
    print(f"    - Calibrated Confidence: {conf * 100:.1f}%")
    print(f"    - Probability Distribution: {probs}")
    assert 0.0 <= conf <= 1.0

    print("\n" + "=" * 65)
    print("ALL STEP 2 ML & FEATURE EXTRACTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_step2_feature_extraction_and_ml()
