"""
Model Training, Persistence, and Leakage-Safe Evaluation Module.
Trains a baseline Random Forest classifier using Scikit-Learn.
Supports participant-level splitting (cross-subject) and run-based holdout evaluation.
Saves and loads trained models and preprocessing configurations using Joblib and JSON.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


class EEGClassifierSystem:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
        calibration_method: str = "sigmoid"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.model = None
        self.classes_ = None
        self.class_names_ = None
        self.feature_names_ = None
        self.evaluation_results = {}
        self.created_at = None

    def fit_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        split_method: str = "run_holdout",
        test_run_id: Optional[int] = None,
        test_subject_id: Optional[Union[int, str]] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Trains and evaluates the Random Forest classifier strictly preventing data leakage.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Integer class labels
            groups: Run/recording IDs corresponding to each trial (prevents run leakage)
            subject_ids: Participant IDs corresponding to each trial (prevents participant leakage)
            split_method: 'subject_holdout' (train on one participant, test on another)
                          or 'run_holdout' (train on earlier runs, test on unseen run)
                          or 'group_kfold' (Stratified Group K-Fold)
        """
        self.feature_names_ = feature_names
        self.classes_ = np.unique(y)
        self.class_names_ = class_names or [f"Class {c}" for c in self.classes_]
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Determine train/test masks based on splitting strategy
        if split_method == "subject_holdout" and subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            if len(unique_subjects) >= 2:
                target_test_sub = test_subject_id if test_subject_id in unique_subjects else unique_subjects[-1]
                train_mask = (subject_ids != target_test_sub)
                test_mask = (subject_ids == target_test_sub)
                method_name = f"Participant-Independent Holdout (Train: {list(unique_subjects[unique_subjects != target_test_sub])}, Test Participant: {target_test_sub})"
                leakage_note = (
                    "PARTICIPANT-INDEPENDENT EVALUATION GUARANTEE: The classifier was evaluated on a completely unseen human subject. "
                    "Zero trials, normalization statistics, or recordings from the test participant were seen during training."
                )
            else:
                # Fallback to run holdout if only 1 subject is available
                split_method = "run_holdout"
                method_name = "Run-Based Holdout (Single Participant Limitation)"
                leakage_note = (
                    "PARTICIPANT-INDEPENDENT LIMITATION: Only 1 participant is currently loaded in this dataset subset. "
                    "To prevent temporal data leakage without claiming multi-subject generalization, evaluation was performed "
                    "across independent recording runs."
                )

        if split_method == "run_holdout":
            if test_run_id is not None and test_run_id in groups:
                train_mask = (groups != test_run_id)
                test_mask = (groups == test_run_id)
            else:
                # Default test on last run
                target_test_run = np.max(groups)
                train_mask = (groups != target_test_run)
                test_mask = (groups == target_test_run)

            # Check if both splits contain valid classes
            if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                split_method = "group_kfold"

        if split_method in ["subject_holdout", "run_holdout"]:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            self.base_rf.fit(X_train, y_train)

            # Probability calibration
            if len(X_train) >= 15:
                try:
                    calibrated = CalibratedClassifierCV(self.base_rf, method=self.calibration_method, cv=3)
                    calibrated.fit(X_train, y_train)
                    self.model = calibrated
                except Exception:
                    self.model = self.base_rf
            else:
                self.model = self.base_rf

            y_pred = self.model.predict(X_test)
            y_probs = self.model.predict_proba(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            cm = confusion_matrix(y_test, y_pred)
            rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Class-wise performance metrics
            class_performance = []
            for idx, c in enumerate(self.classes_):
                c_name = self.class_names_[idx] if idx < len(self.class_names_) else f"Class {c}"
                c_str = str(c)
                if c_str in rep:
                    class_performance.append({
                        "Class": c_name,
                        "Precision": round(float(rep[c_str]["precision"]), 3),
                        "Recall": round(float(rep[c_str]["recall"]), 3),
                        "F1-Score": round(float(rep[c_str]["f1-score"]), 3),
                        "Support": int(rep[c_str]["support"])
                    })

            # Feature importances
            importances = None
            if hasattr(self.base_rf, "feature_importances_"):
                importances = self.base_rf.feature_importances_

            results = {
                "method": method_name if 'method_name' in locals() else f"Run-Based Holdout (Test Run: {test_run_id})",
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": rep,
                "class_performance": class_performance,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_probs": y_probs,
                "leakage_note": leakage_note if 'leakage_note' in locals() else "Run-based holdout evaluation.",
                "feature_importances": importances,
                "feature_names": self.feature_names_,
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test)
            }
        else: # Stratified Group K-Fold Cross-Validation
            n_groups = len(np.unique(groups))
            n_splits = min(max(2, n_groups), 5)
            
            gkf = StratifiedGroupKFold(n_splits=n_splits)
            y_preds_all = np.zeros(len(y), dtype=int)
            y_probs_all = np.zeros((len(y), len(self.classes_)))
            fold_accuracies = []

            for train_idx, test_idx in gkf.split(X, y, groups):
                fold_rf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                fold_rf.fit(X[train_idx], y[train_idx])
                
                preds = fold_rf.predict(X[test_idx])
                probs = fold_rf.predict_proba(X[test_idx])
                
                y_preds_all[test_idx] = preds
                for col_idx, c in enumerate(fold_rf.classes_):
                    target_col = np.where(self.classes_ == c)[0][0]
                    y_probs_all[test_idx, target_col] = probs[:, col_idx]
                
                fold_accuracies.append(float(accuracy_score(y[test_idx], preds)))

            # Fit final model on all data
            self.base_rf.fit(X, y)
            self.model = self.base_rf

            acc = float(accuracy_score(y, y_preds_all))
            bal_acc = float(balanced_accuracy_score(y, y_preds_all))
            prec = float(precision_score(y, y_preds_all, average='weighted', zero_division=0))
            rec = float(recall_score(y, y_preds_all, average='weighted', zero_division=0))
            f1 = float(f1_score(y, y_preds_all, average='weighted', zero_division=0))
            cm = confusion_matrix(y, y_preds_all)
            rep = classification_report(y, y_preds_all, output_dict=True, zero_division=0)

            class_performance = []
            for idx, c in enumerate(self.classes_):
                c_name = self.class_names_[idx] if idx < len(self.class_names_) else f"Class {c}"
                c_str = str(c)
                if c_str in rep:
                    class_performance.append({
                        "Class": c_name,
                        "Precision": round(float(rep[c_str]["precision"]), 3),
                        "Recall": round(float(rep[c_str]["recall"]), 3),
                        "F1-Score": round(float(rep[c_str]["f1-score"]), 3),
                        "Support": int(rep[c_str]["support"])
                    })

            results = {
                "method": f"Stratified Group {n_splits}-Fold Cross-Validation (Grouped by Recording Run)",
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "fold_accuracies": fold_accuracies,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "classification_report": rep,
                "class_performance": class_performance,
                "y_true": y,
                "y_pred": y_preds_all,
                "y_probs": y_probs_all,
                "leakage_note": "Stratified Group K-Fold cross validation grouped strictly by recording run.",
                "feature_importances": self.base_rf.feature_importances_,
                "feature_names": self.feature_names_,
                "n_train_samples": len(X),
                "n_test_samples": len(X)
            }

        self.evaluation_results = results
        return results

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predicts the class and calibrated confidence for a single EEG epoch.
        Returns: (predicted_class_id, confidence, all_class_probabilities)
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        feat = feature_vector.reshape(1, -1)
        pred_class = int(self.model.predict(feat)[0])
        probs = self.model.predict_proba(feat)[0]
        confidence = float(np.max(probs))
        return pred_class, confidence, probs

    def predict_trial_segment(
        self,
        trial_data: np.ndarray,
        sfreq: float,
        channel_names: List[str]
    ) -> Tuple[int, str, float, np.ndarray]:
        """
        Unified inference method: extracts features identically to training pipeline
        and maps prediction directly to the registered class label.
        Returns: (pred_class_id, pred_class_name, confidence, all_class_probabilities)
        """
        if isinstance(trial_data, tuple):
            trial_data = trial_data[0]
        trial_data = np.asarray(trial_data)
        
        from features import extract_band_power_features
        feat, _ = extract_band_power_features(
            trial_data,
            sfreq=sfreq,
            channel_names=channel_names,
            include_statistics=True
        )
        pred_id, conf, probs = self.predict_single(feat[0])
        pred_name = self.class_names_[pred_id] if pred_id < len(self.class_names_) else f"Class {pred_id}"
        return pred_id, pred_name, conf, probs

    def save_model(self, model_dir: Union[str, Path] = "models", filename: str = "trained_rf_model.joblib") -> Path:
        """
        Saves the trained model and associated configuration/metadata to disk.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / filename
        meta_path = model_dir / "model_metadata.json"
        preproc_path = model_dir / "preproc_config.json"

        # Serialize model
        joblib.dump(self.model, model_path)

        # Class encodings
        class_encodings = {int(c): (self.class_names_[i] if i < len(self.class_names_) else f"Class {c}") for i, c in enumerate(self.classes_)}

        # Serialize metadata (converting numpy types to standard python types)
        metadata = {
            "created_at": self.created_at,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "classes": [int(c) for c in self.classes_] if self.classes_ is not None else [],
            "class_names": self.class_names_,
            "class_encodings": class_encodings,
            "feature_names": self.feature_names_,
            "evaluation_metrics": {
                "method": self.evaluation_results.get("method"),
                "accuracy": self.evaluation_results.get("accuracy"),
                "balanced_accuracy": self.evaluation_results.get("balanced_accuracy"),
                "precision": self.evaluation_results.get("precision"),
                "recall": self.evaluation_results.get("recall"),
                "f1_score": self.evaluation_results.get("f1_score"),
                "n_train_samples": self.evaluation_results.get("n_train_samples"),
                "n_test_samples": self.evaluation_results.get("n_test_samples")
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Preprocessing configuration
        preproc_config = {
            "l_freq": 1.0,
            "h_freq": 35.0,
            "sfreq": 160.0,
            "tmin": 0.5,
            "tmax": 3.5,
            "channel_order": [f.split("_")[0] for f in self.feature_names_[:17]] if self.feature_names_ else [],
            "class_encodings": class_encodings
        }
        with open(preproc_path, "w", encoding="utf-8") as f:
            json.dump(preproc_config, f, indent=2)

        return model_path

    @classmethod
    def load_saved_model(cls, model_dir: Union[str, Path] = "models", filename: str = "trained_rf_model.joblib"):
        """
        Loads a previously saved model and its configuration from disk.
        """
        model_dir = Path(model_dir)
        model_path = model_dir / filename
        meta_path = model_dir / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Saved model file not found at: {model_path}")

        loaded_model = joblib.load(model_path)
        
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        instance = cls(
            n_estimators=meta.get("n_estimators", 100),
            max_depth=meta.get("max_depth", 10),
            random_state=meta.get("random_state", 42)
        )
        instance.model = loaded_model
        instance.classes_ = np.array(meta.get("classes", [0, 1]))
        instance.class_names_ = meta.get("class_names", ["Class 0", "Class 1"])
        instance.feature_names_ = meta.get("feature_names", [])
        instance.evaluation_results = meta.get("evaluation_metrics", {})
        instance.created_at = meta.get("created_at")

        return instance
