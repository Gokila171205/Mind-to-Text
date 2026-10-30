# Neuroscript: Mind-to-Text Communication System using EEG Motor Imagery Classification

A complete, working Brain-Computer Interface (BCI) prototype that loads **real, labelled EEG data** from the public [PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)](https://physionet.org/content/eegmmidb/1.0.0/), visualizes sensorimotor task waveforms, extracts Welch band power features (Mu & Beta rhythms), trains a calibrated Random Forest classifier with **strict leakage prevention**, and maps predicted motor imagery classes to an assistive communication interface with **in-browser Text-to-Speech (TTS)**.

---

## 🧠 Scientific Foundations & Reality Check

### What this system DOES:
- **Sensorimotor Rhythms (SMR):** Detects **Event-Related Desynchronization (ERD)** and **Event-Related Synchronization (ERS)** in the sensorimotor cortex (centered over $C_3$, $C_z$, and $C_4$). When a user imagines moving their left fist, Mu-band (8–12 Hz) power decreases contralaterally over the right hemisphere ($C_4$).
- **Application-Defined Mapping:** The mapping from motor imagery tasks to communication tokens:
  - $\text{Rest} \longrightarrow \textbf{WAIT}$
  - $\text{Left Fist Imagery} \longrightarrow \textbf{LEFT}$
  - $\text{Right Fist Imagery} \longrightarrow \textbf{RIGHT}$
  - $\text{Both Feet Imagery} \longrightarrow \textbf{HELP}$
  - $\text{Both Fists Imagery} \longrightarrow \textbf{SELECT}$
  is an **engineering convention** defined by this application, **not a thought or word read directly from the brain**.

### What this system DOES NOT do:
- **No Mind Reading:** Scalp EEG cannot read arbitrary cognitive thoughts, inner speech, or semantic intent.
- **No Direct Yes/No Intentions:** The PhysioNet dataset does not contain conversational intentions; it contains motor imagery trials.
- **Honest Performance:** Non-invasive scalp EEG is noisy and subject-dependent (~20% of users exhibit BCI illiteracy). Models evaluated honestly across independent recording runs achieve typical baseline accuracies (~50–70%), unlike leaked models that report fake 99% accuracy.

---

## 🛡️ Leakage Prevention Architecture

In EEG analysis, **data leakage** is a widespread pitfall:
1. **The Flaw:** Shuffling sliding windows or overlapping epochs randomly between training and test sets leaks strong temporal auto-correlations across neighboring windows of the exact same trial, yielding artificially inflated (>90%) accuracy that collapses in real deployment.
2. **Our Solution:** We strictly enforce **Recording/Run-Based Separation**:
   - **Run Holdout Evaluation:** The model is trained on independent runs (e.g. Runs 4 & 8) and evaluated on a completely unseen future run (e.g. Run 12).
   - **Stratified Group K-Fold:** Cross-validation folds are grouped by recording run so that no trial from a test recording ever enters the training set.

---

## 📂 Project Architecture

```
Neuroscript/
├── app.py                # Main Streamlit web application & interactive BCI interface
├── config.py             # Channel definitions, frequency bands, default mappings
├── data_loader.py        # Real PhysioNet EDF+ downloader/loader via MNE + Synthetic demo generator
├── preprocessing.py      # Bandpass (8-30 Hz Mu/Beta), notch filtering, standard 10-05 montage, epoching
├── features.py           # Welch Power Spectral Density, log band power (Mu, Beta, Variance, Ratio)
├── model_trainer.py      # Random Forest with probability calibration & run-based leakage prevention
├── communicator.py       # Assistive virtual keyboard, sentence builder, history log, browser TTS
├── test_pipeline.py      # Automated end-to-end verification test suite
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## 🚀 Setup & Running Locally

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone / Open Directory
```bash
cd Neuroscript
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Step 1 Dataset & Visualization Verification Test
```bash
python test_dataset_visualization.py
```

### 5. Run Step 2 Feature Extraction & ML Classification Test
```bash
python test_ml_classification.py
```

### 6. Run Full End-to-End Pipeline Test
```bash
python test_pipeline.py
```

### 6. Launch Both Backend and Frontend

You can run both the **FastAPI Backend REST API** and the **Streamlit Web Application** concurrently:

#### Option A: Start Backend (FastAPI REST API)
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```
- **Backend API URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Endpoints:**
  - `GET /api/health`: Health status and scientific reality check.
  - `GET /api/recordings`: List available PhysioNet recordings.
  - `GET /api/recording/{run_id}/metadata`: Full recording metadata and channels.
  - `GET /api/recording/{run_id}/events`: Labelled task segments (`T0`, `T1`, `T2`).
  - `GET /api/recording/{run_id}/signal`: Raw and filtered waveforms for any channel.
  - `POST /api/predict`: Model classification and communication mapping.
  - `GET /api/history`: Classification history.

#### Option B: Start Frontend (Streamlit Dashboard)
```bash
python -m streamlit run app.py
```
- **Frontend URL:** `http://localhost:8501`
- Includes interactive recording loader, Matplotlib EEG visualization, multi-task comparisons, leakage-free ML evaluation, and browser Text-to-Speech (TTS).

---

## 🏷️ PhysioNet Annotation Semantics by Run Type

PhysioNet EEGMMIDB records 14 experimental runs per subject. **Annotations must be interpreted based on the specific run type**:

| Run Numbers | Experimental Protocol | `T0` Annotation | `T1` Annotation | `T2` Annotation |
| :--- | :--- | :--- | :--- | :--- |
| **Run 1** | Baseline, Eyes Open | Rest | — | — |
| **Run 2** | Baseline, Eyes Closed | Rest | — | — |
| **Runs 3, 7, 11** | Motor Execution (Fists) | Rest | **Left Fist (Execution)** | **Right Fist (Execution)** |
| **Runs 4, 8, 12** | Motor Imagery (Fists) | Rest | **Left Fist Imagery** | **Right Fist Imagery** |
| **Runs 5, 9, 13** | Motor Execution (Both Fists / Feet) | Rest | **Both Fists (Execution)** | **Both Feet (Execution)** |
| **Runs 6, 10, 14** | Motor Imagery (Both Fists / Feet) | Rest | **Both Fists Imagery** | **Both Feet Imagery** |

> ⚠️ **Important:** Do not assume `T1` or `T2` mean the same task across all runs! In Run 4, `T2` represents **Right Fist Imagery**, while in Run 6, `T2` represents **Both Feet Imagery**. Our [data_loader.py](file:///c:/Users/dharanesh/OneDrive/Desktop/Neuroscript/data_loader.py) correctly maps each annotation based on the run's experimental context.

---

## 📥 Manual Download Procedure (If Offline)

If automated download via MNE is blocked by network restrictions:
1. Visit the official PhysioNet page: `https://physionet.org/content/eegmmidb/1.0.0/S001/`
2. Download the needed `.edf` files (e.g. `S001R01.edf`, `S001R04.edf`, `S001R06.edf`, `S001R08.edf`, `S001R10.edf`, `S001R12.edf`).
3. Place them in `~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/` or in a `data/` folder inside the project.
4. The application automatically detects them upon launch.

---

## 📊 Features & Workflow

1. **Dataset Explorer:**
   - One-click automatic download of real Subject 1 Motor Imagery runs directly from PhysioNet via MNE.
   - Switch between **Real PhysioNet EEG** and **Clearly Labelled Synthetic Demo Mode**.
   - Interactive waveform viewer displaying real EEG amplitude ($\mu\text{V}$) vs time (seconds) alongside the dataset's actual annotated event markers (`T0`, `T1`, `T2`).
2. **Spectral Analysis:**
   - Power Spectral Density (PSD) curves highlighting Mu (8–12 Hz) and Beta (13–30 Hz) bands.
   - Channel-wise power comparison across $C_3$, $C_z$, and $C_4$.
3. **Calibrated Machine Learning:**
   - Random Forest classifier with probability calibration.
   - 2-Class, 3-Class, or 4-Class motor imagery tasks.
   - Confusion matrix, balanced accuracy, and feature importances.
4. **Smart Virtual BCI Keyboard & Assistive Speech Synthesis:**
   - Full interactive on-screen **QWERTY Keyboard** (A–Z letters, Space, Delete, Clear, Select).
   - **Dual Operating Modes**:
     - **🟢 Manual Demo Mode:** Click keyboard buttons or navigate with directional controls to test typing and suggestions without EEG claims.
     - **🧠 EEG Prediction Mode:** Actions are triggered **strictly** by genuine EEG trial predictions.
   - **PhysioNet Run Verification:** Genuinely checks if the active run supports Left/Right/Feet/Fists without inventing fake labels.
   - **Safety Guards:** Configurable confidence threshold (safety gate), debounce cooldown timer, and trial segment deduplication.
   - **Dictionary Word Suggestions:** Fast prefix-matching on 1,500+ frequent & assistive English words (e.g. `HELP`, `WATER`, `DOCTOR`, `PLEASE`).
   - **Browser Text-to-Speech (TTS):** Integrated Web Speech API (`window.speechSynthesis`) to speak accumulated text aloud.
   - **Strict Provenance Tracking:** Audited action history clearly separating `Real EEG Prediction`, `Synthetic Demo Prediction`, and `Manual Click`.

---

## 🧪 Verification & Regression Testing

Run the full suite of automated verification tests:

```bash
# 1. Test Dataset Loader & Waveform Visualizer
python test_dataset_visualization.py

# 2. Test Feature Extraction & Honest ML Classification
python test_ml_classification.py

# 3. Test Full Pipeline Integration
python test_pipeline.py

# 4. Test Model Prediction & Root-Cause Regression Suite (9 Checks)
python test_regression_predictions.py

# 5. Test Smart Virtual BCI Keyboard (Navigation, Guards, Suggestions, Provenance)
python test_smart_virtual_keyboard.py
```

---

## 📚 References
- **Dataset:** Goldberger, A., et al. *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation 101 (23), pp. e215–e220 (2000).
- **BCI2000:** Schalk, G., et al. *BCI2000: a general-purpose brain-computer interface (BCI) system.* IEEE TBME 51(6), 1034–1043 (2004).
- **MNE-Python:** Gramfort, A., et al. *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7, 267 (2013).
