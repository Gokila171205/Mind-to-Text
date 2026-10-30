"""
Neuroscript Configuration Module
Central definitions for EEG channels, frequency bands, default communication mappings,
and scientific disclaimers.
"""

# PhysioNet EEGBCI standard motor cortex channels (64-channel 10-05 system subset)
MOTOR_CHANNELS = [
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
]

# Core channels for focal ERD/ERS inspection
CORE_CHANNELS = ['C3', 'Cz', 'C4']

# Standard Physiological Frequency Bands (in Hz)
DELTA_BAND = (0.5, 4.0)    # Deep sleep, slow wave
THETA_BAND = (4.0, 8.0)    # Drowsiness, sensorimotor gating
ALPHA_BAND = (8.0, 12.0)   # Mu / Alpha rhythm (sensorimotor idling / ERD)
MU_BAND = (8.0, 12.0)      # Sensorimotor Mu rhythm
BETA_BAND = (13.0, 30.0)   # Motor planning, active movement
GAMMA_BAND = (30.0, 45.0)  # High-frequency cognitive processing

# Standard filter passband preserving Delta to Beta for feature extraction and classification
CLASSIFIER_PASSBAND = (1.0, 35.0)
BROAD_PASSBAND = (0.5, 40.0)
PASSBAND = (1.0, 35.0)     # Default preprocessing passband

# Sampling frequency of PhysioNet EEGBCI dataset
SFREQ_DEFAULT = 160.0

# PhysioNet Run Type definitions
RUN_INFO = {
    1: {"name": "Baseline, Eyes Open", "type": "baseline", "classes": ["Rest"]},
    2: {"name": "Baseline, Eyes Closed", "type": "baseline", "classes": ["Rest"]},
    3: {"name": "Motor Execution: Left vs Right Fist", "type": "execution", "classes": ["Rest", "Left Fist (Exec)", "Right Fist (Exec)"]},
    4: {"name": "Motor Imagery: Left vs Right Fist", "type": "imagery", "classes": ["Rest", "Left Fist Imagery", "Right Fist Imagery"]},
    5: {"name": "Motor Execution: Both Fists vs Both Feet", "type": "execution", "classes": ["Rest", "Both Fists (Exec)", "Both Feet (Exec)"]},
    6: {"name": "Motor Imagery: Both Fists vs Both Feet", "type": "imagery", "classes": ["Rest", "Both Fists Imagery", "Both Feet Imagery"]},
    7: {"name": "Motor Execution: Left vs Right Fist (Repeat)", "type": "execution", "classes": ["Rest", "Left Fist (Exec)", "Right Fist (Exec)"]},
    8: {"name": "Motor Imagery: Left vs Right Fist (Repeat)", "type": "imagery", "classes": ["Rest", "Left Fist Imagery", "Right Fist Imagery"]},
    9: {"name": "Motor Execution: Both Fists vs Both Feet (Repeat)", "type": "execution", "classes": ["Rest", "Both Fists (Exec)", "Both Feet (Exec)"]},
    10: {"name": "Motor Imagery: Both Fists vs Both Feet (Repeat)", "type": "imagery", "classes": ["Rest", "Both Fists Imagery", "Both Feet Imagery"]},
    11: {"name": "Motor Execution: Left vs Right Fist (Repeat)", "type": "execution", "classes": ["Rest", "Left Fist (Exec)", "Right Fist (Exec)"]},
    12: {"name": "Motor Imagery: Left vs Right Fist (Repeat)", "type": "imagery", "classes": ["Rest", "Left Fist Imagery", "Right Fist Imagery"]},
    13: {"name": "Motor Execution: Both Fists vs Both Feet (Repeat)", "type": "execution", "classes": ["Rest", "Both Fists (Exec)", "Both Feet (Exec)"]},
    14: {"name": "Motor Imagery: Both Fists vs Both Feet (Repeat)", "type": "imagery", "classes": ["Rest", "Both Fists Imagery", "Both Feet Imagery"]},
}

# Default Application-Defined Communication Mapping
DEFAULT_MAPPING = {
    "Rest": "WAIT",
    "Left Fist Imagery": "LEFT",
    "Right Fist Imagery": "RIGHT",
    "Both Feet Imagery": "HELP",
    "Both Fists Imagery": "SELECT",
    "Left Fist (Exec)": "LEFT",
    "Right Fist (Exec)": "RIGHT",
    "Both Feet (Exec)": "HELP",
    "Both Fists (Exec)": "SELECT",
    "Left Fist": "LEFT",
    "Right Fist": "RIGHT",
    "Both Feet": "HELP",
    "Both Fists": "SELECT",
}

# Scientific Limitations Notice
SCIENTIFIC_DISCLAIMER = """
**Scientific Limitations & Reality Check:**
- **No Mind Reading:** This system DOES NOT read natural language thoughts, inner speech, or arbitrary cognitive states.
- **Sensorimotor Rhythms (SMR):** The classifier detects Event-Related Desynchronization (ERD) and Synchronization (ERS) in the sensorimotor cortex caused by imagining physical movement.
- **Application-Defined Mapping:** The mapping from motor imagery tasks (e.g. *Left Fist* -> *LEFT*, *Both Feet* -> *HELP*) is purely an engineering convention assigned by the application, NOT a semantic meaning inherent in the brain signal.
- **Signal-to-Noise Ratio (SNR):** Non-invasive scalp EEG is noisy, non-stationary, and subject-dependent (~20% of users exhibit BCI illiteracy). Calibrated probabilities are reported to avoid false overconfidence.
"""
