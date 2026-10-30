"""
Neuroscript: Mind-to-Text Communication System using EEG Motor Imagery Classification
Main Streamlit Application - Modular EEG Dataset, Waveform Visualization, and AI Classification
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import (
    ALPHA_BAND,
    BETA_BAND,
    CORE_CHANNELS,
    DEFAULT_MAPPING,
    DELTA_BAND,
    MOTOR_CHANNELS,
    MU_BAND,
    RUN_INFO,
    SCIENTIFIC_DISCLAIMER,
    SFREQ_DEFAULT,
    THETA_BAND,
)
from communicator import MindToTextSystem, generate_tts_html
from keyboard import (
    ACTION_DESCRIPTIONS,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_NEXT_ROW,
    ACTION_NO_ACTION,
    ACTION_SELECT,
    ACTION_SPACE,
    ALL_KEYS,
    DEFAULT_BCI_KEYBOARD_MAPPING,
    KEYBOARD_ROWS,
    QUICK_PHRASES,
    VirtualBCIKeyboard,
    get_word_suggestions,
)
from data_loader import (
    download_physionet_data,
    generate_synthetic_demo_run,
    get_manual_download_instructions,
    get_recording_metadata,
    list_available_runs,
    load_real_eeg_run,
)
from features import (
    compute_spectral_density_curve,
    extract_band_power_features,
    extract_single_trial_features_table,
)
from model_trainer import EEGClassifierSystem
from preprocessing import (
    create_epochs_from_events,
    get_single_trial_signal,
    prepare_labelled_dataset_segments,
    preprocess_raw,
    validate_recording_and_channels,
)
from visualization import (
    plot_multichannel_montage_matplotlib,
    plot_psd_matplotlib,
    plot_segment_comparison_matplotlib,
    plot_single_segment_matplotlib,
)

# ---------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Neuroscript | EEG AI Classification & Mind-to-Text",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(120deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }
    .badge-real {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-synthetic {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #b91c1c;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    .card-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .card-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .leakage-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        margin: 12px 0;
        color: #e2e8f0;
    }
    .sentence-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #4338ca;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .keyboard-text-display {
        background: linear-gradient(135deg, #090d16, #111827);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4), inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .text-output-string {
        font-family: 'Courier New', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #60a5fa;
        min-height: 48px;
        letter-spacing: 2px;
        word-break: break-all;
    }
    .provenance-real {
        background-color: #064e3b;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #059669;
    }
    .provenance-manual {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #4338ca;
    }
    .provenance-synthetic {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #b91c1c;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management & Persistent Model Cache
# ---------------------------------------------------------
if "communicator" not in st.session_state:
    st.session_state.communicator = MindToTextSystem()

if "virtual_keyboard" not in st.session_state:
    st.session_state.virtual_keyboard = VirtualBCIKeyboard(
        confidence_threshold=0.60,
        cooldown_seconds=1.0,
        nav_mode="ROW_COLUMN"
    )

@st.cache_resource(show_spinner=False)
def get_cached_bci_classifier(model_dir: str = "models") -> Optional[EEGClassifierSystem]:
    """Loads and caches the trained EEG classifier from disk to avoid slow disk reloads."""
    try:
        return EEGClassifierSystem.load_saved_model(model_dir=model_dir)
    except Exception:
        return None

if "trained_clf_system" not in st.session_state:
    st.session_state.trained_clf_system = get_cached_bci_classifier()

if "eval_results" not in st.session_state:
    if st.session_state.trained_clf_system is not None:
        st.session_state.eval_results = st.session_state.trained_clf_system.evaluation_results
    else:
        st.session_state.eval_results = None

# ---------------------------------------------------------
# Sidebar: Dataset Source & Participant Selection
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=56)
    st.title("Neuroscript BCI")
    st.caption("PhysioNet EEGMMIDB Motor Imagery System")

    st.markdown("---")
    st.subheader("Data Source Mode")
    data_mode = st.radio(
        "Operating Mode",
        options=["Real PhysioNet EEG", "Synthetic Demo Mode (Simulated)"],
        help="Use real PhysioNet EDF recordings or explore with clearly labeled simulated data."
    )

    is_synthetic = (data_mode == "Synthetic Demo Mode (Simulated)")
    subject_id = 1
    selected_run_id = 4
    raw_eeg = None
    labelled_events = []
    metadata = {}

    if not is_synthetic:
        # Check available participants
        sub1_runs = list_available_runs(subject=1)
        sub2_runs = list_available_runs(subject=2)
        
        available_subs = []
        if len(sub1_runs) > 0:
            available_subs.append(1)
        if len(sub2_runs) > 0:
            available_subs.append(2)

        if len(available_subs) == 0:
            st.warning("No PhysioNet recordings found locally.")
            if st.button("📥 Download Subject 1 Runs (PhysioNet)", type="primary"):
                with st.spinner("Downloading real EEG from PhysioNet..."):
                    try:
                        download_physionet_data(subject=1, runs=[1, 2, 4, 6, 8, 10, 12, 14], verbose=False)
                        st.success("Download complete! Refreshing...")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Download error: {err}")
        else:
            subject_id = st.selectbox(
                "Participant ID",
                options=available_subs,
                format_func=lambda s: f"Subject S{s:03d} ({len(list_available_runs(subject=s))} runs)"
            )

            available_runs = list_available_runs(subject=subject_id)
            run_options = sorted(list(available_runs.keys()))
            run_labels = {r: f"Run {r}: {RUN_INFO.get(r, {}).get('name', 'Recording')}" for r in run_options}
            selected_run_id = st.selectbox(
                "Select EEG Recording Run",
                options=run_options,
                format_func=lambda r: run_labels.get(r, f"Run {r}"),
                index=run_options.index(4) if 4 in run_options else 0
            )

            # Load the selected recording
            try:
                raw_eeg, labelled_events = load_real_eeg_run(available_runs[selected_run_id], selected_run_id)
                metadata = get_recording_metadata(
                    raw=raw_eeg,
                    run_id=selected_run_id,
                    file_path=available_runs[selected_run_id],
                    labelled_events=labelled_events
                )
            except Exception as e:
                st.error(f"Error loading recording: {e}")
    else:
        st.warning("⚠️ Synthetic Demo Mode active (clearly labeled as simulated).")
        raw_eeg, labelled_events = generate_synthetic_demo_run(duration_sec=90.0, random_seed=42)
        metadata = get_recording_metadata(
            raw=raw_eeg,
            run_id=4,
            file_path="SYNTHETIC_SIMULATED_DEMO_RUN.edf",
            labelled_events=labelled_events
        )

    st.markdown("---")
    st.subheader("Sensor Channel & Filtering")
    default_ch_list = [ch for ch in MOTOR_CHANNELS if raw_eeg is not None and ch in raw_eeg.ch_names] or (raw_eeg.ch_names if raw_eeg else ["C3"])
    selected_ch = st.selectbox(
        "Active EEG Channel",
        options=default_ch_list,
        index=default_ch_list.index("C3") if "C3" in default_ch_list else 0
    )
    apply_bandpass = st.checkbox("Apply 8–30 Hz Bandpass Filter (Mu/Beta)", value=True)
    apply_notch = st.checkbox("Apply 60 Hz Notch Filter", value=False)

# ---------------------------------------------------------
# Top Navigation & Header
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<div class='main-title'>Neuroscript: Mind-to-Text System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>EEG Motor Imagery Classification with Multi-Band Features & Leakage-Free Machine Learning</div>", unsafe_allow_html=True)
with col_h2:
    if is_synthetic:
        st.markdown("<div style='text-align:right;'><span class='badge-synthetic'>SIMULATED DEMO DATA</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right;'><span class='badge-real'>REAL PHYSIONET EEG LOADED</span></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "📁 Dataset & Signal Info",
    "📈 EEG Waveform Visualization (Matplotlib)",
    "🔬 Cross-Task Segment Comparison",
    "🤖 AI Classification",
    "💬 Mind-to-Text Virtual BCI Keyboard"
])

# =========================================================
# SECTION 1: DATASET & SIGNAL INFO
# =========================================================
with tabs[0]:
    st.subheader("1. PhysioNet EEGMMIDB Recording Metadata")

    if raw_eeg is None:
        st.error("No EEG recording is currently loaded.")
        st.markdown(get_manual_download_instructions(subject=subject_id))
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.markdown(f"""
        <div class='metric-card'>
            <div class='card-title'>Recording File</div>
            <div class='card-value'>{metadata.get('filename', 'Unknown')}</div>
        </div>
        """, unsafe_allow_html=True)

        col_m2.markdown(f"""
        <div class='metric-card'>
            <div class='card-title'>Sampling Frequency</div>
            <div class='card-value'>{metadata.get('sampling_frequency_hz', 160.0)} Hz</div>
        </div>
        """, unsafe_allow_html=True)

        col_m3.markdown(f"""
        <div class='metric-card'>
            <div class='card-title'>Duration & Channels</div>
            <div class='card-value'>{metadata.get('duration_seconds', 0.0)}s | {metadata.get('total_channels', 0)} Ch</div>
        </div>
        """, unsafe_allow_html=True)

        col_m4.markdown(f"""
        <div class='metric-card'>
            <div class='card-title'>Total Task Events</div>
            <div class='card-value'>{metadata.get('total_events', 0)} segments</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_inf1, col_inf2 = st.columns([1, 1])

        with col_inf1:
            st.markdown("#### Run Type & Annotation Semantics")
            st.info(f"""
            - **Run Title:** {metadata.get('run_name', 'Unknown')}
            - **Run Category:** `{metadata.get('run_type', 'Unknown')}`
            - **Annotation Protocol Rule:**
              - In Runs 3, 4, 7, 8, 11, 12: `T1` = Left Fist, `T2` = Right Fist.
              - In Runs 5, 6, 9, 10, 13, 14: `T1` = Both Fists, `T2` = Both Feet.
              - In all task runs: `T0` = Rest / Inter-trial baseline.
            """)

            st.markdown("#### Task Event Distribution in This Run")
            df_dist = pd.DataFrame(
                list(metadata.get('event_distribution', {}).items()),
                columns=["Task Label", "Segment Count"]
            )
            st.table(df_dist)

        with col_inf2:
            st.markdown("#### Available EEG Sensor Channels (64-Channel 10-05 System)")
            all_chs = metadata.get("channels_list", [])
            motor_set = set(MOTOR_CHANNELS)
            formatted_chs = [f"**{ch}**" if ch in motor_set else ch for ch in all_chs]
            st.markdown(f"Total **{len(all_chs)} channels** (bolded are primary motor cortex sensors):")
            st.caption(", ".join(formatted_chs[:40]) + f"... and {len(formatted_chs) - 40} more.")

            with st.expander("📋 Full Event Annotations Table", expanded=False):
                st.dataframe(pd.DataFrame(labelled_events), use_container_width=True)

            with st.expander("📥 Manual Download Instructions (If Offline)", expanded=False):
                st.markdown(get_manual_download_instructions(subject=subject_id))

# =========================================================
# SECTION 2: EEG WAVEFORM VISUALIZATION (MATPLOTLIB)
# =========================================================
with tabs[1]:
    st.subheader("2. Real EEG Waveform Visualization (Matplotlib)")
    st.caption("Displays time (seconds) and amplitude (μV) axes with actual dataset event task labels.")

    if raw_eeg is None:
        st.error("No EEG recording available to visualize.")
    else:
        event_options = [
            f"Segment #{ev['index']}: {ev['label']} [{ev['code']}] (Onset: {ev['onset_sec']}s, Dur: {ev['duration_sec']}s)"
            for ev in labelled_events
        ]
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_seg_idx = st.selectbox(
                "Select Labelled Task Segment from Current Recording",
                options=range(len(labelled_events)),
                format_func=lambda i: event_options[i]
            )
        with col_sel2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            st.markdown(f"**Active Channel:** `{selected_ch}` | **Sample Rate:** `{raw_eeg.info['sfreq']} Hz`")

        selected_ev = labelled_events[selected_seg_idx]
        onset = selected_ev["onset_sec"]
        dur = selected_ev["duration_sec"]

        t_raw, s_raw, sfreq = get_single_trial_signal(raw_eeg, onset, dur, selected_ch)

        s_filt = None
        if apply_bandpass:
            raw_filt = preprocess_raw(
                raw_eeg,
                l_freq=8.0,
                h_freq=30.0,
                notch_freq=60.0 if apply_notch else None,
                copy=True
            )
            _, s_filt, _ = get_single_trial_signal(raw_filt, onset, dur, selected_ch)

        fig_matplotlib = plot_single_segment_matplotlib(
            time_array=t_raw,
            raw_uv=s_raw,
            filtered_uv=s_filt,
            channel_name=selected_ch,
            task_label=selected_ev["label"],
            event_code=selected_ev["code"],
            is_synthetic=selected_ev.get("is_synthetic", False)
        )
        st.pyplot(fig_matplotlib, use_container_width=True)
        plt.close(fig_matplotlib)

        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        col_st1.metric("Peak-to-Peak Amplitude", f"{np.ptp(s_raw):.1f} μV")
        col_st2.metric("RMS Signal Power", f"{np.sqrt(np.mean(s_raw**2)):.1f} μV")
        col_st3.metric("Standard Deviation", f"{np.std(s_raw):.1f} μV")
        col_st4.metric("Dataset Event Code", f"{selected_ev['code']}")

# =========================================================
# SECTION 3: CROSS-TASK SEGMENT COMPARISON
# =========================================================
with tabs[2]:
    st.subheader("3. Cross-Task Signal Comparison & Multi-Channel Montage")
    st.markdown("""
    Compare the brain signals of different motor imagery tasks (**Rest**, **Left Fist**, **Right Fist**, **Both Feet**) 
    on the exact same amplitude ($\mu\text{V}$) scale to inspect physiological sensorimotor differences.
    """)

    if raw_eeg is None:
        st.error("No EEG data available.")
    else:
        unique_tasks = {}
        for ev in labelled_events:
            label = ev["label"]
            if label not in unique_tasks:
                unique_tasks[label] = ev

        st.markdown(f"#### Aligned Waveforms Comparison for Channel `{selected_ch}`")

        comparison_dict = {}
        raw_filt_comp = preprocess_raw(raw_eeg, l_freq=8.0, h_freq=30.0, copy=True) if apply_bandpass else None

        for t_label, ev_info in unique_tasks.items():
            t_rel, r_data, _ = get_single_trial_signal(raw_eeg, ev_info["onset_sec"], ev_info["duration_sec"], selected_ch)
            f_data = None
            if raw_filt_comp:
                _, f_data, _ = get_single_trial_signal(raw_filt_comp, ev_info["onset_sec"], ev_info["duration_sec"], selected_ch)
            
            comparison_dict[t_label] = {
                "time": t_rel,
                "raw_uv": r_data,
                "filt_uv": f_data,
                "event_code": ev_info["code"],
                "is_synthetic": ev_info.get("is_synthetic", False)
            }

        fig_comp = plot_segment_comparison_matplotlib(comparison_dict, channel_name=selected_ch)
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

        st.markdown("---")
        st.markdown(f"#### Multi-Channel Motor Cortex Montage ($C_3, C_z, C_4$)")
        
        target_ev = labelled_events[selected_seg_idx]
        montage_channels = [ch for ch in CORE_CHANNELS if ch in raw_eeg.ch_names]
        
        if montage_channels:
            channels_dict = {}
            for ch in montage_channels:
                _, ch_sig, _ = get_single_trial_signal(
                    raw_filt_comp if raw_filt_comp else raw_eeg,
                    target_ev["onset_sec"],
                    target_ev["duration_sec"],
                    ch
                )
                channels_dict[ch] = ch_sig

            fig_montage = plot_multichannel_montage_matplotlib(
                time_array=t_raw,
                channels_data=channels_dict,
                task_label=target_ev["label"],
                event_code=target_ev["code"]
            )
            st.pyplot(fig_montage, use_container_width=True)
            plt.close(fig_montage)

# =========================================================
# SECTION 4: AI CLASSIFICATION (STEP 2 CORE SECTION)
# =========================================================
with tabs[3]:
    st.subheader("4. AI Classification: Multi-Band Features & Machine Learning")

    st.markdown("""
    <div class='leakage-box'>
        <strong>🛡️ Scientific Rigor & Leakage Prevention:</strong><br>
        Models evaluated by randomly shuffling sliding windows across time exhibit falsely high scores (>95%) 
        due to temporal autocorrelation. Here, we evaluate across <strong>Independent Recording Runs</strong> or 
        <strong>Independent Human Participants</strong> without data leakage.
    </div>
    """, unsafe_allow_html=True)

    # Subsection A: Model Training & Persistence Controls
    st.markdown("### Part A: Dataset Preparation & Model Training")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        task_choice = st.selectbox(
            "Target Classification Task",
            options=[
                "2-Class: Left Fist vs Right Fist (Runs 4, 8, 12)",
                "3-Class: Rest vs Left Fist vs Right Fist (Runs 4, 8, 12)",
                "4-Class: Rest vs Left vs Right vs Feet (Runs 4, 6, 8, 10, 12, 14)"
            ],
            index=0
        )
    with col_cfg2:
        split_strategy = st.selectbox(
            "Data Splitting Strategy",
            options=[
                "Run-Based Holdout (Train on Runs 4 & 8, Test on Run 12)",
                "Participant-Independent Holdout (Train on S001, Test on S002)",
                "Stratified Group K-Fold (Grouped by Recording Run)"
            ],
            index=0
        )
    with col_cfg3:
        n_estimators_input = st.slider("Random Forest Trees (n_estimators)", min_value=20, max_value=200, value=100, step=20)

    col_btn_tr, col_btn_save, col_btn_load = st.columns([2, 1, 1])
    with col_btn_tr:
        train_clicked = st.button("🚀 Train & Evaluate Random Forest Classifier", type="primary")
    with col_btn_save:
        save_clicked = st.button("💾 Save Trained Model to Disk")
    with col_btn_load:
        load_clicked = st.button("📂 Load Model from Disk")

    # Handle Save
    if save_clicked:
        if st.session_state.trained_clf_system is not None:
            saved_path = st.session_state.trained_clf_system.save_model(model_dir="models")
            st.success(f"Model and metadata saved successfully to `{saved_path}`!")
        else:
            st.warning("No trained model exists yet to save.")

    # Handle Load
    if load_clicked:
        try:
            loaded_clf = EEGClassifierSystem.load_saved_model(model_dir="models")
            st.session_state.trained_clf_system = loaded_clf
            st.session_state.eval_results = loaded_clf.evaluation_results
            st.session_state.class_names = loaded_clf.class_names_
            st.success("Trained model and preprocessing metadata loaded from disk!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    # Handle Training Execution
    if train_clicked:
        with st.spinner("Extracting multi-band spectral features (Delta, Theta, Alpha, Beta) and training..."):
            try:
                # Gather available recordings
                sub1_avail = list_available_runs(subject=1)
                sub2_avail = list_available_runs(subject=2)

                task_mode_key = "2-class" if "2-Class" in task_choice else ("3-class" if "3-Class" in task_choice else "4-class")
                runs_needed = [4, 8, 12] if task_mode_key != "4-class" else [4, 6, 8, 10, 12, 14]

                recs_to_load = []
                # Load Subject 1 runs
                for r_id in runs_needed:
                    if r_id in sub1_avail:
                        recs_to_load.append({"path": sub1_avail[r_id], "run_id": r_id, "subject": "S001"})

                # If participant-independent splitting requested, also include Subject 2
                if "Participant-Independent" in split_strategy and len(sub2_avail) > 0:
                    for r_id in [4, 8]:
                        if r_id in sub2_avail:
                            recs_to_load.append({"path": sub2_avail[r_id], "run_id": r_id, "subject": "S002"})

                if is_synthetic or len(recs_to_load) == 0:
                    # Synthetic fallback
                    for r_id in runs_needed:
                        syn_raw, syn_evs = generate_synthetic_demo_run(duration_sec=90.0, random_seed=r_id)
                        # Save temp synthetic file or process directly
                        pass

                # Extract cleaned dataset segments and display valid segment counts
                seg_res = prepare_labelled_dataset_segments(
                    recordings_info=recs_to_load,
                    task_mode=task_mode_key,
                    selected_channels=MOTOR_CHANNELS,
                    tmin=0.5,
                    tmax=3.5
                )

                st.session_state.prep_segment_summary = seg_res

                # Extract Multi-Band Features (Delta, Theta, Alpha, Beta, Mean, StdDev)
                X_feat, feature_names = extract_band_power_features(
                    epochs_data=seg_res["X_epochs"],
                    sfreq=seg_res["sfreq"],
                    channel_names=seg_res["channels"],
                    include_statistics=True
                )

                clf = EEGClassifierSystem(n_estimators=n_estimators_input, random_state=42)

                split_meth = "subject_holdout" if "Participant-Independent" in split_strategy else ("run_holdout" if "Holdout" in split_strategy else "group_kfold")
                eval_out = clf.fit_and_evaluate(
                    X=X_feat,
                    y=seg_res["y"],
                    groups=seg_res["groups"],
                    subject_ids=seg_res["subjects"],
                    split_method=split_meth,
                    test_run_id=12 if split_meth == "run_holdout" else None,
                    test_subject_id="S002" if split_meth == "subject_holdout" else None,
                    feature_names=feature_names,
                    class_names=seg_res["class_names"]
                )

                st.session_state.trained_clf_system = clf
                st.session_state.eval_results = eval_out
                st.session_state.class_names = seg_res["class_names"]
                
                # Auto-save model
                clf.save_model(model_dir="models")
                st.success("Random Forest training complete and model saved to disk!")
            except Exception as err:
                st.error(f"Training failed: {err}")

    # Display Dataset Preparation Summary if available
    if "prep_segment_summary" in st.session_state:
        seg_res = st.session_state.prep_segment_summary
        st.markdown("#### Valid Labelled Segments Extracted")
        col_sc1, col_sc2 = st.columns([2, 1])
        with col_sc1:
            df_counts = pd.DataFrame(
                list(seg_res["class_counts"].items()),
                columns=["Task Class", "Valid Segment Count"]
            )
            st.table(df_counts)
        with col_sc2:
            st.metric("Total Segments Inspected", f"{seg_res['total_segments_checked']}")
            st.metric("Excluded Incomplete/Artifact Trials", f"{seg_res['excluded_segments_count']}")

    # Display Evaluation Metrics
    if st.session_state.eval_results is not None:
        res = st.session_state.eval_results
        st.markdown("---")
        st.markdown(f"#### Evaluation Results: `{res.get('method', 'Evaluation Protocol')}`")
        st.caption(res.get("leakage_note", ""))

        col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
        col_ev1.metric("Overall Accuracy", f"{res.get('accuracy', 0.0) * 100:.1f}%")
        col_ev2.metric("Balanced Accuracy", f"{res.get('balanced_accuracy', 0.0) * 100:.1f}%")
        col_ev3.metric("Weighted F1-Score", f"{res.get('f1_score', 0.0):.3f}")
        col_ev4.metric("Test Samples Evaluated", f"{res.get('n_test_samples', 0)}")

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### Confusion Matrix")
            if "confusion_matrix" in res and res["confusion_matrix"] is not None:
                cm = np.array(res["confusion_matrix"])
                c_lbls = st.session_state.get("class_names", [f"C{i}" for i in range(len(cm))])[:len(cm)]
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=c_lbls,
                    y=c_lbls,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted Task", y="Actual Task Label", color="Trials"),
                    title="Confusion Matrix (Honest Predictions)"
                )
                fig_cm.update_layout(height=340, template="plotly_dark")
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_c2:
            st.markdown("##### Class-Wise Performance Metrics")
            if "class_performance" in res and len(res["class_performance"]) > 0:
                st.dataframe(pd.DataFrame(res["class_performance"]), use_container_width=True)
            elif "classification_report" in res:
                st.dataframe(pd.DataFrame(res["classification_report"]).transpose(), use_container_width=True)

    # Subsection B: Interactive Trial Inference & Feature Inspector
    st.markdown("---")
    st.markdown("### Part B: Interactive Trial Inference & Feature Inspector")
    st.caption("Select any EEG trial segment to visualize its waveform, inspect extracted frequency band powers, and compare AI prediction with actual label.")

    if raw_eeg is None:
        st.error("Please load a recording in the sidebar.")
    else:
        col_t_sel1, col_t_sel2 = st.columns([2, 1])
        with col_t_sel1:
            inspect_idx = st.selectbox(
                "Select EEG Trial to Inspect & Classify",
                options=range(len(labelled_events)),
                format_func=lambda i: f"Trial #{labelled_events[i]['index']}: {labelled_events[i]['label']} [{labelled_events[i]['code']}] (t={labelled_events[i]['onset_sec']}s)",
                key="ai_inspect_trial"
            )
        with col_t_sel2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            inspect_ch = st.selectbox("Inspect Sensor Channel", options=MOTOR_CHANNELS, index=MOTOR_CHANNELS.index("C3") if "C3" in MOTOR_CHANNELS else 0)

        target_trial = labelled_events[inspect_idx]
        t_onset = target_trial["onset_sec"]
        t_dur = target_trial["duration_sec"]

        # 1. Waveform Preview
        raw_filt_insp = preprocess_raw(raw_eeg, l_freq=1.0, h_freq=35.0, copy=True)
        t_seg, s_raw_seg, sf = get_single_trial_signal(raw_eeg, t_onset, t_dur, inspect_ch)
        _, s_filt_seg, _ = get_single_trial_signal(raw_filt_insp, t_onset, t_dur, inspect_ch)

        fig_insp_wave = plot_single_segment_matplotlib(
            time_array=t_seg,
            raw_uv=s_raw_seg,
            filtered_uv=s_filt_seg,
            channel_name=inspect_ch,
            task_label=target_trial["label"],
            event_code=target_trial["code"],
            is_synthetic=target_trial.get("is_synthetic", False),
            figsize=(10, 3)
        )
        st.pyplot(fig_insp_wave, use_container_width=True)
        plt.close(fig_insp_wave)

        # 2. Multi-Band Feature Extraction for this trial
        # Extract features across all motor channels for this trial window
        ep_single = create_epochs_from_events(raw_filt_insp, tmin=0.5, tmax=3.5, selected_channels=MOTOR_CHANNELS)
        
        if inspect_idx < len(ep_single):
            single_trial_data = ep_single[inspect_idx].get_data() # (1, n_channels, n_times)
            
            st.markdown(f"#### Extracted Multi-Band Features for Trial #{target_trial['index']}")
            df_trial_features = extract_single_trial_features_table(
                single_epoch_data=single_trial_data,
                sfreq=sf,
                channel_names=MOTOR_CHANNELS
            )

            col_ft1, col_ft2 = st.columns([1, 1])
            with col_ft1:
                st.markdown("##### Channel-Wise Band Powers & Signal Statistics")
                st.dataframe(df_trial_features.set_index("Channel"), height=250, use_container_width=True)

            with col_ft2:
                # Bar chart comparing Delta, Theta, Alpha, Beta across C3, Cz, C4
                df_core = df_trial_features[df_trial_features["Channel"].isin(["C3", "Cz", "C4"])]
                if len(df_core) > 0:
                    fig_band_bar = px.bar(
                        df_core,
                        x="Channel",
                        y=["Delta (0.5–4Hz)", "Theta (4–8Hz)", "Alpha/Mu (8–12Hz)", "Beta (13–30Hz)"],
                        barmode="group",
                        title="Sensorimotor Band Powers (Log PSD)",
                        template="plotly_dark",
                        height=260
                    )
                    st.plotly_chart(fig_band_bar, use_container_width=True)

            # 3. Model Prediction vs Actual Label
            st.markdown("#### Classification Prediction vs Actual Label")
            if st.session_state.trained_clf_system is not None:
                clf_inst = st.session_state.trained_clf_system
                
                # Check for out-of-domain Rest trial on 2-class model
                if len(clf_inst.classes_) == 2 and target_trial["code"] == "T0":
                    st.warning(f"⚠️ **Out-of-Domain Segment:** Trial #{target_trial['index']} is a **Rest baseline (T0)** segment. The active model was trained as a **2-Class classifier** (Left Fist vs Right Fist), so it cannot classify Rest. Please select a Left/Right motor imagery trial, or train a 3-Class model (Rest vs Left vs Right).")
                
                # Extract trial segment with exact sample alignment matching training
                raw_filt_infer = preprocess_raw(raw_eeg, l_freq=1.0, h_freq=35.0, copy=True)
                raw_picked_infer = raw_filt_infer.copy().pick(MOTOR_CHANNELS)
                sf_infer = raw_picked_infer.info['sfreq']
                start_samp = int((target_trial["onset_sec"] + 0.5) * sf_infer)
                stop_samp = start_samp + int(3.0 * sf_infer)
                
                if stop_samp <= raw_picked_infer.n_times:
                    trial_signal_data = raw_picked_infer.get_data()[:, start_samp:stop_samp]
                    pred_c_id, pred_name, conf_score, all_probs = clf_inst.predict_trial_segment(
                        trial_data=trial_signal_data,
                        sfreq=sf_infer,
                        channel_names=MOTOR_CHANNELS
                    )
                    actual_name = target_trial["label"]
                    mapped_cmd = st.session_state.communicator.map_class_to_command(pred_name)

                    col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
                    col_pr1.metric("Predicted Category", pred_name)
                    col_pr2.metric("Mapped BCI Command", mapped_cmd)
                    col_pr3.metric("Calibrated Confidence", f"{conf_score * 100:.1f}%")
                    col_pr4.metric("Actual Dataset Label", actual_name)

                    # Honest comparison note
                    if pred_name == actual_name:
                        st.success(f"✅ Prediction matches dataset ground truth: **{pred_name}** → Mapped Command: **{mapped_cmd}**")
                    else:
                        st.info(f"ℹ️ Prediction: **{pred_name}** | Ground Truth: **{actual_name}** (Calibrated Confidence: {conf_score * 100:.1f}%)")
            else:
                st.info("Train a model above in Part A or click **Load Model** to run predictions.")

# =========================================================
# SECTION 5: MIND-TO-TEXT SMART VIRTUAL BCI KEYBOARD
# =========================================================
with tabs[4]:
    st.subheader("5. EEG-Based Smart Virtual BCI Keyboard & Speech Synthesis")
    st.markdown("""
    Type text using **classified EEG motor imagery states** or **interactive manual testing**.
    Includes real-time cursor navigation, dictionary-based word suggestions, safety debouncing,
    trial segment deduplication, and browser **Text-to-Speech (TTS)**.
    """)

    kb = st.session_state.virtual_keyboard

    # -----------------------------------------------------
    # 1. Mode Selection: Manual Demo vs EEG Prediction Mode
    # -----------------------------------------------------
    col_mode_sel, col_mode_badge = st.columns([2, 1])
    with col_mode_sel:
        active_mode = st.radio(
            "Operating Mode",
            options=["🟢 Manual Demo Mode (Interactive Click & Test)", "🧠 EEG Prediction Mode (Brain Signal Triggered)"],
            horizontal=True,
            help="Choose whether keyboard actions are triggered manually or strictly by real EEG predictions."
        )
    is_eeg_mode = ("EEG Prediction Mode" in active_mode)

    with col_mode_badge:
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        if is_eeg_mode:
            if is_synthetic:
                st.markdown("<span class='provenance-synthetic'>MODE: SIMULATED EEG PREDICTION</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='provenance-real'>MODE: REAL EEG BRAIN-TRIGGERED</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='provenance-manual'>MODE: MANUAL DEMO (NO EEG CLAIM)</span>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 2. Genuine Dataset Verification & Action Mapping Config
    # -----------------------------------------------------
    current_run_classes = metadata.get("classes", []) if metadata else []
    with st.expander("⚙️ BCI Navigation Mode, Dataset Support & Action Mapping", expanded=False):
        col_nav_strat, col_guards = st.columns([1, 1])
        with col_nav_strat:
            st.markdown("##### Navigation Speed & Strategy")
            nav_choice = st.radio(
                "BCI Navigation Strategy",
                options=["⚡ Row-Column 2-Step Scanning (Much Faster BCI)", "🔄 Linear Sequential Scanning"],
                horizontal=True,
                index=0 if kb.nav_mode == "ROW_COLUMN" else 1,
                help="Row-Column scanning lets you reach any letter in 2-4 steps instead of up to 30 steps."
            )
            kb.nav_mode = "ROW_COLUMN" if "Row-Column" in nav_choice else "LINEAR"

            st.markdown("##### Genuine Dataset Support Check")
            st.markdown(f"**Selected Recording:** `Run {selected_run_id}` ({metadata.get('run_name', 'Current')})")
            st.markdown(f"**PhysioNet Event Classes in this Run:** `{current_run_classes}`")
            if "Both Feet" not in str(current_run_classes) and "Both Fists" not in str(current_run_classes):
                st.info("ℹ️ **Run Notice:** This recording contains **Left Fist**, **Right Fist**, and **Rest** events. Tasks like *Both Feet* or *Both Fists* occur in Runs 5, 6, 9, 10, 13, 14. Left/Right imagery drives cursor navigation.")
            else:
                st.success("✅ **Run Notice:** This recording contains multi-limb tasks suitable for full 4-class navigation and selection.")

        with col_guards:
            st.markdown("##### Safety Guards & Thresholds")
            new_conf_thresh = st.slider(
                "Confidence Threshold (Safety Gate)",
                min_value=0.50, max_value=0.95,
                value=float(kb.confidence_threshold),
                step=0.05,
                help="EEG predictions below this threshold will be blocked to avoid false typing."
            )
            kb.confidence_threshold = new_conf_thresh

            new_cooldown = st.slider(
                "Action Cooldown (Debounce)",
                min_value=0.5, max_value=3.0,
                value=float(kb.cooldown_seconds),
                step=0.25,
                help="Minimum seconds between accepted actions to prevent accidental repeated typing."
            )
            kb.cooldown_seconds = new_cooldown

        st.markdown("---")
        st.markdown("##### EEG Task to Keyboard Action Mapping")
        action_opts = [
            ACTION_MOVE_LEFT,
            ACTION_MOVE_RIGHT,
            ACTION_SELECT,
            ACTION_SPACE,
            ACTION_NEXT_ROW,
            ACTION_NO_ACTION
        ]
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            map_left = st.selectbox("Left Fist Imagery", options=action_opts, index=action_opts.index(kb.mapping.get("Left Fist Imagery", ACTION_MOVE_LEFT)))
            kb.mapping["Left Fist Imagery"] = map_left
            kb.mapping["Left Fist (Exec)"] = map_left
        with col_m2:
            map_right = st.selectbox("Right Fist Imagery", options=action_opts, index=action_opts.index(kb.mapping.get("Right Fist Imagery", ACTION_MOVE_RIGHT)))
            kb.mapping["Right Fist Imagery"] = map_right
            kb.mapping["Right Fist (Exec)"] = map_right
        with col_m3:
            map_fists = st.selectbox("Both Fists Imagery", options=action_opts, index=action_opts.index(kb.mapping.get("Both Fists Imagery", ACTION_SELECT)))
            kb.mapping["Both Fists Imagery"] = map_fists
            kb.mapping["Both Fists (Exec)"] = map_fists
        with col_m4:
            map_feet = st.selectbox("Both Feet Imagery", options=action_opts, index=action_opts.index(kb.mapping.get("Both Feet Imagery", ACTION_SPACE)))
            kb.mapping["Both Feet Imagery"] = map_feet
            kb.mapping["Both Feet (Exec)"] = map_feet
        with col_m5:
            map_rest = st.selectbox("Rest Baseline", options=action_opts, index=action_opts.index(kb.mapping.get("Rest", ACTION_NO_ACTION)))
            kb.mapping["Rest"] = map_rest

    # -----------------------------------------------------
    # 3. Prominent Accumulated Text Output & Quick Phrases
    # -----------------------------------------------------
    st.markdown("#### Composed Sentence Output")
    text_val = kb.accumulated_text
    char_count = len(text_val)
    word_count = len(text_val.strip().split()) if text_val.strip() else 0

    st.markdown(f"""
    <div class='keyboard-text-display'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
            <span style='color:#94a3b8; font-size:0.8rem; font-weight:700; text-transform:uppercase;'>Composed Sentence Buffer</span>
            <span style='color:#64748b; font-size:0.8rem;'>{word_count} words | {char_count} chars</span>
        </div>
        <div class='text-output-string'>
            {text_val if text_val else "<span style='color:#475569;'>[Start typing via Virtual Keyboard, Quick Phrases, or EEG below...]</span>"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3A. Quick Assistive Phrases Bar (Instant One-Touch Entry)
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#38bdf8; margin-bottom:4px;'>⚡ Quick Assistive Phrases (One-Click Instant Entry):</div>", unsafe_allow_html=True)
    phrase_cols = st.columns(len(QUICK_PHRASES))
    for p_idx, phrase in enumerate(QUICK_PHRASES):
        with phrase_cols[p_idx]:
            if st.button(f"💬 {phrase}", key=f"quick_phrase_{p_idx}", use_container_width=True):
                src = "Manual Click" if not is_eeg_mode else ("Synthetic Demo" if is_synthetic else "Real EEG Assisted")
                kb.apply_phrase(phrase, source=src)
                st.rerun()

    # 3B. Word Prediction and Autocomplete (Top 3-5 Matches)
    suggestions = get_word_suggestions(text_val, max_suggestions=5)
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    col_sug_lbl, col_sug_btns = st.columns([1, 4])
    with col_sug_lbl:
        st.markdown("<div style='padding-top:6px; font-size:0.85rem; font-weight:700; color:#a78bfa;'>💡 Word Autocomplete:</div>", unsafe_allow_html=True)
    with col_sug_btns:
        cols_sug = st.columns(len(suggestions) if suggestions else 1)
        if suggestions:
            for idx, sug_word in enumerate(suggestions):
                with cols_sug[idx]:
                    if st.button(f"✨ {sug_word}", key=f"sug_btn_{idx}", use_container_width=True):
                        source_label = "Manual Click" if not is_eeg_mode else ("Synthetic Demo" if is_synthetic else "Real EEG Assisted")
                        kb.apply_word_suggestion(sug_word, source=source_label)
                        st.rerun()
        else:
            st.caption("No suggestions available.")

    # 3C. Sentence Action Controls: Delete, Clear, Space, and Speak/Copy TTS
    col_act1, col_act2, col_act3, col_act4 = st.columns([1, 1, 1.5, 3])
    with col_act1:
        if st.button("⌫ Delete", use_container_width=True):
            kb.backspace(source="Manual Click" if not is_eeg_mode else "EEG Mode Manual")
            st.rerun()
    with col_act2:
        if st.button("🗑️ Clear", use_container_width=True):
            kb.clear(source="Manual Click" if not is_eeg_mode else "EEG Mode Manual")
            st.rerun()
    with col_act3:
        if st.button("␣ Space", use_container_width=True):
            kb.type_letter_direct("SPACE", source="Manual Click" if not is_eeg_mode else "EEG Mode Manual")
            st.rerun()
    with col_act4:
        tts_html = generate_tts_html(text_val if text_val.strip() else "Buffer is empty", autoplay=False)
        components.html(tts_html, height=45)

    # -----------------------------------------------------
    # 4. Interactive QWERTY Keyboard with Row-Column Guidance
    # -----------------------------------------------------
    st.markdown("---")
    if kb.nav_mode == "ROW_COLUMN":
        if kb.nav_stage == "ROW":
            st.info(f"📍 **Navigation Stage 1 (Select Row):** Currently targeting **Row {kb.active_row_idx + 1}** (`{' '.join(KEYBOARD_ROWS[kb.active_row_idx])}`). Trigger **Select (↵)** or click a key to select this row, or **Left/Right** to scan rows.")
        else:
            st.success(f"📍 **Navigation Stage 2 (Select Letter):** Inside **Row {kb.active_row_idx + 1}**, active cursor is on key `[ {kb.current_key} ]`. Trigger **Select (↵)** to type it!")
    else:
        st.markdown(f"#### Interactive On-Screen QWERTY Keyboard (Active Cursor: `[ {kb.current_key} ]`)")

    # Render QWERTY Keyboard Rows with Visual Highlighting
    for row_idx, row_keys in enumerate(KEYBOARD_ROWS):
        is_row_targeted = (kb.nav_mode == "ROW_COLUMN" and kb.active_row_idx == row_idx)
        key_cols = st.columns(len(row_keys))
        for k_idx, key in enumerate(row_keys):
            # In Row-Column mode, highlight based on stage
            if kb.nav_mode == "ROW_COLUMN":
                if kb.nav_stage == "ROW":
                    is_active = (is_row_targeted and k_idx == 0)
                else:
                    is_active = (is_row_targeted and k_idx == kb.active_col_idx)
            else:
                is_active = (key == kb.current_key)

            with key_cols[k_idx]:
                btn_label = f"[{key}]" if is_active else key
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(btn_label, key=f"kb_key_{row_idx}_{k_idx}", type=btn_type, use_container_width=True):
                    if not is_eeg_mode:
                        if key in ["SPACE", "DELETE", "CLEAR"]:
                            kb.type_letter_direct(key, source="Manual Click")
                        elif key == "SELECT":
                            kb.select_current_key(source="Manual Click")
                        else:
                            kb.type_letter_direct(key, source="Manual Click")
                    else:
                        kb.set_highlighted_key(key)
                    st.rerun()

    # Manual Navigation Pad for Demonstration in Manual Mode
    if not is_eeg_mode:
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        col_nav_info, col_nl, col_nr, col_nsel, col_nbk = st.columns([2, 1, 1, 1, 1])
        with col_nav_info:
            nav_desc = "Row scanning" if (kb.nav_mode == "ROW_COLUMN" and kb.nav_stage == "ROW") else "Letter scanning"
            st.caption(f"🎮 **Manual Navigation Demo ({nav_desc}):**")
        with col_nl:
            if st.button("← Previous", use_container_width=True):
                kb.navigate_left()
                st.rerun()
        with col_nr:
            if st.button("Next →", use_container_width=True):
                kb.navigate_right()
                st.rerun()
        with col_nsel:
            sel_btn_lbl = "↵ Select Row" if (kb.nav_mode == "ROW_COLUMN" and kb.nav_stage == "ROW") else "↵ Select Key"
            if st.button(sel_btn_lbl, type="primary", use_container_width=True):
                kb.select_current_key(source="Manual Click")
                st.rerun()
        with col_nbk:
            if kb.nav_mode == "ROW_COLUMN" and kb.nav_stage == "COLUMN":
                if st.button("↩️ Back to Rows", use_container_width=True):
                    kb.cancel_to_row_stage()
                    st.rerun()
            else:
                if st.button("↓ Next Row", use_container_width=True):
                    kb.navigate_next_row()
                    st.rerun()

    # -----------------------------------------------------
    # 5. EEG Prediction Mode: Brain Signal Ingestion & Dispatch
    # -----------------------------------------------------
    if is_eeg_mode:
        st.markdown("---")
        st.markdown("### 🧠 EEG Brain Signal Dispatcher (Real Data Only)")
        st.caption("Select a real EEG segment, evaluate the calibrated AI model, and dispatch the predicted motor imagery action to the keyboard.")

        if st.session_state.trained_clf_system is None:
            st.warning("⚠️ No trained AI classifier loaded. Train one in **Tab 4 (AI Classification)** or click below to load the trained model:")
            if st.button("📂 Load Saved Calibrated Model from Disk", type="primary"):
                loaded_clf = get_cached_bci_classifier()
                if loaded_clf is not None:
                    st.session_state.trained_clf_system = loaded_clf
                    st.session_state.class_names = loaded_clf.class_names_
                    st.success("Loaded model successfully!")
                    st.rerun()
                else:
                    st.error("No saved model found in models/ directory. Train one in Tab 4 first.")
        elif raw_eeg is None or len(labelled_events) == 0:
            st.error("No EEG recording is loaded. Please select a recording in the sidebar.")
        else:
            clf_inst = st.session_state.trained_clf_system

            # Select EEG trial
            col_tr_sel1, col_tr_sel2 = st.columns([3, 1])
            with col_tr_sel1:
                eeg_trial_idx = st.selectbox(
                    "Select Genuine EEG Trial Segment",
                    options=range(len(labelled_events)),
                    format_func=lambda i: f"Trial #{labelled_events[i]['index']}: {labelled_events[i]['label']} [{labelled_events[i]['code']}] (t={labelled_events[i]['onset_sec']:.2f}s, dur={labelled_events[i]['duration_sec']:.1f}s)",
                    key="eeg_kb_trial_select"
                )
            with col_tr_sel2:
                st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                if st.button("⏭️ Next Trial", use_container_width=True):
                    next_idx = (eeg_trial_idx + 1) % len(labelled_events)
                    st.session_state.eeg_kb_trial_select = next_idx
                    st.rerun()

            target_eeg_ev = labelled_events[eeg_trial_idx]
            seg_unique_id = f"R{selected_run_id}_T{target_eeg_ev['index']}_{target_eeg_ev['code']}_{target_eeg_ev['onset_sec']:.2f}"

            # Fast cached signal slicing: Avoid re-filtering entire recording on every interaction
            sig_cache_key = f"cached_filt_raw_r{selected_run_id}_{is_synthetic}"
            if sig_cache_key not in st.session_state:
                raw_filt_bci = preprocess_raw(raw_eeg, l_freq=1.0, h_freq=35.0, copy=True)
                st.session_state[sig_cache_key] = raw_filt_bci.copy().pick(MOTOR_CHANNELS)

            raw_picked_bci = st.session_state[sig_cache_key]
            sf_bci = raw_picked_bci.info['sfreq']
            start_samp = int(round((target_eeg_ev["onset_sec"] + 0.5) * sf_bci))
            stop_samp = start_samp + int(round(3.0 * sf_bci))
            stop_samp = min(stop_samp, raw_picked_bci.n_times)

            trial_sig_data = raw_picked_bci.get_data()[:, start_samp:stop_samp]

            # Model prediction on genuine segment
            pred_id, pred_label, conf_score, probs = clf_inst.predict_trial_segment(
                trial_data=trial_sig_data,
                sfreq=sf_bci,
                channel_names=MOTOR_CHANNELS
            )

            resolved_action = kb.mapping.get(pred_label, ACTION_NO_ACTION)
            action_desc = ACTION_DESCRIPTIONS.get(resolved_action, resolved_action)

            # Trial Waveform & Prediction Status Display
            col_prev_wave, col_prev_meta = st.columns([1, 1])
            with col_prev_wave:
                st.markdown(f"**EEG Waveform Segment (Channel C3, t={target_eeg_ev['onset_sec']:.2f}s):**")
                t_plot, s_plot, _ = get_single_trial_signal(raw_picked_bci, target_eeg_ev["onset_sec"], target_eeg_ev["duration_sec"], "C3")
                fig_mini_wave = plot_single_segment_matplotlib(
                    time_array=t_plot,
                    raw_uv=s_plot,
                    filtered_uv=None,
                    channel_name="C3",
                    task_label=target_eeg_ev["label"],
                    event_code=target_eeg_ev["code"],
                    is_synthetic=target_eeg_ev.get("is_synthetic", False),
                    figsize=(8, 2.5)
                )
                st.pyplot(fig_mini_wave, use_container_width=True)
                plt.close(fig_mini_wave)

            with col_prev_meta:
                st.markdown("**Genuine EEG Segment & Model Prediction Analysis:**")
                
                col_mp1, col_mp2 = st.columns(2)
                col_mp1.metric("Actual Dataset Label", f"{target_eeg_ev['label']} [{target_eeg_ev['code']}]")
                col_mp2.metric("Model Predicted Class", pred_label)

                col_mp3, col_mp4 = st.columns(2)
                col_mp3.metric("Calibrated Confidence", f"{conf_score * 100:.1f}%")
                col_mp4.metric("Triggered Keyboard Action", action_desc)

                # Deduplication and Safety Check Alert
                is_duplicate = (seg_unique_id == kb.last_processed_segment_id)
                meets_threshold = (conf_score >= kb.confidence_threshold)
                
                if is_duplicate:
                    st.warning("⚠️ **Deduplication Guard:** This trial segment was already processed. Advance to next trial to avoid duplicate action.")
                elif not meets_threshold:
                    st.info(f"ℹ️ **Confidence Guard:** Confidence ({conf_score*100:.1f}%) is below your threshold ({kb.confidence_threshold*100:.1f}%).")
                else:
                    st.success("🟢 **Safety Check Passed:** Trial segment is fresh and confidence exceeds threshold.")

            # Dispatch Trigger Button
            col_disp_btn, col_disp_info = st.columns([1, 2])
            with col_disp_btn:
                dispatch_clicked = st.button("⚡ Dispatch EEG Prediction to Keyboard", type="primary", use_container_width=True)

            if dispatch_clicked:
                res_entry = kb.process_eeg_prediction(
                    predicted_class=pred_label,
                    confidence=conf_score,
                    segment_id=seg_unique_id,
                    mapping=kb.mapping,
                    actual_label=target_eeg_ev["label"],
                    is_synthetic=is_synthetic
                )

                if res_entry["status"] == "EXECUTED":
                    st.toast(f"✅ Action Executed: {res_entry['detail']}", icon="🧠")
                    st.success(f"**Dispatched Action:** `{res_entry['action']}` — {res_entry['detail']}. Text: **\"{kb.accumulated_text}\"**")
                    st.rerun()
                elif res_entry["status"] == "DUPLICATE_IGNORED":
                    st.error(f"🛑 Duplicate Blocked: {res_entry['detail']}")
                elif res_entry["status"] == "LOW_CONFIDENCE_IGNORED":
                    st.warning(f"🛑 Confidence Blocked: {res_entry['detail']}")
                elif res_entry["status"] == "COOLDOWN_IGNORED":
                    st.warning(f"🛑 Cooldown Blocked: {res_entry['detail']}")

    # -----------------------------------------------------
    # 6. Action Audit Log & Session History
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("#### Session Action History & Provenance Log")
    if kb.action_history:
        df_kb_hist = pd.DataFrame(kb.action_history)
        st.dataframe(df_kb_hist, use_container_width=True)
        if st.button("Clear Keyboard Audit Log"):
            kb.action_history.clear()
            st.rerun()
    else:
        st.info("No actions logged yet in this session.")
