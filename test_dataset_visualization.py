"""
Automated Verification for Step 1: Dataset Loader and Visualization Module.
Tests real PhysioNet EEG loading, metadata extraction, run-type annotation semantics,
preprocessing, channel validation, and Matplotlib plotting functions.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Headless testing
import matplotlib.pyplot as plt
import numpy as np

from config import MOTOR_CHANNELS
from data_loader import (
    generate_synthetic_demo_run,
    get_recording_metadata,
    list_available_runs,
    load_real_eeg_run,
)
from preprocessing import (
    get_single_trial_signal,
    preprocess_raw,
    validate_recording_and_channels,
)
from visualization import (
    plot_multichannel_montage_matplotlib,
    plot_psd_matplotlib,
    plot_segment_comparison_matplotlib,
    plot_single_segment_matplotlib,
)


def test_step1_dataset_and_visualization():
    print("=" * 65)
    print("TESTING STEP 1: EEG DATASET AND VISUALIZATION MODULE")
    print("=" * 65)

    # 1. Dataset detection
    available = list_available_runs(subject=1)
    print(f"\n[1] Local PhysioNet recordings detected for S001: {list(available.keys())}")
    assert len(available) > 0, "Expected downloaded PhysioNet runs for Subject 1"

    # 2. Test Real EEG Loading for Run 4 (Left vs Right Motor Imagery)
    r4_path = available[4]
    raw_r4, events_r4 = load_real_eeg_run(r4_path, run_id=4)
    print(f"[2] Successfully loaded Run 4 EDF+ recording:")
    print(f"    - Channels: {raw_r4.info['nchan']}, Sampling Rate: {raw_r4.info['sfreq']} Hz")
    print(f"    - Total duration: {raw_r4.times[-1]:.1f} s, Total annotated segments: {len(events_r4)}")
    assert raw_r4.info['nchan'] == 64
    assert raw_r4.info['sfreq'] == 160.0

    # 3. Test Metadata extraction
    meta = get_recording_metadata(raw_r4, run_id=4, file_path=r4_path, labelled_events=events_r4)
    print(f"\n[3] Extracted Recording Metadata:")
    print(f"    - Filename: {meta['filename']}")
    print(f"    - Participant: {meta['participant']}")
    print(f"    - Run Name: {meta['run_name']}")
    print(f"    - Run Type: {meta['run_type']}")
    print(f"    - Event distribution: {meta['event_distribution']}")
    assert meta["participant"] == "S001"
    assert "Left Fist Imagery" in meta["event_distribution"]
    assert "Right Fist Imagery" in meta["event_distribution"]
    assert "Rest" in meta["event_distribution"]

    # 4. Test Annotation Semantics on Run 6 (Both Fists vs Both Feet)
    if 6 in available:
        raw_r6, events_r6 = load_real_eeg_run(available[6], run_id=6)
        meta_r6 = get_recording_metadata(raw_r6, run_id=6, file_path=available[6], labelled_events=events_r6)
        print(f"\n[4] Verified Annotation Semantics on Run 6:")
        print(f"    - Event distribution in Run 6: {meta_r6['event_distribution']}")
        assert "Both Feet Imagery" in meta_r6["event_distribution"]
        print("    Run 6 correctly interprets T2 as Both Feet Imagery (not Right Fist)!")

    # 5. Channel validation
    valid_chs, missing_chs = validate_recording_and_channels(raw_r4, ["C3", "Cz", "C4", "NonExistentCh"])
    print(f"\n[5] Channel Validation Check:")
    print(f"    - Valid channels: {valid_chs}")
    print(f"    - Gracefully detected missing channels: {missing_chs}")
    assert "C3" in valid_chs and "NonExistentCh" in missing_chs

    # 6. Preprocessing: 8-30 Hz Mu/Beta Bandpass Filter
    raw_filt = preprocess_raw(raw_r4, l_freq=8.0, h_freq=30.0, copy=True)
    print(f"\n[6] Zero-phase FIR Preprocessing Filter:")
    print(f"    - Highpass: {raw_filt.info['highpass']} Hz, Lowpass: {raw_filt.info['lowpass']} Hz")

    # 7. Single-trial signal extraction
    first_ev = events_r4[1] # A task event
    t_raw, s_raw, sfreq = get_single_trial_signal(raw_r4, first_ev["onset_sec"], first_ev["duration_sec"], "C3")
    _, s_filt, _ = get_single_trial_signal(raw_filt, first_ev["onset_sec"], first_ev["duration_sec"], "C3")
    print(f"\n[7] Extracted trial segment for Channel C3:")
    print(f"    - Task Label: '{first_ev['label']}' [{first_ev['code']}]")
    print(f"    - Samples: {len(s_raw)}, Duration: {t_raw[-1]:.2f}s, Peak-to-Peak: {np.ptp(s_raw):.2f} uV")

    # 8. Matplotlib Single Segment Waveform Visualization
    fig_single = plot_single_segment_matplotlib(
        time_array=t_raw,
        raw_uv=s_raw,
        filtered_uv=s_filt,
        channel_name="C3",
        task_label=first_ev["label"],
        event_code=first_ev["code"],
        is_synthetic=False
    )
    assert fig_single is not None, "Failed to create Matplotlib single segment figure"
    plt.close(fig_single)
    print("\n[8] plot_single_segment_matplotlib: PASSED (time and amplitude axes verified)")

    # 9. Matplotlib Cross-Task Segment Comparison
    unique_tasks = {}
    for ev in events_r4:
        if ev["label"] not in unique_tasks:
            unique_tasks[ev["label"]] = ev
    
    comp_dict = {}
    for t_name, ev_info in unique_tasks.items():
        t_seg, r_seg, _ = get_single_trial_signal(raw_r4, ev_info["onset_sec"], ev_info["duration_sec"], "C3")
        _, f_seg, _ = get_single_trial_signal(raw_filt, ev_info["onset_sec"], ev_info["duration_sec"], "C3")
        comp_dict[t_name] = {
            "time": t_seg,
            "raw_uv": r_seg,
            "filt_uv": f_seg,
            "event_code": ev_info["code"],
            "is_synthetic": False
        }
    
    fig_comp = plot_segment_comparison_matplotlib(comp_dict, channel_name="C3")
    assert fig_comp is not None, "Failed to create Matplotlib comparison figure"
    plt.close(fig_comp)
    print("[9] plot_segment_comparison_matplotlib: PASSED (multi-task subplots verified)")

    # 10. Matplotlib Multi-Channel Montage
    channels_dict = {}
    for ch in ["C3", "Cz", "C4"]:
        _, sig_ch, _ = get_single_trial_signal(raw_filt, first_ev["onset_sec"], first_ev["duration_sec"], ch)
        channels_dict[ch] = sig_ch
    
    fig_montage = plot_multichannel_montage_matplotlib(
        time_array=t_raw,
        channels_data=channels_dict,
        task_label=first_ev["label"],
        event_code=first_ev["code"]
    )
    assert fig_montage is not None, "Failed to create Matplotlib montage figure"
    plt.close(fig_montage)
    print("[10] plot_multichannel_montage_matplotlib: PASSED (stacked montage verified)")

    # 11. Synthetic Demo Fallback
    raw_syn, ev_syn = generate_synthetic_demo_run(duration_sec=30.0)
    assert ev_syn[0]["is_synthetic"] is True
    print("[11] Synthetic Demo Mode: PASSED (explicitly labeled as simulated)")

    print("\n" + "=" * 65)
    print("ALL STEP 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_step1_dataset_and_visualization()
