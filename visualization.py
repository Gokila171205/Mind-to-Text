"""
EEG Waveform and Spectral Visualization Module using Matplotlib.
Implements single-segment waveform plotting, multi-task segment comparison,
multi-channel montage plotting, and Power Spectral Density (PSD) figures.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Apply clean, modern aesthetic styling to Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


def plot_single_segment_matplotlib(
    time_array: np.ndarray,
    raw_uv: np.ndarray,
    filtered_uv: Optional[np.ndarray] = None,
    channel_name: str = "C3",
    task_label: str = "Rest",
    event_code: str = "T0",
    is_synthetic: bool = False,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Plots a single EEG trial segment with explicit time (s) and amplitude (uV) axes.
    Displays the dataset's actual task label and event code prominently beside the waveform.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Background and grid
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')

    # Plot raw waveform
    ax.plot(
        time_array,
        raw_uv,
        color='#94a3b8',
        linewidth=1.0,
        alpha=0.75,
        label=f"Raw Signal ({channel_name})"
    )

    # Plot filtered waveform if provided
    if filtered_uv is not None:
        ax.plot(
            time_array,
            filtered_uv,
            color='#38bdf8',
            linewidth=1.8,
            label="Filtered (8–30 Hz Mu/Beta)"
        )

    # Labels and Titles
    ax.set_xlabel("Time (seconds relative to cue onset)", fontsize=11, color='#e2e8f0', fontweight='500')
    ax.set_ylabel("Amplitude (μV)", fontsize=11, color='#e2e8f0', fontweight='500')
    ax.tick_params(colors='#cbd5e1', labelsize=10)

    # Prominent title and label badge
    data_badge = "[SIMULATED / SYNTHETIC]" if is_synthetic else "[REAL PHYSIONET EEG]"
    badge_color = "#f87171" if is_synthetic else "#34d399"
    
    title_text = f"Channel: {channel_name} | Dataset Label: {task_label} (Code: {event_code})"
    ax.set_title(title_text, fontsize=12, color='#f8fafc', fontweight='bold', pad=12)

    # Add text banner on the top-right
    ax.text(
        0.98, 0.92,
        data_badge,
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        color=badge_color,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor=badge_color, alpha=0.85)
    )

    ax.legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=9)
    fig.tight_layout()
    return fig


def plot_segment_comparison_matplotlib(
    segments_dict: Dict[str, Dict],
    channel_name: str = "C3",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Displays multiple labelled task segments (e.g. Rest vs Left vs Right vs Feet)
    in aligned subplots with identical amplitude limits for direct visual comparison.
    
    segments_dict structure:
        {
            "Task Label": {
                "time": np.ndarray,
                "raw_uv": np.ndarray,
                "filt_uv": np.ndarray,
                "event_code": str,
                "is_synthetic": bool
            },
            ...
        }
    """
    n_tasks = len(segments_dict)
    if n_tasks == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No segments available for comparison", ha='center', va='center')
        return fig

    fig, axes = plt.subplots(n_tasks, 1, figsize=figsize, sharex=True, dpi=100)
    fig.patch.set_facecolor('#0f172a')

    if n_tasks == 1:
        axes = [axes]

    # Calculate global amplitude range for synchronized y-axes
    all_filt = []
    for info in segments_dict.values():
        if info.get("filt_uv") is not None:
            all_filt.extend(info["filt_uv"])
        else:
            all_filt.extend(info["raw_uv"])
    
    if len(all_filt) > 0:
        max_abs = max(10.0, np.percentile(np.abs(all_filt), 99) * 1.2)
        ylim = (-max_abs, max_abs)
    else:
        ylim = (-50, 50)

    task_colors = {
        "Rest": "#10b981",
        "Left Fist Imagery": "#3b82f6",
        "Right Fist Imagery": "#8b5cf6",
        "Both Feet Imagery": "#ec4899",
        "Both Fists Imagery": "#f59e0b"
    }

    for idx, (label, data) in enumerate(segments_dict.items()):
        ax = axes[idx]
        ax.set_facecolor('#1e293b')
        ax.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')
        
        t = data["time"]
        r_uv = data["raw_uv"]
        f_uv = data.get("filt_uv")
        code = data.get("event_code", "T0")
        is_syn = data.get("is_synthetic", False)

        # Plot raw in subtle gray
        ax.plot(t, r_uv, color='#64748b', linewidth=0.8, alpha=0.6, label="Raw")
        
        # Color trace according to task category
        line_color = task_colors.get(label, "#38bdf8")
        if f_uv is not None:
            ax.plot(t, f_uv, color=line_color, linewidth=1.6, label="8–30 Hz Filtered")

        ax.set_ylim(ylim)
        ax.tick_params(colors='#cbd5e1', labelsize=9)
        ax.set_ylabel("Amplitude (μV)", fontsize=9, color='#e2e8f0')

        # Annotation badge inside subplot
        badge = f"Task: {label} [{code}]"
        if is_syn:
            badge += " (Simulated)"
        ax.text(
            0.02, 0.88,
            badge,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            color='#ffffff',
            va='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=line_color, edgecolor='none', alpha=0.8)
        )

        if idx == 0:
            ax.set_title(f"Channel {channel_name} | Cross-Task Signal Comparison (Synchronized μV Scale)",
                         fontsize=12, color='#f8fafc', fontweight='bold', pad=10)
        if idx == n_tasks - 1:
            ax.set_xlabel("Time (seconds relative to trial onset)", fontsize=10, color='#e2e8f0', fontweight='500')

    fig.tight_layout()
    return fig


def plot_multichannel_montage_matplotlib(
    time_array: np.ndarray,
    channels_data: Dict[str, np.ndarray],
    task_label: str = "Left Fist Imagery",
    event_code: str = "T1",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Displays a stacked multi-channel EEG trace (e.g., C3, Cz, C4) for a single trial segment.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')

    ch_names = list(channels_data.keys())
    offset_step = 60.0 # microvolts spacing between channels

    for i, ch in enumerate(ch_names):
        sig = channels_data[ch]
        offset = (len(ch_names) - 1 - i) * offset_step
        ax.plot(time_array, sig + offset, linewidth=1.4, label=ch)
        ax.text(time_array[0] - 0.05, offset, ch, color='#f8fafc', fontweight='bold',
                ha='right', va='center', fontsize=10)

    ax.set_xlabel("Time (seconds relative to cue onset)", fontsize=11, color='#e2e8f0')
    ax.set_yticks([(len(ch_names) - 1 - i) * offset_step for i in range(len(ch_names))])
    ax.set_yticklabels(ch_names, color='#cbd5e1', fontsize=10)
    ax.set_ylabel("EEG Channels (Stacked μV Trace)", fontsize=11, color='#e2e8f0')
    ax.tick_params(colors='#cbd5e1')
    ax.set_title(f"Multi-Channel Motor Cortex Montage | Task: {task_label} [{event_code}]",
                 fontsize=12, color='#f8fafc', fontweight='bold', pad=12)

    fig.tight_layout()
    return fig


def plot_psd_matplotlib(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    channel_name: str = "C3",
    task_label: str = "Rest",
    figsize: Tuple[int, int] = (8, 4)
) -> plt.Figure:
    """
    Plots the Welch Power Spectral Density (PSD) using Matplotlib,
    highlighting the sensorimotor Mu (8-12 Hz) and Beta (13-30 Hz) rhythm frequency bands.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')

    ax.plot(freqs, psd_db, color='#a855f7', linewidth=2.0, label=f"{channel_name} Welch PSD")

    # Highlight Mu band (8-12 Hz)
    ax.axvspan(8.0, 12.0, color='#3b82f6', alpha=0.22, label="Mu Band (8–12 Hz)")
    # Highlight Beta band (13-30 Hz)
    ax.axvspan(13.0, 30.0, color='#ec4899', alpha=0.18, label="Beta Band (13–30 Hz)")

    ax.set_xlabel("Frequency (Hz)", fontsize=10, color='#e2e8f0')
    ax.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=10, color='#e2e8f0')
    ax.set_title(f"Power Spectral Density | Channel {channel_name} ({task_label})",
                 fontsize=11, color='#f8fafc', fontweight='bold', pad=10)
    ax.tick_params(colors='#cbd5e1', labelsize=9)
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=9)
    fig.tight_layout()
    return fig
