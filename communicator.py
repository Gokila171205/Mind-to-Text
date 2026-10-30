"""
Mind-to-Text Communication Module.
Translates motor imagery classifications into application-defined communication commands.
Maintains history, sentence buffer, and provides browser Text-to-Speech (TTS) integration.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from config import DEFAULT_MAPPING, SCIENTIFIC_DISCLAIMER
from keyboard import DEFAULT_BCI_KEYBOARD_MAPPING, VirtualBCIKeyboard


class MindToTextSystem:
    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping = mapping.copy() if mapping else DEFAULT_MAPPING.copy()
        self.sentence_tokens: List[str] = []
        self.history: List[Dict] = []

    def update_mapping(self, new_mapping: Dict[str, str]):
        """Updates the application-defined communication dictionary."""
        self.mapping = new_mapping.copy()

    def map_class_to_command(self, class_label: str) -> str:
        """
        Maps a physiological classification label (e.g. 'Left Fist Imagery')
        to its user-configured communication token (e.g. 'LEFT').
        """
        if class_label in self.mapping:
            return self.mapping[class_label]

        # Normalize label by checking base form
        base_label = class_label.replace(" Imagery", "").replace(" (Exec)", "").replace("[SYNTHETIC DEMO]", "").strip()
        if base_label in self.mapping:
            return self.mapping[base_label]

        return self.mapping.get(class_label, class_label.upper())

    def record_prediction(
        self,
        predicted_class: str,
        confidence: float,
        actual_class: Optional[str] = None,
        source: str = "PhysioNet Real EEG",
        run_id: Optional[int] = None,
        trial_id: Optional[int] = None,
        auto_append: bool = True
    ) -> Dict:
        """
        Records a single BCI classification event into session history
        and optionally appends the command to the current sentence buffer.
        """
        command = self.map_class_to_command(predicted_class)
        timestamp = datetime.now().strftime("%H:%M:%S")

        entry = {
            "timestamp": timestamp,
            "predicted_task": predicted_class,
            "mapped_command": command,
            "calibrated_confidence": round(confidence, 3),
            "actual_label": actual_class if actual_class else "Unknown",
            "is_correct": (predicted_class == actual_class) if actual_class else None,
            "source": source,
            "run_id": run_id if run_id is not None else "-",
            "trial_id": trial_id if trial_id is not None else "-"
        }
        self.history.append(entry)

        # Do not append 'WAIT' / 'Rest' automatically to sentence unless intended
        if auto_append and command not in ["WAIT", "REST"]:
            self.sentence_tokens.append(command)

        return entry

    def append_command(self, command: str):
        """Manually appends a command to the sentence buffer."""
        self.sentence_tokens.append(command)

    def backspace(self):
        """Removes the last appended command."""
        if self.sentence_tokens:
            self.sentence_tokens.pop()

    def clear_sentence(self):
        """Clears the sentence buffer."""
        self.sentence_tokens.clear()

    def clear_history(self):
        """Clears classification history log."""
        self.history.clear()

    def get_sentence_text(self) -> str:
        """Returns the formatted sentence string."""
        return " ".join(self.sentence_tokens)

    def get_history_dataframe(self) -> pd.DataFrame:
        """Returns the session classification history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "timestamp", "predicted_task", "mapped_command",
                "calibrated_confidence", "actual_label", "is_correct", "source"
            ])
        return pd.DataFrame(self.history)


def generate_tts_html(text: str, autoplay: bool = False) -> str:
    """
    Generates an HTML snippet using the browser's native Web SpeechSynthesis API.
    Works directly inside Streamlit via st.components.v1.html.
    """
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    autoplay_script = "window.onload = function() { speakText(); };" if autoplay else ""

    html = f"""
    <div style="font-family: system-ui, sans-serif; padding: 6px 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <button id="tts-speak-btn" onclick="speakText()" style="
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            </svg>
            🔊 Speak Aloud (TTS)
        </button>

        <button id="copy-btn" onclick="copyText()" style="
            background: #1e293b;
            color: #38bdf8;
            border: 1px solid #38bdf8;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: transform 0.1s ease;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            📋 Copy Text
        </button>

        <span id="tts-status" style="margin-left: 6px; font-size: 13px; color: #94a3b8; font-weight: 500;"></span>
    </div>
    <script>
        function speakText() {{
            const statusEl = document.getElementById('tts-status');
            if (!('speechSynthesis' in window)) {{
                statusEl.innerText = "Browser TTS is not supported in this browser.";
                statusEl.style.color = "#ef4444";
                return;
            }}
            window.speechSynthesis.cancel();
            const text = "{safe_text}";
            if (!text.trim()) {{
                statusEl.innerText = "Text buffer is empty.";
                statusEl.style.color = "#eab308";
                return;
            }}
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.onstart = function() {{
                statusEl.innerText = "Speaking: \\"" + text + "\\"";
                statusEl.style.color = "#6366f1";
            }};
            utterance.onend = function() {{
                statusEl.innerText = "Completed speech.";
                statusEl.style.color = "#10b981";
                setTimeout(() => {{ statusEl.innerText = ""; }}, 3000);
            }};
            utterance.onerror = function(event) {{
                statusEl.innerText = "Speech error: " + event.error;
                statusEl.style.color = "#ef4444";
            }};
            window.speechSynthesis.speak(utterance);
        }}

        function copyText() {{
            const statusEl = document.getElementById('tts-status');
            const text = "{safe_text}";
            if (!text.trim()) {{
                statusEl.innerText = "Nothing to copy (buffer empty).";
                statusEl.style.color = "#eab308";
                return;
            }}
            navigator.clipboard.writeText(text).then(function() {{
                statusEl.innerText = "Copied to clipboard!";
                statusEl.style.color = "#10b981";
                setTimeout(() => {{ statusEl.innerText = ""; }}, 3000);
            }}, function(err) {{
                statusEl.innerText = "Could not copy: " + err;
                statusEl.style.color = "#ef4444";
            }});
        }}
        {autoplay_script}
    </script>
    """
    return html
