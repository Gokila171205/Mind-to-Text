"""
Neuroscript Smart Virtual BCI Keyboard Module.
Provides interactive QWERTY layout navigation, EEG-to-Keyboard action dispatching,
debounce / cooldown mechanics, trial segment deduplication, dictionary-based word suggestions,
and multi-modal provenance logging (Real EEG vs. Simulated Demo vs. Manual Click).
"""

import time
from typing import Dict, List, Optional, Tuple


# Standard QWERTY Layout Grid
KEYBOARD_ROWS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["SPACE", "DELETE", "CLEAR", "SELECT"]
]

# Flattened list of keys for linear cyclic navigation
ALL_KEYS: List[str] = [key for row in KEYBOARD_ROWS for key in row]

# Action definitions
ACTION_MOVE_LEFT = "MOVE_LEFT"
ACTION_MOVE_RIGHT = "MOVE_RIGHT"
ACTION_NEXT_ROW = "NEXT_ROW"
ACTION_SELECT = "SELECT"
ACTION_SPACE = "SPACE"
ACTION_NO_ACTION = "NO_ACTION"

ACTION_DESCRIPTIONS = {
    ACTION_MOVE_LEFT: "Move selection left (←)",
    ACTION_MOVE_RIGHT: "Move selection right (→)",
    ACTION_NEXT_ROW: "Move to next row (↓)",
    ACTION_SELECT: "Select highlighted key (↵)",
    ACTION_SPACE: "Insert space (␣)",
    ACTION_NO_ACTION: "Rest / No action (⏸)"
}

# Default EEG-to-Keyboard Action Mapping
DEFAULT_BCI_KEYBOARD_MAPPING: Dict[str, str] = {
    "Left Fist Imagery": ACTION_MOVE_LEFT,
    "Right Fist Imagery": ACTION_MOVE_RIGHT,
    "Both Fists Imagery": ACTION_SELECT,
    "Both Feet Imagery": ACTION_SPACE,
    "Rest": ACTION_NO_ACTION,
    "Left Fist (Exec)": ACTION_MOVE_LEFT,
    "Right Fist (Exec)": ACTION_MOVE_RIGHT,
    "Both Fists (Exec)": ACTION_SELECT,
    "Both Feet (Exec)": ACTION_SPACE,
}

# Quick Assistive Phrases for instant one-click communication
QUICK_PHRASES: List[str] = [
    "YES",
    "NO",
    "HELP",
    "I NEED WATER",
    "I AM FINE",
    "THANK YOU"
]

# Curated Medical & Daily Assistive English Word Suggestions Dictionary
COMMON_WORDS: List[str] = [
    # Urgent Assistive / Healthcare Needs
    "HELP", "WATER", "FOOD", "DOCTOR", "NURSE", "PAIN", "TIRED", "HUNGRY", "THIRSTY",
    "BATHROOM", "MEDICINE", "COLD", "HOT", "PLEASE", "THANKS", "THANK", "YES", "NO",
    "SLEEP", "FAMILY", "REST", "HOSPITAL", "DIZZY", "HURT", "BREATHE", "BED", "CHAIR",
    # Common Greetings & Core English
    "HELLO", "GOOD", "MORNING", "EVENING", "NIGHT", "HOW", "ARE", "YOU", "FINE", "OK",
    "TODAY", "TOMORROW", "NOW", "LATER", "SOON", "COME", "GO", "WANT", "NEED", "FEEL",
    "LIKE", "LOVE", "HAPPY", "SAD", "SMILE", "SIT", "STAND", "WALK", "TALK", "HEAR",
    # High-Frequency English Vocabulary
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I", "IT", "FOR", "NOT",
    "ON", "WITH", "HE", "AS", "YOU", "DO", "AT", "THIS", "BUT", "HIS", "BY", "FROM",
    "THEY", "WE", "SAY", "HER", "SHE", "OR", "AN", "WILL", "MY", "ONE", "ALL", "WOULD",
    "THERE", "THEIR", "WHAT", "SO", "UP", "OUT", "IF", "ABOUT", "WHO", "GET", "WHICH",
    "TIME", "JUST", "HIM", "KNOW", "TAKE", "PEOPLE", "INTO", "YEAR", "YOUR", "SOME",
    "COULD", "THEM", "SEE", "OTHER", "THAN", "THEN", "LOOK", "ONLY", "ITS", "OVER",
    "THINK", "ALSO", "BACK", "AFTER", "USE", "TWO", "OUR", "WORK", "FIRST", "WELL",
    "WAY", "EVEN", "NEW", "BECAUSE", "ANY", "THESE", "GIVE", "DAY", "MOST", "US",
    "GREAT", "LIFE", "MAN", "WORLD", "NAME", "HOME", "HOUSE", "FRIEND", "CALL", "OPEN",
    "CLOSE", "LIGHT", "DARK", "READ", "WRITE", "TEXT", "MIND", "BRAIN", "SIGNAL",
    "COMPUTER", "NEURON", "KEYBOARD", "SYSTEM", "COMMUNICATION", "ACCURATE", "MODEL"
]


def get_word_suggestions(current_text: str, max_suggestions: int = 5) -> List[str]:
    """
    Returns top prefix-matching word suggestions based on the currently active word fragment.
    If the text is empty or ends in a space, suggests top assistive / conversational tokens.
    """
    if not current_text or current_text.strip() == "":
        return ["HELP", "WATER", "YES", "NO", "PLEASE"][:max_suggestions]

    # Extract current word fragment (letters since last whitespace)
    if current_text.endswith(" "):
        # Last token finished, suggest common sentence starters / assistive words
        return ["PLEASE", "THANK", "HELP", "YES", "WATER"][:max_suggestions]

    tokens = current_text.split()
    prefix = tokens[-1].upper() if tokens else ""
    if not prefix:
        return ["HELP", "WATER", "YES", "NO", "PLEASE"][:max_suggestions]

    matches = [w for w in COMMON_WORDS if w.startswith(prefix) and w != prefix]
    # Sort matches by length (shorter/more frequent words first)
    matches.sort(key=lambda w: (len(w), w))

    seen = set()
    deduped = []
    for w in matches:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
            if len(deduped) >= max_suggestions:
                break

    return deduped


class VirtualBCIKeyboard:
    """
    Manages interactive on-screen QWERTY keyboard state, hierarchical Row-Column navigation,
    linear navigation, letter accumulation, EEG prediction ingestion, debounce cooldowns,
    segment deduplication, and provenance auditing.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        cooldown_seconds: float = 0.2,
        mapping: Optional[Dict[str, str]] = None,
        nav_mode: str = "LINEAR"
    ):
        self.accumulated_text: str = ""
        self.highlighted_index: int = 0  # For linear navigation
        self.confidence_threshold: float = confidence_threshold
        self.cooldown_seconds: float = cooldown_seconds
        self.mapping: Dict[str, str] = mapping.copy() if mapping else DEFAULT_BCI_KEYBOARD_MAPPING.copy()

        # Navigation Mode: 'ROW_COLUMN' (Fast 2-step scanning) or 'LINEAR'
        self.nav_mode: str = nav_mode
        self.active_row_idx: int = 0
        self.active_col_idx: int = 0
        self.nav_stage: str = "ROW"  # 'ROW' (selecting row) or 'COLUMN' (selecting key in row)

        # Protection against repeat triggering and rapid bouncing
        self.last_action_timestamp: float = 0.0
        self.last_processed_segment_id: Optional[str] = None

        # Detailed audit history of all actions
        self.action_history: List[Dict] = []

    @property
    def current_key(self) -> str:
        """Returns the currently highlighted key symbol depending on navigation mode."""
        if self.nav_mode == "ROW_COLUMN":
            row = KEYBOARD_ROWS[self.active_row_idx]
            if self.nav_stage == "COLUMN":
                return row[min(self.active_col_idx, len(row) - 1)]
            else:
                # In ROW stage, highlight first key of row or indicate row
                return row[0]
        else:
            if 0 <= self.highlighted_index < len(ALL_KEYS):
                return ALL_KEYS[self.highlighted_index]
            return ALL_KEYS[0]

    def set_highlighted_key(self, key_label: str):
        """Sets cursor to a specific key symbol."""
        if key_label in ALL_KEYS:
            self.highlighted_index = ALL_KEYS.index(key_label)
            # Synchronize row-column position
            for r_i, r in enumerate(KEYBOARD_ROWS):
                if key_label in r:
                    self.active_row_idx = r_i
                    self.active_col_idx = r.index(key_label)
                    self.nav_stage = "COLUMN"
                    break

    def navigate_left(self):
        """Moves cursor to the previous element."""
        if self.nav_mode == "ROW_COLUMN":
            if self.nav_stage == "ROW":
                self.active_row_idx = (self.active_row_idx - 1) % len(KEYBOARD_ROWS)
            else:
                row_len = len(KEYBOARD_ROWS[self.active_row_idx])
                self.active_col_idx = (self.active_col_idx - 1) % row_len
        else:
            self.highlighted_index = (self.highlighted_index - 1) % len(ALL_KEYS)

    def navigate_right(self):
        """Moves cursor to the next element."""
        if self.nav_mode == "ROW_COLUMN":
            if self.nav_stage == "ROW":
                self.active_row_idx = (self.active_row_idx + 1) % len(KEYBOARD_ROWS)
            else:
                row_len = len(KEYBOARD_ROWS[self.active_row_idx])
                self.active_col_idx = (self.active_col_idx + 1) % row_len
        else:
            self.highlighted_index = (self.highlighted_index + 1) % len(ALL_KEYS)

    def navigate_next_row(self):
        """Moves cursor to the next row (or switches back to row stage)."""
        if self.nav_mode == "ROW_COLUMN":
            if self.nav_stage == "COLUMN":
                # Return to row selection stage
                self.nav_stage = "ROW"
            else:
                self.active_row_idx = (self.active_row_idx + 1) % len(KEYBOARD_ROWS)
        else:
            curr_key = self.current_key
            row_idx = 0
            col_idx = 0
            for r_i, row in enumerate(KEYBOARD_ROWS):
                if curr_key in row:
                    row_idx = r_i
                    col_idx = row.index(curr_key)
                    break

            next_row_idx = (row_idx + 1) % len(KEYBOARD_ROWS)
            next_row = KEYBOARD_ROWS[next_row_idx]
            new_col_idx = min(col_idx, len(next_row) - 1)
            new_key = next_row[new_col_idx]
            self.highlighted_index = ALL_KEYS.index(new_key)

    def cancel_to_row_stage(self):
        """In Row-Column mode, resets cursor back to row selection stage."""
        self.nav_stage = "ROW"

    def select_current_key(self, source: str = "Manual") -> Tuple[str, str]:
        """
        Executes the selection action.
        In Row-Column mode:
          - Stage 'ROW': Confirms row, switches to 'COLUMN' stage.
          - Stage 'COLUMN': Executes the chosen key, resets to 'ROW' stage.
        In Linear mode:
          - Executes the highlighted key directly.
        """
        if self.nav_mode == "ROW_COLUMN" and self.nav_stage == "ROW":
            # Confirm row selection -> transition to column scanning
            self.nav_stage = "COLUMN"
            self.active_col_idx = 0
            selected_row_keys = " ".join(KEYBOARD_ROWS[self.active_row_idx])
            return "ROW_SELECTED", f"Selected Row {self.active_row_idx + 1} [{selected_row_keys}]. Now select letter."

        # Execute key action
        if self.nav_mode == "ROW_COLUMN":
            row = KEYBOARD_ROWS[self.active_row_idx]
            key = row[min(self.active_col_idx, len(row) - 1)]
            # Reset stage back to ROW for next character
            self.nav_stage = "ROW"
        else:
            key = self.current_key

        if key == "SPACE":
            self.accumulated_text += " "
            action = "SPACE"
            detail = "Appended space"
        elif key == "DELETE":
            if self.accumulated_text:
                self.accumulated_text = self.accumulated_text[:-1]
            action = "DELETE"
            detail = "Removed last character"
        elif key == "CLEAR":
            self.accumulated_text = ""
            action = "CLEAR"
            detail = "Cleared text buffer"
        elif key == "SELECT":
            action = "SELECT"
            detail = "Select button confirmed"
        else:
            self.accumulated_text += key
            action = f"TYPE_{key}"
            detail = f"Typed letter '{key}'"

        return action, detail

    def apply_phrase(self, phrase: str, source: str = "Manual") -> Dict:
        """Appends a quick assistive phrase to the accumulated text."""
        phrase = phrase.strip().upper()
        if self.accumulated_text and not self.accumulated_text.endswith(" "):
            self.accumulated_text += (" " + phrase + " ")
        else:
            self.accumulated_text += (phrase + " ")

        entry = self._record_event(
            action="QUICK_PHRASE",
            detail=f"Inserted phrase: '{phrase}'",
            source=source,
            key=phrase,
            status="SUCCESS"
        )
        return entry

    def type_letter_direct(self, letter: str, source: str = "Manual") -> Dict:
        """Directly types a letter (e.g. from manual on-screen button click)."""
        letter = letter.upper()
        if letter in ALL_KEYS and letter not in ["SPACE", "DELETE", "CLEAR", "SELECT"]:
            self.accumulated_text += letter
            self.set_highlighted_key(letter)
            entry = self._record_event(
                action=f"CLICK_{letter}",
                detail=f"Typed letter '{letter}'",
                source=source,
                key=letter,
                status="SUCCESS"
            )
            return entry
        elif letter == "SPACE":
            self.accumulated_text += " "
            self.set_highlighted_key("SPACE")
            return self._record_event("CLICK_SPACE", "Inserted space", source, "SPACE", "SUCCESS")
        elif letter == "DELETE":
            if self.accumulated_text:
                self.accumulated_text = self.accumulated_text[:-1]
            self.set_highlighted_key("DELETE")
            return self._record_event("CLICK_DELETE", "Backspace delete", source, "DELETE", "SUCCESS")
        elif letter == "CLEAR":
            self.accumulated_text = ""
            self.set_highlighted_key("CLEAR")
            return self._record_event("CLICK_CLEAR", "Cleared text", source, "CLEAR", "SUCCESS")

        return {"status": "IGNORED", "reason": "Invalid key"}

    def apply_word_suggestion(self, suggestion_word: str, source: str = "Manual") -> Dict:
        """Replaces current word fragment with chosen suggestion and adds a space."""
        suggestion_word = suggestion_word.upper().strip()
        tokens = self.accumulated_text.split(" ")
        if self.accumulated_text.endswith(" ") or not self.accumulated_text:
            self.accumulated_text += (suggestion_word + " ")
        else:
            tokens[-1] = suggestion_word
            self.accumulated_text = " ".join(tokens) + " "

        entry = self._record_event(
            action="APPLY_SUGGESTION",
            detail=f"Applied word '{suggestion_word}'",
            source=source,
            key=suggestion_word,
            status="SUCCESS"
        )
        return entry

    def backspace(self, source: str = "Manual"):
        """Removes the trailing character."""
        if self.accumulated_text:
            self.accumulated_text = self.accumulated_text[:-1]
            self._record_event("BACKSPACE", "Removed last character", source, "⌫", "SUCCESS")

    def clear(self, source: str = "Manual"):
        """Clears accumulated text completely."""
        self.accumulated_text = ""
        self._record_event("CLEAR_ALL", "Cleared text buffer", source, "CLEAR", "SUCCESS")

    def process_eeg_prediction(
        self,
        predicted_class: str,
        confidence: float,
        segment_id: str,
        mapping: Optional[Dict[str, str]] = None,
        actual_label: Optional[str] = None,
        is_synthetic: bool = False
    ) -> Dict:
        """
        Translates a genuine EEG classification prediction into a virtual keyboard action.
        Enforces confidence threshold, short cooldown, and segment deduplication.
        """
        source_label = "Synthetic Demo Prediction" if is_synthetic else "Real EEG Prediction"
        current_time = time.time()
        active_mapping = mapping if mapping is not None else self.mapping

        # 1. Deduplication Protection: Prevent processing the exact same trial segment twice
        if self.last_processed_segment_id is not None and segment_id == self.last_processed_segment_id:
            entry = self._record_event(
                action="REJECTED_DUPLICATE",
                detail=f"Segment '{segment_id}' was already processed. Advance to next EEG trial.",
                source=source_label,
                key=self.current_key,
                status="DUPLICATE_IGNORED",
                predicted_class=predicted_class,
                actual_label=actual_label,
                confidence=confidence,
                segment_id=segment_id
            )
            return entry

        # 2. Confidence Threshold Protection
        if confidence < self.confidence_threshold:
            entry = self._record_event(
                action="REJECTED_LOW_CONFIDENCE",
                detail=f"Confidence {confidence*100:.1f}% below required threshold ({self.confidence_threshold*100:.1f}%).",
                source=source_label,
                key=self.current_key,
                status="LOW_CONFIDENCE_IGNORED",
                predicted_class=predicted_class,
                actual_label=actual_label,
                confidence=confidence,
                segment_id=segment_id
            )
            return entry

        # 3. Cooldown Protection
        elapsed = current_time - self.last_action_timestamp
        if self.last_action_timestamp > 0 and elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            entry = self._record_event(
                action="REJECTED_COOLDOWN",
                detail=f"Cooldown active ({remaining:.1f}s remaining). Wait before next command.",
                source=source_label,
                key=self.current_key,
                status="COOLDOWN_IGNORED",
                predicted_class=predicted_class,
                actual_label=actual_label,
                confidence=confidence,
                segment_id=segment_id
            )
            return entry

        # 4. Resolve Action from Configurable Mapping
        action = active_mapping.get(predicted_class)
        if not action:
            # Fallback to normalized base label
            base_lbl = predicted_class.replace(" Imagery", "").replace(" (Exec)", "").replace("[SYNTHETIC DEMO]", "").strip()
            action = active_mapping.get(base_lbl, ACTION_NO_ACTION)

        action_detail = ""
        action_key = self.current_key

        if action == ACTION_MOVE_LEFT:
            self.navigate_left()
            action_detail = f"Moved cursor left to '{self.current_key}'"
        elif action == ACTION_MOVE_RIGHT:
            self.navigate_right()
            action_detail = f"Moved cursor right to '{self.current_key}'"
        elif action == ACTION_NEXT_ROW:
            self.navigate_next_row()
            action_detail = f"Moved cursor down to '{self.current_key}'"
        elif action == ACTION_SELECT:
            _, sel_detail = self.select_current_key(source=source_label)
            action_detail = f"Selected '{action_key}': {sel_detail}"
        elif action == ACTION_SPACE:
            self.accumulated_text += " "
            action_detail = "Inserted Space"
        elif action == ACTION_NO_ACTION:
            action_detail = "Rest state detected: No keyboard action"
        else:
            action_detail = f"Unknown action: {action}"

        # Update guard state
        self.last_action_timestamp = current_time
        self.last_processed_segment_id = segment_id

        entry = self._record_event(
            action=action,
            detail=action_detail,
            source=source_label,
            key=self.current_key,
            status="EXECUTED",
            predicted_class=predicted_class,
            actual_label=actual_label,
            confidence=confidence,
            segment_id=segment_id
        )
        return entry

    def _record_event(
        self,
        action: str,
        detail: str,
        source: str,
        key: str,
        status: str,
        predicted_class: Optional[str] = None,
        actual_label: Optional[str] = None,
        confidence: Optional[float] = None,
        segment_id: Optional[str] = None
    ) -> Dict:
        """Appends an event to audit history."""
        timestamp = time.strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "action": action,
            "detail": detail,
            "highlighted_key": key,
            "status": status,
            "source": source,
            "predicted_class": predicted_class if predicted_class else "-",
            "actual_label": actual_label if actual_label else "-",
            "confidence": f"{confidence*100:.1f}%" if confidence is not None else "-",
            "segment_id": segment_id if segment_id else "-",
            "accumulated_text": self.accumulated_text
        }
        self.action_history.append(entry)
        return entry
