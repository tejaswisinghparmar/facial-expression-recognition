"""
Fusion Logic — Weighted Decision Engine
========================================
Combines the ``dominant_emotion`` from DeepFace (Thread A) and the
``class_label`` from the EEG SVM classifier (Thread C) into a single
human‑readable "Translated Speech" string.

Strategy
--------
A simple weighted‑decision matrix:

    final_output = f(emotion_weight × emotion_context,
                     eeg_weight    × motor_intent)

The emotion provides *affect / tone* while the EEG motor‑imagery label
provides *intent / action*.  The two are fused into a sentence that a
speech synthesiser could vocalise.

Weights default to 50 / 50 but are tunable via ``set_weights()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ──────────────────────── Intent mapping ─────────────────────────────
# Maps EEG motor‑imagery labels → intent phrases
EEG_INTENT = {
    "left_fist":  "I want to go left",
    "right_fist": "I want to go right",
    "rest":       "I am resting",
    "unknown":    "…",
    "untrained":  "(calibrating…)",
}

# Maps DeepFace dominant emotion → tone / context modifiers
EMOTION_CONTEXT = {
    "happy":    {"tone": "happily",   "prefix": "I'm feeling great!"},
    "sad":      {"tone": "sadly",     "prefix": "I feel a bit down."},
    "angry":    {"tone": "urgently",  "prefix": "I'm frustrated."},
    "fear":     {"tone": "anxiously", "prefix": "I'm scared."},
    "surprise": {"tone": "excitedly", "prefix": "Wow!"},
    "disgust":  {"tone": "uneasily",  "prefix": "Something's off."},
    "neutral":  {"tone": "calmly",    "prefix": ""},
}

# Composite templates keyed by (emotion_category, intent_category)
# emotion_category: "positive" | "negative" | "neutral"
# intent_category:  "action" | "rest" | "unknown"
_EMOTION_CATEGORIES = {
    "happy": "positive", "surprise": "positive",
    "sad": "negative", "angry": "negative",
    "fear": "negative", "disgust": "negative",
    "neutral": "neutral",
}

_INTENT_CATEGORIES = {
    "left_fist": "action", "right_fist": "action",
    "rest": "rest",
    "unknown": "unknown", "untrained": "unknown",
}


@dataclass
class FusionResult:
    """Output of the fusion engine."""
    translated_speech: str       # final sentence for display / TTS
    emotion: str                 # raw emotion label
    eeg_label: str               # raw EEG label
    emotion_confidence: float    # 0–1
    eeg_confidence: float        # 0–1
    composite_confidence: float  # weighted combination


class FusionEngine:
    """
    Stateless weighted‑decision fusion between emotion and EEG streams.

    Parameters
    ----------
    emotion_weight : float
        Weight for emotion channel (0–1).
    eeg_weight : float
        Weight for EEG channel (0–1).
        ``emotion_weight + eeg_weight`` should equal 1.0.
    """

    def __init__(self, emotion_weight: float = 0.5, eeg_weight: float = 0.5):
        self._ew = emotion_weight
        self._bw = eeg_weight
        self._normalise_weights()

    def set_weights(self, emotion_weight: float, eeg_weight: float):
        self._ew = emotion_weight
        self._bw = eeg_weight
        self._normalise_weights()

    def _normalise_weights(self):
        total = self._ew + self._bw
        if total <= 0:
            self._ew, self._bw = 0.5, 0.5
        else:
            self._ew /= total
            self._bw /= total

    # ── main fusion call ─────────────────────────────────────────────
    def fuse(
        self,
        dominant_emotion: str,
        emotion_confidence: float,
        eeg_label: str,
        eeg_confidence: float,
    ) -> FusionResult:
        """
        Combine emotion and EEG into a translated speech string.

        Parameters
        ----------
        dominant_emotion : str
            e.g. "happy", "sad", "neutral"
        emotion_confidence : float
            0–1 confidence from DeepFace.
        eeg_label : str
            e.g. "left_fist", "right_fist", "rest"
        eeg_confidence : float
            0–1 confidence from SVM.

        Returns
        -------
        FusionResult
        """
        emotion = dominant_emotion.lower()
        eeg = eeg_label.lower()

        # Look up emotion context
        emo_ctx = EMOTION_CONTEXT.get(emotion, EMOTION_CONTEXT["neutral"])
        tone = emo_ctx["tone"]
        prefix = emo_ctx["prefix"]

        # Look up EEG intent
        intent = EEG_INTENT.get(eeg, EEG_INTENT["unknown"])

        # Composite confidence
        composite = (self._ew * emotion_confidence) + (self._bw * eeg_confidence)

        # Build the sentence
        emo_cat = _EMOTION_CATEGORIES.get(emotion, "neutral")
        intent_cat = _INTENT_CATEGORIES.get(eeg, "unknown")

        if intent_cat == "unknown":
            # No EEG signal — fall back to emotion‑only
            if prefix:
                sentence = f"{prefix} (waiting for motor imagery…)"
            else:
                sentence = "Waiting for EEG input…"
        elif intent_cat == "rest":
            if emo_cat == "positive":
                sentence = f"{prefix} {intent} — feeling {tone}."
            elif emo_cat == "negative":
                sentence = f"{prefix} {intent}."
            else:
                sentence = f"{intent}."
        else:
            # We have both emotion + action
            if prefix:
                sentence = f"{prefix} {intent}, {tone}."
            else:
                sentence = f"{intent}, {tone}."

        # Clean up whitespace
        sentence = " ".join(sentence.split())

        return FusionResult(
            translated_speech=sentence,
            emotion=emotion,
            eeg_label=eeg,
            emotion_confidence=emotion_confidence,
            eeg_confidence=eeg_confidence,
            composite_confidence=round(composite, 3),
        )


# ── module‑level convenience instance ────────────────────────────────
_default_engine = FusionEngine()


def fuse_signals(
    dominant_emotion: str,
    emotion_confidence: float,
    eeg_label: str,
    eeg_confidence: float,
) -> FusionResult:
    """Convenience wrapper around the default ``FusionEngine``."""
    return _default_engine.fuse(
        dominant_emotion, emotion_confidence,
        eeg_label, eeg_confidence,
    )
