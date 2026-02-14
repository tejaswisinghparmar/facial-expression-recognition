"""
Emotion Detection Module
Uses DeepFace to detect emotions: Happiness, Sadness, Fear, Disgust, Anger, Surprise.
"""

import cv2
import numpy as np
from deepface import DeepFace


# Emotion labels we care about
EMOTION_LABELS = ["happy", "sad", "fear", "disgust", "angry", "surprise"]

# Colors for each emotion (BGR)
EMOTION_COLORS = {
    "happy":    (0, 255, 255),   # Yellow
    "sad":      (255, 150, 0),   # Blue
    "fear":     (200, 200, 200), # Gray
    "disgust":  (0, 180, 0),     # Green
    "angry":    (0, 0, 255),     # Red
    "surprise": (255, 0, 255),   # Magenta
    "neutral":  (200, 200, 200), # Gray
}


def run_emotion_detection(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Analyse a single BGR frame for emotions.

    Returns
    -------
    annotated_frame : np.ndarray
        Copy of *frame* with bounding‑box + emotion text drawn on it.
    results : dict
        ``{"dominant": str, "scores": {emotion: float, ...}}``
        Empty dict when no face is detected.
    """
    annotated = frame.copy()
    results: dict = {}

    try:
        analyses = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv",
        )

        # DeepFace may return a list or a single dict
        if isinstance(analyses, dict):
            analyses = [analyses]

        for analysis in analyses:
            region = analysis.get("region", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            dominant = analysis.get("dominant_emotion", "neutral")
            emotions = analysis.get("emotion", {})

            # Filter to only the emotions we care about
            filtered = {k: round(v, 1) for k, v in emotions.items() if k in EMOTION_LABELS}

            color = EMOTION_COLORS.get(dominant, (200, 200, 200))

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw dominant emotion label
            label = f"{dominant.upper()} ({filtered.get(dominant, 0):.0f}%)"
            cv2.putText(
                annotated, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

            # Build bar chart on the side
            bar_x = x + w + 10
            bar_y = y
            for i, (emo, score) in enumerate(sorted(filtered.items(), key=lambda e: -e[1])):
                bar_w = int(score * 1.2)
                emo_color = EMOTION_COLORS.get(emo, (200, 200, 200))
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), emo_color, -1)
                cv2.putText(
                    annotated,
                    f"{emo[:3]} {score:.0f}%",
                    (bar_x + bar_w + 4, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, emo_color, 1,
                )
                bar_y += 20

            results = {"dominant": dominant, "scores": filtered}

    except Exception:
        # No face detected or analysis failed — return frame as‑is
        pass

    return annotated, results
