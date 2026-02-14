"""
ASL Finger‑Spelling Recognition Module
Uses MediaPipe Hand Landmarker (Tasks API) to detect hand landmarks
and classify static ASL letters.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Path to the downloaded .task model file
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")

# MediaPipe hand‑connection list for drawing
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                                # palm
]


def _draw_landmarks(image: np.ndarray, landmarks, w: int, h: int) -> None:
    """Draw hand landmarks + connections on *image* (in‑place)."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (144, 238, 144), 2)
    for px, py in pts:
        cv2.circle(image, (px, py), 4, (200, 120, 255), -1)


class ASLRecognizer:
    """Wraps a MediaPipe HandLandmarker and maps landmarks → ASL letters."""

    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        self.letter_buffer: list[str] = []
        self._prev_letter: str | None = None
        self._stable_count: int = 0
        self._STABLE_THRESHOLD = 12  # frames before accepting a letter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_asl_recognition(self, frame: np.ndarray) -> tuple[np.ndarray, str | None]:
        """
        Process a single BGR frame.

        Returns
        -------
        annotated : np.ndarray
            Frame with landmarks drawn.
        letter : str | None
            Newly recognised letter (only when stable), else None.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Convert BGR → RGB and wrap in mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        detected_letter: str | None = None

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw landmarks manually
                _draw_landmarks(annotated, hand_landmarks, w, h)

                letter = self._classify(hand_landmarks)

                if letter:
                    # Stability filter — same letter for N consecutive frames
                    if letter == self._prev_letter:
                        self._stable_count += 1
                    else:
                        self._prev_letter = letter
                        self._stable_count = 1

                    if self._stable_count == self._STABLE_THRESHOLD:
                        self.letter_buffer.append(letter)
                        detected_letter = letter

                    # Show current prediction
                    cv2.putText(
                        annotated, f"ASL: {letter}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 120, 255), 3,
                    )
        else:
            self._prev_letter = None
            self._stable_count = 0

        # Draw the accumulated buffer
        buffer_text = "".join(self.letter_buffer)
        cv2.putText(
            annotated, f"Buffer: {buffer_text}",
            (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2,
        )

        return annotated, detected_letter

    def get_buffer_string(self) -> str:
        return "".join(self.letter_buffer)

    def clear_buffer(self):
        self.letter_buffer.clear()
        self._prev_letter = None
        self._stable_count = 0

    def release(self):
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Landmark → letter heuristic classifier
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(hand_landmarks) -> str | None:
        """
        Rule‑based classifier for a subset of static ASL letters using
        normalised landmark positions.

        The Tasks API returns a list of NormalizedLandmark objects (with
        .x, .y, .z attributes), indexed 0‑20 the same way as the legacy
        solutions API.
        """
        lm = hand_landmarks  # list[NormalizedLandmark]

        def tip_above_pip(tip_id: int, pip_id: int) -> bool:
            """True when a fingertip is *above* (lower y) its PIP joint → finger extended."""
            return lm[tip_id].y < lm[pip_id].y

        thumb_open = lm[4].x < lm[3].x  # (right‑hand assumption)
        index_open = tip_above_pip(8, 6)
        middle_open = tip_above_pip(12, 10)
        ring_open = tip_above_pip(16, 14)
        pinky_open = tip_above_pip(20, 18)

        fingers_up = [thumb_open, index_open, middle_open, ring_open, pinky_open]
        count = sum(fingers_up)

        # --- Simple deterministic rules for common letters ---

        # A — fist with thumb alongside (no fingers open)
        if count == 0:
            return "A"

        # B — four fingers up, thumb closed
        if not thumb_open and index_open and middle_open and ring_open and pinky_open:
            return "B"

        # C — all fingers curved — simplified: thumb open + nothing else
        if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
            return "C"

        # D — index up, others closed
        if not thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
            return "D"

        # L — thumb + index open (L shape)
        if thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
            return "L"

        # V / Peace — index + middle open
        if not thumb_open and index_open and middle_open and not ring_open and not pinky_open:
            return "V"

        # W — index + middle + ring open
        if not thumb_open and index_open and middle_open and ring_open and not pinky_open:
            return "W"

        # Y — thumb + pinky open
        if thumb_open and not index_open and not middle_open and not ring_open and pinky_open:
            return "Y"

        # I — only pinky open
        if not thumb_open and not index_open and not middle_open and not ring_open and pinky_open:
            return "I"

        # 5 / Open hand — all fingers open
        if count == 5:
            return "5"

        return None
