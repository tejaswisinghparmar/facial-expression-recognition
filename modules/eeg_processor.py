"""
EEG Processor — Thread C
=========================
Consumes raw EEG chunks from the stream simulator queue, applies the
DSP pipeline from the Technical Feasibility Report, and runs CSP + SVM
classification.

DSP Pipeline (per the report)
-----------------------------
1. FIR High‑Pass filter at 0.5 Hz  (remove DC drift)
2. Notch filter at 50 Hz           (remove mains interference)
3. Epoch extraction                 (segment around events)
4. CSP spatial filtering            (maximise class variance)
5. SVM classification              (left‑fist vs right‑fist)

The module exposes two things consumed by the dashboard:
    • ``latest_result``  — most recent classification label + confidence
    • ``band_power_history`` — rolling Mu/Beta power for the EEG graph

Designed for modularity: swap the simulator queue for an OpenBCI LSL
inlet later without changing this file.
"""

from __future__ import annotations

import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from scipy.signal import welch

import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from modules.eeg_stream_simulator import EEGChunk

logger = logging.getLogger(__name__)

# ──────────────────────── Configuration ──────────────────────────────
MU_BAND = (8, 12)       # Hz — mu rhythm
BETA_BAND = (13, 30)    # Hz — beta rhythm
N_CSP_COMPONENTS = 4     # CSP spatial filters to keep
POWER_HISTORY_LEN = 120  # data‑points kept for the scrolling graph

# Physionet class mapping for the training step
CLASS_LEFT = "T1"        # left fist
CLASS_RIGHT = "T2"       # right fist


@dataclass
class EEGResult:
    """Result of a single EEG classification."""
    class_label: str          # "left_fist", "right_fist", "rest", "unknown"
    confidence: float         # 0–1 (SVM decision function, scaled)
    mu_power: float           # current mu‑band power (µV²)
    beta_power: float         # current beta‑band power (µV²)
    timestamp: float = 0.0


class EEGProcessor(threading.Thread):
    """
    Thread that pulls ``EEGChunk``s from a queue, applies DSP +
    CSP + SVM, and exposes results for the GUI.

    Parameters
    ----------
    chunk_queue : queue.Queue
        Shared queue from the EEG Stream Simulator.
    train_subject : int
        PhysioNet subject used to train the CSP+SVM pipeline.
    train_runs : list[int]
        Runs used for training.
    """

    LABEL_MAP = {1: "rest", 2: "left_fist", 3: "right_fist"}

    def __init__(
        self,
        chunk_queue: queue.Queue,
        train_subject: int = 1,
        train_runs: list[int] | None = None,
    ):
        super().__init__(daemon=True, name="EEG-Processor")
        self._queue = chunk_queue
        self._stop_event = threading.Event()

        self._train_subject = train_subject
        self._train_runs = train_runs or [4, 8, 12]

        # Results — read by the GUI on the main thread
        self.latest_result: EEGResult | None = None
        self.mu_power_history: deque[float] = deque(maxlen=POWER_HISTORY_LEN)
        self.beta_power_history: deque[float] = deque(maxlen=POWER_HISTORY_LEN)

        # Pipeline built during training
        self._pipeline: Pipeline | None = None
        self._info: mne.Info | None = None      # channel info from training data
        self._trained = False

    # ── public API ───────────────────────────────────────────────────
    def stop(self):
        self._stop_event.set()

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── DSP helpers (per Technical Feasibility Report) ───────────────
    @staticmethod
    def _apply_dsp(raw: mne.io.RawArray) -> mne.io.RawArray:
        """
        Apply the DSP pipeline from the report:
        1. FIR High‑Pass 0.5 Hz
        2. Notch 50 Hz
        """
        raw.filter(
            l_freq=0.5,
            h_freq=None,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
        raw.notch_filter(freqs=50.0, method="fir", verbose=False)
        return raw

    @staticmethod
    def _compute_band_power(
        data: np.ndarray,
        sfreq: float,
        band: tuple[float, float],
    ) -> float:
        """Average band power across channels using Welch's method."""
        freqs, psd = welch(data, fs=sfreq, nperseg=min(data.shape[-1], int(sfreq)))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not mask.any():
            return 0.0
        return float(np.mean(psd[..., mask]))

    # ── offline training on PhysioNet ────────────────────────────────
    def _train_pipeline(self):
        """
        Train CSP + SVM on PhysioNet EEGBCI data.
        Uses the same DSP steps that will be applied to live chunks.
        """
        logger.info("Training CSP+SVM on subject %d …", self._train_subject)

        # 1. Load data
        raw_fnames = eegbci.load_data(
            self._train_subject, self._train_runs, update_path=True,
        )
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")

        # Pick sensorimotor channels
        picks = mne.pick_types(raw.info, eeg=True)
        raw.pick(picks)

        # 2. DSP — FIR HP 0.5 Hz + Notch 50 Hz
        self._apply_dsp(raw)

        # 3. Band‑pass for CSP (mu + beta = 8‑30 Hz)
        raw.filter(8.0, 30.0, method="fir", fir_design="firwin", verbose=False)

        # 4. Epoch around events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Map annotation labels → we want classes T1 (left) and T2 (right)
        # PhysioNet annotations: T0=rest, T1=left, T2=right
        wanted = {}
        for k, v in event_id.items():
            if "T1" in k:
                wanted[k] = v
            elif "T2" in k:
                wanted[k] = v
        if not wanted:
            # Fallback: use numeric event IDs 2 and 3
            wanted = {k: v for k, v in event_id.items() if v in (2, 3)}

        if not wanted:
            logger.warning("Could not find T1/T2 events — using all events")
            wanted = event_id

        epochs = mne.Epochs(
            raw, events, event_id=wanted,
            tmin=0.5, tmax=3.5,
            baseline=None, preload=True, verbose=False,
        )
        epochs.drop_bad(verbose=False)

        if len(epochs) < 10:
            logger.error("Not enough epochs (%d) to train — aborting", len(epochs))
            return

        X = epochs.get_data(copy=True)   # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2]          # event ids

        # 5. CSP → SVM pipeline
        csp = CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)
        svm = SVC(kernel="rbf", C=1.0, probability=True)
        self._pipeline = Pipeline([("CSP", csp), ("SVM", svm)])
        self._pipeline.fit(X, y)

        # Quick cross‑val score
        scores = cross_val_score(
            Pipeline([("CSP", CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf", log=True)),
                       ("SVM", SVC(kernel="rbf", C=1.0))]),
            X, y, cv=5, scoring="accuracy",
        )
        logger.info("CSP+SVM cross‑val accuracy: %.2f ± %.2f", scores.mean(), scores.std())

        self._info = epochs.info
        self._trained = True
        logger.info("EEG pipeline training complete.")

    # ── classify a single chunk ──────────────────────────────────────
    def _classify_chunk(self, chunk: EEGChunk) -> EEGResult:
        """Apply DSP + trained pipeline to a single raw chunk."""
        sfreq = chunk.sfreq
        data = chunk.data  # (n_ch, n_samples)

        # Compute band power BEFORE heavy filtering (for the graph)
        mu_power = self._compute_band_power(data, sfreq, MU_BAND)
        beta_power = self._compute_band_power(data, sfreq, BETA_BAND)

        self.mu_power_history.append(mu_power)
        self.beta_power_history.append(beta_power)

        if not self._trained or self._pipeline is None:
            return EEGResult(
                class_label="untrained",
                confidence=0.0,
                mu_power=mu_power,
                beta_power=beta_power,
                timestamp=chunk.timestamp,
            )

        # Build a temporary Raw object so we can use MNE filters
        info = mne.create_info(
            ch_names=chunk.ch_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw_chunk = mne.io.RawArray(data, info, verbose=False)
        self._apply_dsp(raw_chunk)
        raw_chunk.filter(8.0, 30.0, method="fir", fir_design="firwin", verbose=False)

        filtered = raw_chunk.get_data()  # (n_ch, n_samples)

        # The pipeline expects (n_epochs, n_channels, n_times)
        # We need to match the training epoch length
        n_ch_pipeline = self._pipeline["CSP"].filters_.shape[1]
        if filtered.shape[0] != n_ch_pipeline:
            # Pad or trim channels to match training
            if filtered.shape[0] < n_ch_pipeline:
                pad = np.zeros((n_ch_pipeline - filtered.shape[0], filtered.shape[1]))
                filtered = np.vstack([filtered, pad])
            else:
                filtered = filtered[:n_ch_pipeline, :]

        X = filtered[np.newaxis, :, :]  # (1, n_ch, n_samples)

        try:
            pred = self._pipeline.predict(X)[0]
            proba = self._pipeline.predict_proba(X)[0]
            confidence = float(np.max(proba))
        except Exception as exc:
            logger.debug("Classification error: %s", exc)
            pred = 0
            confidence = 0.0

        # Map prediction to label
        label = self.LABEL_MAP.get(int(pred), "unknown")

        return EEGResult(
            class_label=label,
            confidence=confidence,
            mu_power=mu_power,
            beta_power=beta_power,
            timestamp=chunk.timestamp,
        )

    # ── main thread loop ─────────────────────────────────────────────
    def run(self):
        """Thread entry‑point: train the pipeline, then consume chunks."""
        try:
            self._train_pipeline()
        except Exception:
            logger.exception("EEG pipeline training failed")
            # Continue anyway — we'll just produce "untrained" labels

        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            result = self._classify_chunk(chunk)
            self.latest_result = result

        logger.info("EEG processor stopped.")
