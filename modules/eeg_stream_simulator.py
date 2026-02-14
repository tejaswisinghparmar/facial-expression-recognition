"""
EEG Stream Simulator — Thread B
================================
Iterates through PhysioNet Motor Imagery data (EEGBCI) to mimic a live
EEG hardware stream.  Produces chunks of raw EEG samples that the
EEG Processor (Thread C) can consume through a shared queue.

Designed to be drop‑in replaceable with an OpenBCI / LSL reader later —
just swap this module for one that pushes real hardware samples into the
same ``queue.Queue``.

PhysioNet EEGBCI dataset
------------------------
* 64‑channel EEG, 160 Hz sampling rate
* Task 2 (runs 4, 8, 12):  Left‑fist vs Right‑fist motor imagery
"""

from __future__ import annotations

import time
import threading
import queue
import logging
from dataclasses import dataclass, field

import numpy as np
import mne

logger = logging.getLogger(__name__)

# ──────────────────────── Configuration ──────────────────────────────
# PhysioNet subject / runs for left‑fist vs right‑fist motor imagery
DEFAULT_SUBJECT = 1
DEFAULT_RUNS = [4, 8, 12]        # T1=left fist, T2=right fist
CHUNK_DURATION_S = 1.0           # seconds per chunk pushed to the queue
PLAYBACK_SPEED = 1.0             # 1.0 = real‑time, 2.0 = 2× speed
EEG_SFREQ = 160.0               # PhysioNet native sampling rate

# Channels of interest (sensorimotor cortex — mu/beta)
CHANNELS_OF_INTEREST = ["C3", "Cz", "C4"]


@dataclass
class EEGChunk:
    """A single chunk of EEG data pushed through the queue."""
    data: np.ndarray                # shape (n_channels, n_samples)
    ch_names: list[str]
    sfreq: float
    timestamp: float                # wallclock time of emission
    event_id: int | None = None     # ground‑truth label (if known)
    event_name: str | None = None   # e.g. "left_fist", "right_fist"


class EEGStreamSimulator(threading.Thread):
    """
    Thread that reads PhysioNet EEGBCI data and pushes ``EEGChunk``
    objects into a ``queue.Queue`` at (approximately) real‑time pace.

    Parameters
    ----------
    chunk_queue : queue.Queue
        Shared queue consumed by the EEG Processor thread.
    subject : int
        PhysioNet subject number (1–109).
    runs : list[int]
        Run numbers to load (default = motor imagery runs).
    chunk_duration : float
        Duration of each chunk in seconds.
    loop : bool
        Whether to loop the data continuously.
    """

    # Map PhysioNet event codes → human labels
    EVENT_MAP = {1: "rest", 2: "left_fist", 3: "right_fist"}

    def __init__(
        self,
        chunk_queue: queue.Queue,
        subject: int = DEFAULT_SUBJECT,
        runs: list[int] | None = None,
        chunk_duration: float = CHUNK_DURATION_S,
        loop: bool = True,
    ):
        super().__init__(daemon=True, name="EEG-StreamSimulator")
        self._queue = chunk_queue
        self._subject = subject
        self._runs = runs or DEFAULT_RUNS
        self._chunk_dur = chunk_duration
        self._loop = loop
        self._stop_event = threading.Event()

        # Will be populated in _load_data()
        self._raw: mne.io.Raw | None = None
        self._events: np.ndarray | None = None
        self._ch_names: list[str] = []
        self._sfreq: float = EEG_SFREQ

    # ── public API ───────────────────────────────────────────────────
    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── data loading ─────────────────────────────────────────────────
    def _load_data(self):
        """Download (if needed) and load PhysioNet EEGBCI runs."""
        logger.info(
            "Loading PhysioNet EEGBCI — subject %d, runs %s …",
            self._subject, self._runs,
        )

        raw_fnames = mne.datasets.eegbci.load_data(
            self._subject, self._runs, update_path=True,
        )
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)

        # Standardise channel names to 10‑20 system
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")

        # Pick only channels of interest (if available)
        available = [ch for ch in CHANNELS_OF_INTEREST if ch in raw.ch_names]
        if available:
            raw.pick_channels(available, ordered=True)
        else:
            # Fallback: pick first 3 EEG channels
            raw.pick_types(eeg=True)
            raw.pick_channels(raw.ch_names[:3], ordered=True)

        # Extract events embedded in annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        self._raw = raw
        self._events = events
        self._ch_names = list(raw.ch_names)
        self._sfreq = raw.info["sfreq"]

        logger.info(
            "Loaded %d channels, %.1f s of data at %.0f Hz",
            len(self._ch_names),
            raw.times[-1],
            self._sfreq,
        )

    # ── the main thread loop ─────────────────────────────────────────
    def run(self):
        """Thread entry‑point: load data then stream chunks."""
        try:
            self._load_data()
        except Exception:
            logger.exception("Failed to load EEG data")
            return

        data = self._raw.get_data()               # (n_ch, total_samples)
        n_samples_total = data.shape[1]
        chunk_samples = int(self._sfreq * self._chunk_dur)
        sleep_time = self._chunk_dur / PLAYBACK_SPEED

        idx = 0
        while not self._stop_event.is_set():
            end = idx + chunk_samples
            if end > n_samples_total:
                if self._loop:
                    idx = 0
                    continue
                else:
                    break

            chunk_data = data[:, idx:end]

            # Find event in this window (if any)
            event_id = None
            event_name = None
            if self._events is not None and len(self._events):
                mask = (self._events[:, 0] >= idx) & (self._events[:, 0] < end)
                if mask.any():
                    eid = int(self._events[mask][-1, 2])
                    event_id = eid
                    event_name = self.EVENT_MAP.get(eid, f"event_{eid}")

            chunk = EEGChunk(
                data=chunk_data,
                ch_names=self._ch_names,
                sfreq=self._sfreq,
                timestamp=time.time(),
                event_id=event_id,
                event_name=event_name,
            )

            try:
                self._queue.put(chunk, timeout=1.0)
            except queue.Full:
                pass  # drop chunk if consumer is too slow

            idx = end
            self._stop_event.wait(timeout=sleep_time)

        logger.info("EEG stream simulator stopped.")
