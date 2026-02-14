"""
NeuroSignSpeak — Master Dashboard
==================================
A unified CustomTkinter GUI that integrates:

    Thread A  — Real‑time webcam + DeepFace emotion detection
    Thread B  — EEG stream simulator (PhysioNet EEGBCI)
    Thread C  — EEG DSP processor (MNE → CSP → SVM)
    Fusion    — Weighted decision combining emotion + EEG

Dashboard panels
----------------
┌──────────────┬────────────────────────────────────┐
│  Sidebar     │  Live video feed (DeepFace bbox)   │
│  ─────────   ├────────────────────────────────────┤
│  Controls    │  EEG Mu / Beta power graph         │
│  Status      ├────────────────────────────────────┤
│  Weights     │  ★ TRANSLATED SPEECH ★             │
│              │  Fusion details                     │
└──────────────┴────────────────────────────────────┘
"""

from __future__ import annotations

import queue
import threading
import time
import logging
from collections import deque

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from modules.emotion_detector import run_emotion_detection
from modules.eeg_stream_simulator import EEGStreamSimulator
from modules.eeg_processor import EEGProcessor
from modules.fusion import FusionEngine, FusionResult

logger = logging.getLogger(__name__)

# ─────────────────────────────── Theme ───────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

PURPLE       = "#7B2FBE"
PURPLE_HOVER = "#9B4FDE"
PURPLE_DARK  = "#4A1A72"
BG_DARK      = "#1A1A2E"
BG_SIDEBAR   = "#16213E"
BG_CARD      = "#0F3460"
TEXT_LIGHT    = "#E0E0E0"
TEXT_DIM      = "#A0A0B0"
GREEN_OK     = "#27AE60"
RED_ERR      = "#E74C3C"
YELLOW_WARN  = "#F39C12"

# Graph colours
MU_COLOR     = "#00D4FF"
BETA_COLOR   = "#FF6EC7"
GRID_COLOR   = "#2A2A4E"


class NeuroSignSpeakDashboard(ctk.CTk):
    """Master controller window for NeuroSignSpeak."""

    WIDTH  = 1280
    HEIGHT = 820

    def __init__(self):
        super().__init__()

        self.title("NeuroSignSpeak — Multimodal Translation Dashboard")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(1024, 700)
        self.configure(fg_color=BG_DARK)

        # ── Shared state ─────────────────────────────────────────────
        self._closing = False
        self._running = False

        # Webcam / emotion (Thread A)
        self._cap: cv2.VideoCapture | None = None
        self._cam_thread: threading.Thread | None = None
        self._cam_stop = threading.Event()
        self._last_emotion: str = "neutral"
        self._last_emotion_conf: float = 0.0
        self._last_emotion_scores: dict = {}

        # EEG (Threads B + C)
        self._eeg_queue: queue.Queue = queue.Queue(maxsize=50)
        self._eeg_sim: EEGStreamSimulator | None = None
        self._eeg_proc: EEGProcessor | None = None

        # Fusion
        self._fusion = FusionEngine(emotion_weight=0.5, eeg_weight=0.5)
        self._last_fusion: FusionResult | None = None

        # Graph data
        self._mu_history: deque[float] = deque(maxlen=120)
        self._beta_history: deque[float] = deque(maxlen=120)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start the periodic GUI updater
        self._tick()

    # ================================================================
    #  UI Construction
    # ================================================================
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────
        sb = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=BG_SIDEBAR)
        sb.grid(row=0, column=0, sticky="nswe")
        sb.grid_rowconfigure(20, weight=1)
        self._sidebar = sb

        row = 0
        ctk.CTkLabel(
            sb, text="🧠 NeuroSignSpeak",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=PURPLE,
        ).grid(row=row, column=0, padx=20, pady=(24, 4)); row += 1

        ctk.CTkLabel(
            sb, text="Multimodal Translation",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
        ).grid(row=row, column=0, padx=20, pady=(0, 20)); row += 1

        # ── Start / Stop ──
        self.btn_start = ctk.CTkButton(
            sb, text="▶  Start All", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=GREEN_OK, hover_color="#2ECC71", height=44, corner_radius=10,
            command=self._start_all,
        )
        self.btn_start.grid(row=row, column=0, padx=18, pady=(0, 6), sticky="ew"); row += 1

        self.btn_stop = ctk.CTkButton(
            sb, text="⏹  Stop All", font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=RED_ERR, hover_color="#C0392B", height=44, corner_radius=10,
            command=self._stop_all, state="disabled",
        )
        self.btn_stop.grid(row=row, column=0, padx=18, pady=(0, 14), sticky="ew"); row += 1

        # Separator
        ctk.CTkFrame(sb, height=2, fg_color=PURPLE_DARK).grid(
            row=row, column=0, padx=18, pady=8, sticky="ew"); row += 1

        # ── Weight sliders ──
        ctk.CTkLabel(
            sb, text="Fusion Weights",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=PURPLE,
        ).grid(row=row, column=0, padx=20, pady=(8, 2), sticky="w"); row += 1

        ctk.CTkLabel(sb, text="Emotion weight", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).grid(row=row, column=0, padx=20, sticky="w"); row += 1
        self.slider_emo = ctk.CTkSlider(
            sb, from_=0, to=1, number_of_steps=20,
            fg_color=BG_CARD, progress_color=PURPLE, button_color=PURPLE_HOVER,
            command=self._on_weight_change,
        )
        self.slider_emo.set(0.5)
        self.slider_emo.grid(row=row, column=0, padx=20, pady=(0, 4), sticky="ew"); row += 1

        ctk.CTkLabel(sb, text="EEG weight", font=ctk.CTkFont(size=11),
                     text_color=TEXT_DIM).grid(row=row, column=0, padx=20, sticky="w"); row += 1
        self.slider_eeg = ctk.CTkSlider(
            sb, from_=0, to=1, number_of_steps=20,
            fg_color=BG_CARD, progress_color=MU_COLOR, button_color=PURPLE_HOVER,
            command=self._on_weight_change,
        )
        self.slider_eeg.set(0.5)
        self.slider_eeg.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew"); row += 1

        self.weight_label = ctk.CTkLabel(
            sb, text="Emo 50% · EEG 50%",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
        )
        self.weight_label.grid(row=row, column=0, padx=20, sticky="w"); row += 1

        # Separator
        ctk.CTkFrame(sb, height=2, fg_color=PURPLE_DARK).grid(
            row=row, column=0, padx=18, pady=10, sticky="ew"); row += 1

        # ── Status indicators ──
        ctk.CTkLabel(
            sb, text="Thread Status",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=PURPLE,
        ).grid(row=row, column=0, padx=20, pady=(4, 4), sticky="w"); row += 1

        self.status_cam = ctk.CTkLabel(
            sb, text="● Webcam:  Stopped", font=ctk.CTkFont(size=11),
            text_color=RED_ERR,
        )
        self.status_cam.grid(row=row, column=0, padx=20, sticky="w"); row += 1

        self.status_eeg_sim = ctk.CTkLabel(
            sb, text="● EEG Sim: Stopped", font=ctk.CTkFont(size=11),
            text_color=RED_ERR,
        )
        self.status_eeg_sim.grid(row=row, column=0, padx=20, sticky="w"); row += 1

        self.status_eeg_proc = ctk.CTkLabel(
            sb, text="● EEG Proc: Stopped", font=ctk.CTkFont(size=11),
            text_color=RED_ERR,
        )
        self.status_eeg_proc.grid(row=row, column=0, padx=20, sticky="w"); row += 1

        self.status_fusion = ctk.CTkLabel(
            sb, text="● Fusion:  Idle", font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM,
        )
        self.status_fusion.grid(row=row, column=0, padx=20, sticky="w"); row += 1

        # ── Bottom status ──
        self.status_general = ctk.CTkLabel(
            sb, text="Status: Idle", font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM, wraplength=200, justify="left",
        )
        self.status_general.grid(row=21, column=0, padx=20, pady=(0, 14), sticky="sw")

        # ── Main panel ───────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        main.grid(row=0, column=1, sticky="nswe", padx=(4, 0))
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=3)  # video
        main.grid_rowconfigure(1, weight=2)  # EEG graph
        main.grid_rowconfigure(2, weight=0)  # translated speech

        # ── Video panel ──────────────────────────────────────────────
        self.video_label = ctk.CTkLabel(
            main, text="Press ▶ Start All to begin",
            font=ctk.CTkFont(size=16), text_color=TEXT_DIM,
            fg_color=BG_CARD, corner_radius=12,
        )
        self.video_label.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="nswe")

        # ── EEG graph panel ──────────────────────────────────────────
        graph_frame = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        graph_frame.grid(row=1, column=0, padx=12, pady=4, sticky="nswe")
        graph_frame.grid_columnconfigure(0, weight=1)
        graph_frame.grid_rowconfigure(1, weight=1)

        graph_header = ctk.CTkFrame(graph_frame, fg_color="transparent")
        graph_header.grid(row=0, column=0, padx=12, pady=(8, 0), sticky="ew")
        ctk.CTkLabel(
            graph_header, text="EEG Band Power",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=PURPLE,
        ).pack(side="left")
        ctk.CTkLabel(
            graph_header, text="■ Mu (8–12 Hz)",
            font=ctk.CTkFont(size=11), text_color=MU_COLOR,
        ).pack(side="left", padx=(20, 0))
        ctk.CTkLabel(
            graph_header, text="■ Beta (13–30 Hz)",
            font=ctk.CTkFont(size=11), text_color=BETA_COLOR,
        ).pack(side="left", padx=(12, 0))

        self.eeg_canvas = ctk.CTkCanvas(
            graph_frame, bg="#0D0D2B", highlightthickness=0,
        )
        self.eeg_canvas.grid(row=1, column=0, padx=8, pady=(4, 8), sticky="nswe")

        # ── Translated speech panel ──────────────────────────────────
        speech_frame = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        speech_frame.grid(row=2, column=0, padx=12, pady=(4, 12), sticky="ew")
        speech_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            speech_frame, text="★ Translated Speech",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=PURPLE,
        ).grid(row=0, column=0, padx=16, pady=(10, 0), sticky="w")

        self.speech_label = ctk.CTkLabel(
            speech_frame, text="Waiting for input…",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_LIGHT,
            wraplength=800, justify="left",
        )
        self.speech_label.grid(row=1, column=0, padx=16, pady=(4, 4), sticky="w")

        self.fusion_detail = ctk.CTkLabel(
            speech_frame, text="",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
            wraplength=800, justify="left",
        )
        self.fusion_detail.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="w")

    # ================================================================
    #  Weight slider callback
    # ================================================================
    def _on_weight_change(self, _=None):
        ew = self.slider_emo.get()
        bw = self.slider_eeg.get()
        self._fusion.set_weights(ew, bw)
        self.weight_label.configure(text=f"Emo {ew:.0%} · EEG {bw:.0%}")

    # ================================================================
    #  Start / Stop
    # ================================================================
    def _start_all(self):
        if self._running:
            return
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_general.configure(text="Status: Starting threads…")

        # Thread A — Webcam + DeepFace
        self._cam_stop.clear()
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.status_cam.configure(text="● Webcam:  NOT FOUND", text_color=RED_ERR)
            self._cap = None
        else:
            self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True, name="Thread-A-Webcam")
            self._cam_thread.start()
            self.status_cam.configure(text="● Webcam:  Running", text_color=GREEN_OK)

        # Thread B — EEG stream simulator
        self._eeg_sim = EEGStreamSimulator(
            chunk_queue=self._eeg_queue,
            subject=1,
            runs=[4, 8, 12],
            chunk_duration=1.0,
            loop=True,
        )
        self._eeg_sim.start()
        self.status_eeg_sim.configure(text="● EEG Sim: Running", text_color=GREEN_OK)

        # Thread C — EEG processor
        self._eeg_proc = EEGProcessor(
            chunk_queue=self._eeg_queue,
            train_subject=1,
            train_runs=[4, 8, 12],
        )
        self._eeg_proc.start()
        self.status_eeg_proc.configure(text="● EEG Proc: Training…", text_color=YELLOW_WARN)

        self.status_general.configure(text="Status: All threads active")

    def _stop_all(self):
        if not self._running:
            return
        self._running = False
        self.status_general.configure(text="Status: Stopping…")

        # Stop camera
        self._cam_stop.set()
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.status_cam.configure(text="● Webcam:  Stopped", text_color=RED_ERR)

        # Stop EEG simulator
        if self._eeg_sim:
            self._eeg_sim.stop()
            self._eeg_sim.join(timeout=3)
            self._eeg_sim = None
        self.status_eeg_sim.configure(text="● EEG Sim: Stopped", text_color=RED_ERR)

        # Stop EEG processor
        if self._eeg_proc:
            self._eeg_proc.stop()
            self._eeg_proc.join(timeout=3)
            self._eeg_proc = None
        self.status_eeg_proc.configure(text="● EEG Proc: Stopped", text_color=RED_ERR)

        self.status_fusion.configure(text="● Fusion:  Idle", text_color=TEXT_DIM)

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_general.configure(text="Status: Idle")

        try:
            self.video_label.configure(image=None, text="Press ▶ Start All to begin")
        except Exception:
            pass

    # ================================================================
    #  Thread A — Camera + DeepFace loop
    # ================================================================
    def _camera_loop(self):
        while not self._cam_stop.is_set() and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            annotated, results = run_emotion_detection(frame)

            if results:
                self._last_emotion = results.get("dominant", "neutral")
                scores = results.get("scores", {})
                dom_score = scores.get(self._last_emotion, 50.0)
                self._last_emotion_conf = dom_score / 100.0
                self._last_emotion_scores = scores

            if not self._cam_stop.is_set():
                self._show_frame(annotated)

            self._cam_stop.wait(timeout=0.03)

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        try:
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
        except Exception:
            lbl_w, lbl_h = 640, 360
        if lbl_w < 10 or lbl_h < 10:
            lbl_w, lbl_h = 640, 360
        img.thumbnail((lbl_w, lbl_h), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.after(0, lambda i=ctk_img: self.video_label.configure(image=i, text=""))

    # ================================================================
    #  Periodic GUI tick  (runs on main thread every 200 ms)
    # ================================================================
    def _tick(self):
        if self._closing:
            return

        if self._running:
            self._update_eeg_graph()
            self._update_fusion()
            self._update_thread_status()

        self.after(200, self._tick)

    def _update_thread_status(self):
        if self._eeg_proc and self._eeg_proc.is_trained:
            self.status_eeg_proc.configure(text="● EEG Proc: Running", text_color=GREEN_OK)

    def _update_fusion(self):
        """Pull latest data from threads and run fusion."""
        eeg_label = "unknown"
        eeg_conf = 0.0

        if self._eeg_proc and self._eeg_proc.latest_result:
            res = self._eeg_proc.latest_result
            eeg_label = res.class_label
            eeg_conf = res.confidence

        result = self._fusion.fuse(
            dominant_emotion=self._last_emotion,
            emotion_confidence=self._last_emotion_conf,
            eeg_label=eeg_label,
            eeg_confidence=eeg_conf,
        )
        self._last_fusion = result

        self.speech_label.configure(text=result.translated_speech)
        detail = (
            f"Emotion: {result.emotion.upper()} ({result.emotion_confidence:.0%})  ·  "
            f"EEG: {result.eeg_label.upper()} ({result.eeg_confidence:.0%})  ·  "
            f"Composite: {result.composite_confidence:.0%}"
        )
        self.fusion_detail.configure(text=detail)
        self.status_fusion.configure(text="● Fusion:  Active", text_color=GREEN_OK)

    # ================================================================
    #  EEG scrolling graph (pure Canvas — no matplotlib dependency)
    # ================================================================
    def _update_eeg_graph(self):
        canvas = self.eeg_canvas

        # Sync history from processor
        if self._eeg_proc:
            self._mu_history = self._eeg_proc.mu_power_history
            self._beta_history = self._eeg_proc.beta_power_history

        mu = list(self._mu_history)
        beta = list(self._beta_history)

        if not mu and not beta:
            return

        canvas.delete("all")

        try:
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
        except Exception:
            return
        if cw < 20 or ch < 20:
            return

        pad_l, pad_r, pad_t, pad_b = 50, 10, 10, 24
        gw = cw - pad_l - pad_r
        gh = ch - pad_t - pad_b

        # Determine Y range
        all_vals = mu + beta
        if not all_vals:
            return
        y_min = 0
        y_max = max(all_vals) * 1.2 if max(all_vals) > 0 else 1.0

        def y_to_canvas(v):
            if y_max == y_min:
                return pad_t + gh / 2
            return pad_t + gh - (v - y_min) / (y_max - y_min) * gh

        # Grid lines
        for i in range(5):
            gy = pad_t + gh * i / 4
            canvas.create_line(pad_l, gy, pad_l + gw, gy, fill=GRID_COLOR, dash=(2, 4))
            val = y_max - (y_max - y_min) * i / 4
            canvas.create_text(pad_l - 6, gy, text=f"{val:.1e}", fill=TEXT_DIM,
                               font=("Consolas", 8), anchor="e")

        # Plot lines
        max_pts = max(len(mu), len(beta))
        if max_pts < 2:
            return
        dx = gw / (max_pts - 1) if max_pts > 1 else gw

        def draw_line(data, color):
            if len(data) < 2:
                return
            pts = []
            for i, v in enumerate(data):
                x = pad_l + i * dx
                y = y_to_canvas(v)
                pts.extend([x, y])
            canvas.create_line(*pts, fill=color, width=2, smooth=True)

        draw_line(mu, MU_COLOR)
        draw_line(beta, BETA_COLOR)

        # X‑axis label
        canvas.create_text(pad_l + gw / 2, ch - 4, text="Time →", fill=TEXT_DIM,
                           font=("Consolas", 9))

    # ================================================================
    #  Cleanup
    # ================================================================
    def _on_close(self):
        self._closing = True
        self._stop_all()
        self.quit()
        self.destroy()
