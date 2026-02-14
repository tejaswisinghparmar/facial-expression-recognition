"""
app.py — CustomTkinter GUI for Emotion Detection & ASL Recognition
Dark theme with purple accents.
"""

import threading
import time
import customtkinter as ctk
import cv2
from PIL import Image

from modules.emotion_detector import run_emotion_detection
from modules.asl_recognizer import ASLRecognizer
from modules.ollama_client import process_with_ollama


# ─────────────────────────────── Theme ───────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Purple accent palette
PURPLE       = "#7B2FBE"
PURPLE_HOVER = "#9B4FDE"
PURPLE_DARK  = "#4A1A72"
BG_DARK      = "#1A1A2E"
BG_SIDEBAR   = "#16213E"
BG_CARD      = "#0F3460"
TEXT_LIGHT    = "#E0E0E0"
TEXT_DIM      = "#A0A0B0"


class App(ctk.CTk):
    """Main application window."""

    MODE_NONE    = "none"
    MODE_EMOTION = "emotion"
    MODE_ASL     = "asl"

    WIDTH  = 1100
    HEIGHT = 700

    def __init__(self):
        super().__init__()

        self.title("Neuro_Sing_Speak Dashboard")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARK)

        # State
        self._mode: str = self.MODE_NONE
        self._running: bool = False
        self._closing: bool = False
        self._cap: cv2.VideoCapture | None = None
        self._cam_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._asl: ASLRecognizer = ASLRecognizer()
        self._ollama_busy: bool = False

        self._build_ui()

    # ================================================================
    #  UI Construction
    # ================================================================
    def _build_ui(self):
        # Grid: sidebar (col 0) | main area (col 1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=BG_SIDEBAR)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(10, weight=1)  # spacer

        logo = ctk.CTkLabel(
            sidebar, text="⚡ AI Vision",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=PURPLE,
        )
        logo.grid(row=0, column=0, padx=20, pady=(28, 8))

        subtitle = ctk.CTkLabel(
            sidebar, text="Emotion · ASL · Ollama",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 24))

        # Mode buttons
        self.btn_emotion = ctk.CTkButton(
            sidebar, text="😊  Emotion Mode",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=PURPLE, hover_color=PURPLE_HOVER,
            height=44, corner_radius=10,
            command=self._toggle_emotion,
        )
        self.btn_emotion.grid(row=2, column=0, padx=18, pady=(0, 10), sticky="ew")

        self.btn_asl = ctk.CTkButton(
            sidebar, text="🤟  ASL Mode",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=PURPLE, hover_color=PURPLE_HOVER,
            height=44, corner_radius=10,
            command=self._toggle_asl,
        )
        self.btn_asl.grid(row=3, column=0, padx=18, pady=(0, 10), sticky="ew")

        sep = ctk.CTkFrame(sidebar, height=2, fg_color=PURPLE_DARK)
        sep.grid(row=4, column=0, padx=18, pady=12, sticky="ew")

        # Ollama controls
        self.btn_ollama = ctk.CTkButton(
            sidebar, text="🧠  Send to Ollama",
            font=ctk.CTkFont(size=13),
            fg_color=BG_CARD, hover_color=PURPLE_DARK,
            height=38, corner_radius=8,
            command=self._send_to_ollama,
        )
        self.btn_ollama.grid(row=5, column=0, padx=18, pady=(0, 6), sticky="ew")

        self.btn_clear = ctk.CTkButton(
            sidebar, text="🗑  Clear Buffer",
            font=ctk.CTkFont(size=13),
            fg_color=BG_CARD, hover_color=PURPLE_DARK,
            height=38, corner_radius=8,
            command=self._clear_buffer,
        )
        self.btn_clear.grid(row=6, column=0, padx=18, pady=(0, 6), sticky="ew")

        # Space key hint
        self.btn_space = ctk.CTkButton(
            sidebar, text="␣  Add Space",
            font=ctk.CTkFont(size=13),
            fg_color=BG_CARD, hover_color=PURPLE_DARK,
            height=38, corner_radius=8,
            command=self._add_space,
        )
        self.btn_space.grid(row=7, column=0, padx=18, pady=(0, 6), sticky="ew")

        # Status label
        self.status_label = ctk.CTkLabel(
            sidebar, text="Status: Idle",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
            wraplength=180, justify="left",
        )
        self.status_label.grid(row=11, column=0, padx=20, pady=(0, 16), sticky="sw")

        # ── Main area ────────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        main.grid(row=0, column=1, sticky="nswe", padx=(4, 0))
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Video feed
        self.video_label = ctk.CTkLabel(
            main, text="Select a mode to start the camera feed",
            font=ctk.CTkFont(size=16), text_color=TEXT_DIM,
            fg_color=BG_CARD, corner_radius=14,
        )
        self.video_label.grid(row=0, column=0, padx=14, pady=(14, 6), sticky="nswe")

        # Bottom text area
        bottom = ctk.CTkFrame(main, fg_color=BG_DARK, height=160)
        bottom.grid(row=1, column=0, padx=14, pady=(4, 14), sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        # ASL live string
        lbl_asl = ctk.CTkLabel(
            bottom, text="Live ASL String",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=PURPLE,
        )
        lbl_asl.grid(row=0, column=0, padx=(0, 6), sticky="w")

        self.asl_textbox = ctk.CTkTextbox(
            bottom, height=80, font=ctk.CTkFont(size=13),
            fg_color=BG_CARD, text_color=TEXT_LIGHT,
            corner_radius=8, border_color=PURPLE_DARK, border_width=1,
        )
        self.asl_textbox.grid(row=1, column=0, padx=(0, 6), pady=(2, 0), sticky="ew")

        # Ollama interpretation
        lbl_ollama = ctk.CTkLabel(
            bottom, text="Ollama Interpretation",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=PURPLE,
        )
        lbl_ollama.grid(row=0, column=1, padx=(6, 0), sticky="w")

        self.ollama_textbox = ctk.CTkTextbox(
            bottom, height=80, font=ctk.CTkFont(size=13),
            fg_color=BG_CARD, text_color=TEXT_LIGHT,
            corner_radius=8, border_color=PURPLE_DARK, border_width=1,
        )
        self.ollama_textbox.grid(row=1, column=1, padx=(6, 0), pady=(2, 0), sticky="ew")

        # Emotion info label (shown only in emotion mode)
        self.emotion_info = ctk.CTkLabel(
            bottom, text="",
            font=ctk.CTkFont(size=13), text_color=TEXT_LIGHT,
            fg_color=BG_CARD, corner_radius=8,
            wraplength=500, justify="left",
        )
        # Not placed on grid yet — shown dynamically

        # Close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================
    #  Mode toggling
    # ================================================================
    def _toggle_emotion(self):
        if self._mode == self.MODE_EMOTION:
            self._stop_camera()
            return
        self._switch_mode(self.MODE_EMOTION)

    def _toggle_asl(self):
        if self._mode == self.MODE_ASL:
            self._stop_camera()
            return
        self._switch_mode(self.MODE_ASL)

    def _switch_mode(self, new_mode: str):
        """Stop the current camera thread (if any) and start the new mode."""
        self._stop_camera(keep_mode=True)
        self._mode = new_mode
        self._update_buttons()
        self._start_camera()

    def _update_buttons(self):
        active_fg   = "#FFFFFF"
        inactive_fg = PURPLE

        if self._mode == self.MODE_EMOTION:
            self.btn_emotion.configure(fg_color="#5C1F9E", text_color=active_fg)
            self.btn_asl.configure(fg_color=PURPLE, text_color=active_fg)
            self.status_label.configure(text="Status: Emotion Detection active")
        elif self._mode == self.MODE_ASL:
            self.btn_asl.configure(fg_color="#5C1F9E", text_color=active_fg)
            self.btn_emotion.configure(fg_color=PURPLE, text_color=active_fg)
            self.status_label.configure(text="Status: ASL Recognition active")
        else:
            self.btn_emotion.configure(fg_color=PURPLE)
            self.btn_asl.configure(fg_color=PURPLE)
            self.status_label.configure(text="Status: Idle")

    # ================================================================
    #  Camera loop (threaded)
    # ================================================================
    def _start_camera(self):
        self._stop_event.clear()

        # Try opening the camera with a few retries (device may still be releasing)
        cap = None
        for _ in range(5):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(0.15)

        if cap is None or not cap.isOpened():
            if not self._closing:
                self.status_label.configure(text="Status: ⚠ Camera not found")
            return

        self._cap = cap
        self._running = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _stop_camera(self, keep_mode: bool = False):
        """Stop the camera thread and release the device.

        Parameters
        ----------
        keep_mode : bool
            If *True*, don't reset ``_mode`` to NONE (used when switching modes).
        """
        self._running = False
        self._stop_event.set()

        # Wait for the camera thread to finish
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=3.0)
        self._cam_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if not self._closing and not keep_mode:
            self._mode = self.MODE_NONE
            self._update_buttons()
            try:
                self.video_label.configure(image=None, text="Select a mode to start the camera feed")
            except Exception:
                pass

    def _camera_loop(self):
        """Runs in a background thread; grabs frames and updates the GUI."""
        while not self._stop_event.is_set() and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror

            mode = self._mode  # snapshot to avoid mid-loop changes

            if mode == self.MODE_EMOTION:
                display, results = run_emotion_detection(frame)
                if results:
                    self._update_emotion_info(results)
            elif mode == self.MODE_ASL:
                display, letter = self._asl.run_asl_recognition(frame)
                if letter:
                    self._update_asl_textbox()
            else:
                display = frame

            if not self._stop_event.is_set():
                self._show_frame(display)

            # ~30 FPS cap — use event wait so we can break out quickly
            self._stop_event.wait(timeout=0.03)

    def _show_frame(self, frame):
        """Convert an OpenCV BGR frame to a CTkImage and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Scale to fit the label while keeping aspect ratio
        try:
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
        except Exception:
            lbl_w, lbl_h = 640, 480

        if lbl_w < 10 or lbl_h < 10:
            lbl_w, lbl_h = 640, 480

        img.thumbnail((lbl_w, lbl_h), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

        # Schedule GUI update on the main thread
        self.video_label.after(0, lambda i=ctk_img: self.video_label.configure(image=i, text=""))

    # ================================================================
    #  Text‑box helpers
    # ================================================================
    def _update_asl_textbox(self):
        text = self._asl.get_buffer_string()
        def _update():
            self.asl_textbox.delete("1.0", "end")
            self.asl_textbox.insert("1.0", text)
        self.asl_textbox.after(0, _update)

    def _update_emotion_info(self, results: dict):
        dominant = results.get("dominant", "—")
        scores = results.get("scores", {})
        parts = [f"{e.capitalize()}: {s:.0f}%" for e, s in sorted(scores.items(), key=lambda x: -x[1])]
        info = f"Dominant: {dominant.upper()}\n" + "  |  ".join(parts)
        self.status_label.after(0, lambda: self.status_label.configure(text=f"Emotion: {dominant.upper()}"))

    def _add_space(self):
        self._asl.letter_buffer.append(" ")
        self._update_asl_textbox()

    def _clear_buffer(self):
        self._asl.clear_buffer()
        self._update_asl_textbox()
        self.ollama_textbox.delete("1.0", "end")

    # ================================================================
    #  Ollama
    # ================================================================
    def _send_to_ollama(self):
        text = self._asl.get_buffer_string().strip()
        if not text:
            self.ollama_textbox.delete("1.0", "end")
            self.ollama_textbox.insert("1.0", "(buffer is empty)")
            return
        if self._ollama_busy:
            return

        self._ollama_busy = True
        self.btn_ollama.configure(state="disabled", text="⏳ Processing…")
        self.ollama_textbox.delete("1.0", "end")
        self.ollama_textbox.insert("1.0", "Sending to Ollama…")

        def on_result(corrected: str):
            self._ollama_busy = False
            def _update():
                self.ollama_textbox.delete("1.0", "end")
                self.ollama_textbox.insert("1.0", corrected)
                self.btn_ollama.configure(state="normal", text="🧠  Send to Ollama")
            self.ollama_textbox.after(0, _update)

        process_with_ollama(text, callback=on_result)

    # ================================================================
    #  Cleanup
    # ================================================================
    def _on_close(self):
        self._closing = True
        self._stop_camera()
        try:
            self._asl.release()
        except Exception:
            pass
        self.quit()
        self.destroy()
