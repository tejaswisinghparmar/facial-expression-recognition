"""
NeuroSignSpeak — Master Controller Entry Point
================================================
Launches the multimodal dashboard that unifies:
    • Webcam + DeepFace emotion detection   (Thread A)
    • PhysioNet EEG stream simulator        (Thread B)
    • MNE / CSP / SVM EEG processor         (Thread C)
    • Weighted fusion → translated speech

Usage
-----
    python run_dashboard.py              # default — subject 1
    python run_dashboard.py --subject 3  # use PhysioNet subject 3

The original ASL‑only app is still available via:
    python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys

# Configure logging before any module imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="NeuroSignSpeak — Multimodal Translation Dashboard",
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="PhysioNet EEGBCI subject number (1–109). Default: 1",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Import here so logging is configured first
    from dashboard import NeuroSignSpeakDashboard

    app = NeuroSignSpeakDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
