# config.py
# Author: Daramola
# Central hub for all app configurations and constants.

from pathlib import Path

# --- PROJECT METADATA ---
APP_TITLE = "Shell.AI Hackathon 2025 - Blend Property Predictor"
TEAM_INFO = "Prototype by Neuralnetics (Olanrewaju Daramola, Emmanuel Olanrewaju, and Israel Trejo)"

# --- UI ICONS ---
ICONS = {
    "setup": "⚙️",
    "predict": "🔬",
    "results": "📊",
    "gpu": "✅",
    "cpu": "⚪",
    "success": "🎉",
    "error": "🚨",
    "info": "ℹ️",
    "warning": "⚠️",
    "blend": "⚗️"
}

# --- DEFAULT FILE PATHS ---
MODELS_DIR = Path("./models_full")
FEATURES_DIR = Path("./features")
RUNS_DIR = Path("./runs")
COLUMN_NAMES_PATH = Path("./features/column_names.json")