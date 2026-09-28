import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directories & Safe Fallbacks
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PAYLOADS_DIR = os.path.join(BASE_DIR, "payloads")
DATASETS_DIR = str(BASE_DIR / "datasets")

# Model File Paths
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
TABULAR_RANKER_PATH = os.path.join(MODELS_DIR, "tabular_ranker.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.h5")
AUTOENCODER_PATH = os.path.join(MODELS_DIR, "autoencoder.pkl")

# Application & Scanner Limits
MAX_ENDPOINTS = int(os.getenv("MAX_ENDPOINTS", 20))
SCAN_TIMEOUT = int(os.getenv("SCAN_TIMEOUT", 30))

# Flask & Dashboard Settings
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secure-key")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'database.db'}")

# Dashboard Auth & Security
DASHBOARD_AUTH_ENABLED = os.getenv("DASHBOARD_AUTH_ENABLED", "False") == "True"
DASHBOARD_ADMIN_USER = os.getenv("DASHBOARD_ADMIN_USER", "admin")
DASHBOARD_ADMIN_PASSWORD = os.getenv("DASHBOARD_ADMIN_PASSWORD", "admin")
CSRF_ENABLED = os.getenv("CSRF_ENABLED", "True") == "True"
