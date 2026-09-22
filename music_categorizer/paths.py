"""Central place for every file path used by the project.

All paths are anchored to the repository root, so scripts work no matter
which directory they are launched from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = ROOT / 'data'
TAGS_TSV = DATA_DIR / 'raw_30s_cleantags.tsv'
AUDIO_DIR = DATA_DIR / 'audio'        # MTG-Jamendo audio (not committed)
FEATURE_DIR = DATA_DIR / 'features'   # cached feature vectors (not committed)

# Trained models + preprocessing artifacts
MODELS_DIR = ROOT / 'models'

GENRE_MODEL_DIR = MODELS_DIR / 'genre'
GENRE_MODEL_PATH = GENRE_MODEL_DIR / 'genre_model.pt'
GENRE_MLB_PATH = GENRE_MODEL_DIR / 'mlb_genre.pkl'
GENRE_SCALER_PATH = GENRE_MODEL_DIR / 'scaler_genre.pkl'
GENRE_THRESHOLDS_PATH = GENRE_MODEL_DIR / 'genre_thresholds.pkl'

MOOD_MODEL_DIR = MODELS_DIR / 'mood'
MOOD_MODEL_PATH = MOOD_MODEL_DIR / 'mood_model.pt'
MOOD_MLB_PATH = MOOD_MODEL_DIR / 'mlb_mood.pkl'
MOOD_SCALER_PATH = MOOD_MODEL_DIR / 'scaler_mood.pkl'

ENSEMBLE_DIR = MODELS_DIR / 'genre_ensemble'

# Evaluation outputs
RESULTS_DIR = ROOT / 'results'
GENRE_RESULTS_DIR = RESULTS_DIR / 'genre'
MOOD_RESULTS_DIR = RESULTS_DIR / 'mood'
ENSEMBLE_RESULTS_DIR = RESULTS_DIR / 'genre_ensemble'
