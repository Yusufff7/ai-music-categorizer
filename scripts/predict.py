"""Predict the mood and genre(s) of a single audio file.

Usage: python scripts/predict.py <audio_file>
"""
import sys
import numpy as np
import torch
import joblib

from music_categorizer.features import extract_features
from music_categorizer.models import MoodClassifier, MusicGenreModel
from music_categorizer.paths import (
    MOOD_MODEL_PATH, MOOD_MLB_PATH, MOOD_SCALER_PATH,
    GENRE_MODEL_PATH, GENRE_MLB_PATH, GENRE_SCALER_PATH, GENRE_THRESHOLDS_PATH,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_mood_model(input_dim, num_classes):
    model = MoodClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(MOOD_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_genre_model(input_dim, num_classes):
    model = MusicGenreModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(GENRE_MODEL_PATH, map_location=device))
    model.eval()
    return model


def predict(filepath):
    feat = extract_features(filepath).reshape(1, -1)

    # Load scalers and mlbs
    mood_scaler = joblib.load(MOOD_SCALER_PATH)
    mood_mlb = joblib.load(MOOD_MLB_PATH)
    genre_scaler = joblib.load(GENRE_SCALER_PATH)
    genre_mlb = joblib.load(GENRE_MLB_PATH)
    genre_thresholds = joblib.load(GENRE_THRESHOLDS_PATH)

    # Prepare mood prediction
    mood_X = mood_scaler.transform(feat)
    mood_model = load_mood_model(mood_X.shape[1], len(mood_mlb.classes_))
    mood_tensor = torch.tensor(mood_X, dtype=torch.float32).to(device)

    with torch.no_grad():
        mood_outputs = mood_model(mood_tensor)
        mood_probs = torch.softmax(mood_outputs, dim=1).cpu().numpy()
    mood_pred_idx = np.argmax(mood_probs, axis=1)[0]
    mood_pred = mood_mlb.classes_[mood_pred_idx]
    print(f"Predicted Mood: {mood_pred}")

    # Prepare genre prediction
    genre_X = genre_scaler.transform(feat)
    genre_model = load_genre_model(genre_X.shape[1], len(genre_mlb.classes_))
    genre_tensor = torch.tensor(genre_X, dtype=torch.float32).to(device)

    with torch.no_grad():
        genre_outputs = genre_model(genre_tensor)
        genre_probs = genre_outputs.cpu().numpy()[0]

    # Apply thresholds
    predicted_genres = [genre for genre, prob in zip(genre_mlb.classes_, genre_probs)
                        if prob >= genre_thresholds.get(genre, 0.5)]

    print(f"Predicted Genres: {predicted_genres}")
    return mood_pred, predicted_genres


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Running prediction for: {audio_file}")
    predict(audio_file)
