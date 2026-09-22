import os
import numpy as np
import librosa

from music_categorizer.paths import AUDIO_DIR, FEATURE_DIR
from music_categorizer.utils import print_step


def extract_features(filepath):
    """Load the first 30s of an audio file and return a 1-D feature vector."""
    y, sr = librosa.load(filepath, sr=22050, duration=30)

    features = {
        # Spectral/Timbre
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
        'spectral_flatness': librosa.feature.spectral_flatness(y=y),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y),

        # Harmonic
        'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
        'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),

        # Rhythm
        'tempogram': librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr)),
        'tempo': librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
    }

    # Aggregate features
    feature_vector = np.concatenate([
        np.mean(feat, axis=1) if isinstance(feat, np.ndarray)
        else [feat]  # For scalar values like tempo
        for feat in features.values()
    ])
    return feature_vector


def extract_features_cached(filepath):
    """Feature extraction for dataset tracks (paths relative to AUDIO_DIR), cached as .npy."""
    cache_path = os.path.join(FEATURE_DIR, filepath.replace('/', '__') + '.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)

    try:
        print_step(f"Extracting features for {filepath}", 2)
        full_path = os.path.join(AUDIO_DIR, filepath)

        # Add file validation
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            print_step(f"Invalid file: {filepath}", 2)
            return None

        feature_vector = extract_features(full_path)

        print_step(f"Caching features to {cache_path}", 3)
        os.makedirs(FEATURE_DIR, exist_ok=True)
        np.save(cache_path, feature_vector)
        return feature_vector

    except Exception as e:
        print_step(f"Error processing {filepath}: {e}", 2)
        return None
