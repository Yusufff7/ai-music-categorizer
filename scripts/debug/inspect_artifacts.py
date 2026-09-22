import joblib
import torch
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from music_categorizer.models import MusicGenreModel
from music_categorizer.paths import (GENRE_THRESHOLDS_PATH, GENRE_MLB_PATH,
                                     GENRE_SCALER_PATH, GENRE_MODEL_PATH)

# Point these at other artifacts (e.g. an ensemble member) to inspect them instead
threshold_path = GENRE_THRESHOLDS_PATH
mlb_path = GENRE_MLB_PATH
scaler_path = GENRE_SCALER_PATH
model_path = GENRE_MODEL_PATH

# Load thresholds
thresholds = joblib.load(threshold_path)
print("=== GENRE THRESHOLDS ===")
for genre, thresh in sorted(thresholds.items(), key=lambda x: x[1]):
    print(f"{genre.ljust(15)}: {thresh:.4f}")

# Plot threshold distribution
plt.bar(thresholds.keys(), thresholds.values())
plt.xticks(rotation=90)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.title("Genre Threshold Values")
plt.tight_layout()
plt.show()

# Load MultiLabelBinarizer
mlb = joblib.load(mlb_path)
print("\n=== CLASSES ===")
print(mlb.classes_)

print("\n=== SAMPLE TRANSFORM ===")
sample_tags = [['rock'], ['electronic', 'pop']]
print("Original:", sample_tags)
print("Encoded:", mlb.transform(sample_tags))

# Load Scaler
scaler = joblib.load(scaler_path)
print("\n=== SCALER DETAILS ===")
print("Mean shape:", scaler.mean_.shape)
print("Mean (first 5):", scaler.mean_[:5])
print("Scale (first 5):", scaler.scale_[:5])
print("Num features:", len(scaler.mean_))

# Load Model
input_dim = len(scaler.mean_)
num_classes = len(mlb.classes_)
model = MusicGenreModel(input_dim, num_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Print output layer biases
print("\n=== MODEL OUTPUT LAYER BIASES ===")
last_layer = list(model.children())[-1]
if hasattr(last_layer, 'bias') and last_layer.bias is not None:
    biases = last_layer.bias.detach().cpu().numpy()
    rows = [(genre, f"{bias:.4f}") for genre, bias in zip(mlb.classes_, biases)]
    print(tabulate(rows, headers=["Genre", "Bias"]))
else:
    print("No bias terms found in final layer.")
