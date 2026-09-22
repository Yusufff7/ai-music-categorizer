import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import joblib

from music_categorizer.models import MusicGenreModel
from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.paths import ENSEMBLE_DIR, ENSEMBLE_RESULTS_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model_idx, df, report_file):
    model_path = ENSEMBLE_DIR / f'model_{model_idx}.pt'
    mlb_path = ENSEMBLE_DIR / f'mlb_{model_idx}.pkl'
    scaler_path = ENSEMBLE_DIR / f'scaler_{model_idx}.pkl'
    thresholds_path = ENSEMBLE_DIR / 'genre_thresholds.pkl'

    print(f"Evaluating model {model_idx}...")

    mlb = joblib.load(mlb_path)
    scaler = joblib.load(scaler_path)

    # Prepare dataset using this model's mlb and scaler
    X, y, _, _, _ = prepare_dataset(df, 'main_genres', mlb=mlb, scaler=scaler)
    input_dim = X.shape[1]
    num_classes = y.shape[1]

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_np = y if isinstance(y, np.ndarray) else y.numpy()

    # Load and run model
    model = MusicGenreModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # Apply thresholds
    thresholds = joblib.load(thresholds_path)
    threshold_arr = np.array([thresholds[g] for g in mlb.classes_])
    y_pred_bin = (y_pred > threshold_arr).astype(int)

    # Classification report
    report = classification_report(
        y_np, y_pred_bin, target_names=mlb.classes_,
        zero_division=0, digits=3
    )

    with open(report_file, 'a') as f:
        f.write(f"\n\n===== Model {model_idx} =====\n")
        f.write(report)


if __name__ == "__main__":
    os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)
    output_path = ENSEMBLE_RESULTS_DIR / 'ensemble_full.txt'
    if os.path.exists(output_path):
        os.remove(output_path)

    print("Loading dataset...")
    df = load_dataframe(tag_type='genre')

    for i in range(1, 11):  # model_1 to model_10
        evaluate_model(i, df, output_path)

    print(f"\nAll model reports written to {output_path}")
