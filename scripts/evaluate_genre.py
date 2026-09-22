import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import joblib

from music_categorizer.models import MusicGenreModel
from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.paths import GENRE_MODEL_PATH, GENRE_THRESHOLDS_PATH, GENRE_RESULTS_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model():
    print("Loading data...")
    df = load_dataframe(tag_type='genre')
    X, y, mlb, scaler, _ = prepare_dataset(df, 'main_genres')

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_np = y.numpy() if hasattr(y, 'numpy') else y

    print("Loading model...")
    input_dim = X.shape[1]
    num_classes = y.shape[1]

    model = MusicGenreModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(GENRE_MODEL_PATH, map_location=device))
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    print("Loading thresholds...")
    thresholds = joblib.load(GENRE_THRESHOLDS_PATH)
    threshold_arr = np.array([thresholds[g] for g in mlb.classes_])
    y_pred_bin = (y_pred > threshold_arr).astype(int)

    print("Evaluating...")
    report = classification_report(y_np, y_pred_bin, target_names=mlb.classes_, zero_division=0, digits=3)
    print(report)

    # Save raw probabilities
    os.makedirs(GENRE_RESULTS_DIR, exist_ok=True)
    prob_output_path = GENRE_RESULTS_DIR / 'genre_predictions_detailed.tsv'
    prob_df = pd.DataFrame(y_pred, columns=mlb.classes_)
    prob_df.to_csv(prob_output_path, sep='\t', index=False)
    print(f"Saved raw predictions to {prob_output_path}")

    # Save per-track predicted and true labels
    print("Saving per-track predictions with true labels...")
    pred_tags = mlb.inverse_transform(y_pred_bin)
    true_tags = mlb.inverse_transform(y_np)

    results_df = pd.DataFrame({
        'true_genres': [", ".join(tags) for tags in true_tags],
        'predicted_genres': [", ".join(tags) for tags in pred_tags]
    })

    if 'filename' in df.columns:
        results_df['filename'] = df['filename']

    results_path = GENRE_RESULTS_DIR / 'genre_predictions_vs_truth.tsv'
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f"Saved predictions vs. truth to {results_path}")


if __name__ == "__main__":
    evaluate_model()
