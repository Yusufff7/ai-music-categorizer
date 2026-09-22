import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from music_categorizer.models import MoodClassifier
from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.paths import MOOD_MODEL_PATH, MOOD_RESULTS_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model():
    print("Loading data...")
    df = load_dataframe(tag_type='mood')
    X, y, mlb, scaler, _ = prepare_dataset(df, 'moods')

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_np = y.numpy() if hasattr(y, 'numpy') else y

    print("Loading model...")
    input_dim = X.shape[1]
    num_classes = y.shape[1]

    model = MoodClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(MOOD_MODEL_PATH, map_location=device))
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

    y_true_labels = np.argmax(y_np, axis=1)

    print("Evaluating...")
    report = classification_report(y_true_labels, y_pred_labels, target_names=mlb.classes_, zero_division=0, digits=3)
    print(report)

    # Save raw probabilities per class
    os.makedirs(MOOD_RESULTS_DIR, exist_ok=True)
    prob_output_path = MOOD_RESULTS_DIR / 'mood_predictions_detailed.tsv'
    prob_df = pd.DataFrame(y_pred_probs, columns=mlb.classes_)
    prob_df.to_csv(prob_output_path, sep='\t', index=False)
    print(f"Saved raw predictions to {prob_output_path}")

    # Save per-track predicted and true labels
    print("Saving per-track predictions with true labels...")
    pred_tags = [mlb.classes_[i] for i in y_pred_labels]
    true_tags = [mlb.classes_[i] for i in y_true_labels]

    results_df = pd.DataFrame({
        'true_mood': true_tags,
        'predicted_mood': pred_tags
    })

    if 'PATH' in df.columns:
        results_df['filename'] = df['PATH']

    results_path = MOOD_RESULTS_DIR / 'mood_predictions_vs_truth.tsv'
    results_df.to_csv(results_path, sep='\t', index=False)
    print(f"Saved predictions vs. truth to {results_path}")


if __name__ == "__main__":
    evaluate_model()
