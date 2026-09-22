import copy
import os
import tempfile
import numpy as np

from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.genre_trainer import train_model, DEFAULT_MANUAL_BOOSTS


def run_random_boost_search(X, y, mlb, scaler, trials=30):
    genres = list(mlb.classes_)
    base_boosts = DEFAULT_MANUAL_BOOSTS
    boost_ranges = {g: (0.3, 6.0) for g in genres}

    def sample_random_boosts():
        sampled = {}
        for genre in genres:
            base = base_boosts.get(genre, 1.0)
            low = max(0.3, base * 0.5)
            high = min(6.0, base * 1.5)
            sampled[genre] = float(np.random.uniform(low, high))
        return sampled

    best_macro_f1 = 0
    best_boosts = copy.deepcopy(base_boosts)

    for i in range(trials):
        boosts = sample_random_boosts()
        print(f"Trial {i+1}/{trials} boosts:")
        for g in sorted(boosts):
            print(f"  {g:12s}: {boosts[g]:.3f}")

        # Trial models are throwaway, so write them to a temp dir instead of models/
        with tempfile.TemporaryDirectory() as tmp:
            metrics = train_model(
                X, y, mlb,
                model_path=os.path.join(tmp, 'genre_model.pt'),
                mlb_path=os.path.join(tmp, 'mlb_genre.pkl'),
                manual_boosts=boosts,
                epochs=25
            )

        macro_f1 = metrics['val_macro_f1']
        print(f"  Validation Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_boosts = boosts
            print("  --> New best boosts!")

    print("\nBest manual boosts found:")
    for g in sorted(best_boosts):
        print(f"{g:12s}: {best_boosts[g]:.3f}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")

    return best_boosts


if __name__ == "__main__":
    df = load_dataframe(tag_type='genre')
    X, y, mlb, scaler, _ = prepare_dataset(df, 'main_genres')
    run_random_boost_search(X, y, mlb, scaler)
