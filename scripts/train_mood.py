import os
import pandas as pd

from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.mood_trainer import train_model
from music_categorizer.paths import (MOOD_MODEL_DIR, MOOD_MODEL_PATH,
                                     MOOD_MLB_PATH, MOOD_SCALER_PATH)


def main():
    df = load_dataframe(tag_type='mood')
    print("Class distribution in training data:")
    print(pd.Series([m for sublist in df['moods'] for m in sublist]).value_counts())

    X, y, mlb, scaler, y_raw = prepare_dataset(df, label_column='moods')

    os.makedirs(MOOD_MODEL_DIR, exist_ok=True)
    train_model(
        X, y, mlb,
        model_path=MOOD_MODEL_PATH,
        mlb_path=MOOD_MLB_PATH,
        scaler_path=MOOD_SCALER_PATH,
        scaler=scaler
    )


if __name__ == "__main__":
    main()
