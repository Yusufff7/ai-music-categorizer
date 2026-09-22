import os
import pandas as pd

from music_categorizer.data import load_dataframe, prepare_dataset
from music_categorizer.genre_trainer import train_model
from music_categorizer.paths import (GENRE_MODEL_DIR, GENRE_MODEL_PATH,
                                     GENRE_MLB_PATH, GENRE_SCALER_PATH)


def main():
    df = load_dataframe(tag_type='genre')
    print("Class distribution in training data:")
    print(pd.Series([g for sublist in df['main_genres'] for g in sublist]).value_counts())
    X, y, mlb, scaler, y_raw = prepare_dataset(df, 'main_genres')

    os.makedirs(GENRE_MODEL_DIR, exist_ok=True)
    train_model(
        X, y, mlb,
        model_path=GENRE_MODEL_PATH,
        mlb_path=GENRE_MLB_PATH,
        scaler_path=GENRE_SCALER_PATH,
        scaler=scaler
    )


if __name__ == "__main__":
    main()
