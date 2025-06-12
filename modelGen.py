import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models


def extractFeatures(file_path):
    audio, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    return np.mean(mfccs, axis=1)

X = []
Y = []

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

for genre in genres:
    genre_path = os.path.join("genres", genre)

    for filename in os.listdir(genre_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(genre_path, filename)

            features = extractFeatures(file_path)
            X.append(features)
            Y.append(genres.index(genre))

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42  
)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))

test_acc = model.evaluate(X_test, Y_test)[1]
print(f"\nTest accuracy: {test_acc:.2%}")
model.save("genre_classifier.keras")
