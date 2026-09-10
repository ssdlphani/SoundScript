import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv
import urllib.request
import os


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)

CLASS_MAP_FILE = "yamnet_class_map.csv"


# --------------------------------------------------
# Download YAMNet class names
# --------------------------------------------------

def download_class_map():

    if os.path.exists(CLASS_MAP_FILE):
        return

    print("Downloading YAMNet class names...")

    urllib.request.urlretrieve(
        CLASS_MAP_URL,
        CLASS_MAP_FILE
    )

    print("Class names downloaded!")


# --------------------------------------------------
# Load class names
# --------------------------------------------------

def load_class_names():

    class_names = []

    with open(
        CLASS_MAP_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            class_names.append(row["display_name"])

    return class_names


# --------------------------------------------------
# Load audio
# --------------------------------------------------

def load_audio(filename):

    audio, sample_rate = librosa.load(
        filename,
        sr=16000,
        mono=True
    )

    return audio.astype(np.float32)


# --------------------------------------------------
# Classify sound
# --------------------------------------------------

def classify_sound(filename):

    print("\nLoading audio...")

    audio = load_audio(filename)

    print("Analyzing audio...")

    scores, embeddings, spectrogram = model(audio)

    scores = scores.numpy()

    # Average prediction across all frames
    mean_scores = np.mean(scores, axis=0)

    # Get top 10 predictions
    top_indices = np.argsort(
        mean_scores
    )[::-1][:10]

    print("\n🔊 DETECTED SOUNDS")
    print("=" * 45)

    for index in top_indices:

        score = mean_scores[index]

        print(
            f"{class_names[index]:30s} "
            f"{score * 100:6.2f}%"
        )

    print("=" * 45)

    primary_index = top_indices[0]

    primary_sound = class_names[primary_index]

    primary_score = mean_scores[primary_index]

    print(
        f"\n🎯 Primary Sound: "
        f"{primary_sound}"
    )

    print(
        f"📊 Confidence: "
        f"{primary_score * 100:.2f}%"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

print("Loading YAMNet...")

model = hub.load(MODEL_URL)

print("YAMNet loaded!")

download_class_map()

class_names = load_class_names()

print(
    f"Loaded {len(class_names)} sound classes."
)


if __name__ == "__main__":

    classify_sound("test.wav")