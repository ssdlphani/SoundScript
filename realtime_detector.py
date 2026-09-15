import csv
import os
import time
import urllib.request

import numpy as np
import sounddevice as sd
import tensorflow_hub as hub


# ============================================================
# SOUNDSCRIPT - REAL-TIME ENVIRONMENTAL SOUND DETECTOR
# ============================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.975
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAP_PATH = os.path.join(BASE_DIR, "yamnet_class_map.csv")


# ============================================================
# CATEGORY DEFINITIONS
# ============================================================

CATEGORY_KEYWORDS = {
    "Speech": [
        "speech",
        "conversation",
        "narration",
        "monologue",
        "babbling",
        "child speech",
    ],

    "Dog": [
        "dog",
        "bark",
        "bow-wow",
        "growling",
        "whimper",
        "howl",
    ],

    "Vehicle": [
        "vehicle",
        "car",
        "motor vehicle",
        "engine",
        "truck",
        "bus",
        "motorcycle",
        "traffic noise",
        "car passing by",
        "race car",
    ],

    "Alarm": [
        "alarm",
        "siren",
        "smoke detector",
        "fire alarm",
        "burglar alarm",
        "emergency vehicle",
        "buzzer",
        "beep",
        "beeping",
    ],

    "Impact": [
        "clapping",
        "finger snapping",
        "knock",
        "slam",
        "thump",
        "bang",
        "crash",
        "glass",
    ],
}


CATEGORY_ICONS = {
    "Speech": "SPEECH",
    "Dog": "DOG",
    "Vehicle": "VEHICLE",
    "Alarm": "ALARM",
    "Impact": "IMPACT",
}


# Ignore extremely weak predictions
MIN_YAMNET_CONFIDENCE = 0.01

# Only display application categories above this level
DISPLAY_THRESHOLD = 0.03


# ============================================================
# LOAD YAMNET
# ============================================================

print()
print("=" * 65)
print(" SoundScript - Environmental Sound Detection")
print("=" * 65)

print("\nLoading YAMNet...")

yamnet_model = hub.load(YAMNET_URL)

print("YAMNet loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_MAP_PATH):
    print("Downloading YAMNet class map...")
    urllib.request.urlretrieve(CLASS_MAP_URL, CLASS_MAP_PATH)


class_names = []

with open(CLASS_MAP_PATH, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        class_names.append(row["display_name"])


print(f"Loaded {len(class_names)} YAMNet sound classes.")


# ============================================================
# CATEGORY MAPPING
# ============================================================

def label_matches(label, keywords):
    """
    Check whether a YAMNet label belongs to one of our
    application-level categories.
    """

    label = label.lower()

    for keyword in keywords:
        if keyword.lower() in label:
            return True

    return False


def map_to_categories(scores):
    """
    Convert all YAMNet predictions into our application
    categories.

    We use the maximum matching YAMNet confidence for each
    category rather than only looking at the single top label.
    """

    category_scores = {
        "Speech": 0.0,
        "Dog": 0.0,
        "Vehicle": 0.0,
        "Alarm": 0.0,
        "Impact": 0.0,
    }

    for index, score in enumerate(scores):

        score = float(score)

        if score < MIN_YAMNET_CONFIDENCE:
            continue

        label = class_names[index]

        for category, keywords in CATEGORY_KEYWORDS.items():

            if label_matches(label, keywords):

                category_scores[category] = max(
                    category_scores[category],
                    score
                )

    return category_scores


# ============================================================
# YAMNET CLASSIFICATION
# ============================================================

def classify_audio(audio):

    waveform = np.asarray(audio, dtype=np.float32)

    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Average predictions across all YAMNet frames
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Application categories
    categories = map_to_categories(mean_scores)

    # Top raw YAMNet predictions
    top_indices = np.argsort(mean_scores)[::-1][:5]

    raw_predictions = []

    for index in top_indices:
        raw_predictions.append(
            (
                class_names[index],
                float(mean_scores[index])
            )
        )

    return categories, raw_predictions


# ============================================================
# DISPLAY
# ============================================================

def print_detection(categories, raw_predictions):

    active = [
        (category, confidence)
        for category, confidence in categories.items()
        if confidence >= DISPLAY_THRESHOLD
    ]

    active.sort(
        key=lambda item: item[1],
        reverse=True
    )

    print("\n" + "-" * 65)

    if active:

        print("DETECTED SOUNDS\n")

        for category, confidence in active:

            name = CATEGORY_ICONS.get(category, category)

            percentage = confidence * 100

            bars = int(percentage / 5)

            meter = "#" * bars
            meter = meter.ljust(20, "-")

            print(
                f"{name:<10} "
                f"[{meter}] "
                f"{percentage:5.1f}%"
            )

    else:

        print("Listening... no mapped sound detected.")

    print("\nTop YAMNet predictions:")

    for label, confidence in raw_predictions:

        print(
            f"  {label:<35} "
            f"{confidence * 100:5.1f}%"
        )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print("\nMicrophone ready.")
    print()
    print("Try:")
    print("  - speaking")
    print("  - dog barking")
    print("  - vehicle sounds")
    print("  - alarms / sirens")
    print("  - clapping / knocking")
    print()
    print("Press CTRL+C to stop.")
    print()

    try:

        while True:

            audio = sd.rec(
                CHUNK_SIZE,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = audio.flatten()

            categories, raw_predictions = classify_audio(audio)

            print_detection(
                categories,
                raw_predictions
            )

    except KeyboardInterrupt:

        print("\n\nStopping SoundScript...")
        print("Microphone stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()