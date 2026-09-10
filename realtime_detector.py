import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import csv
import os
import urllib.request
import time


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# YAMNet works well with roughly 0.975 second windows
CHUNK_DURATION = 0.975

CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

CLASS_MAP_FILE = "yamnet_class_map.csv"

CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)


# ============================================================
# DOWNLOAD CLASS MAP
# ============================================================

def download_class_map():

    if os.path.exists(CLASS_MAP_FILE):
        return

    print("Downloading YAMNet class names...")

    urllib.request.urlretrieve(
        CLASS_MAP_URL,
        CLASS_MAP_FILE
    )

    print("Class names downloaded!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    class_names = []

    with open(
        CLASS_MAP_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            class_names.append(
                row["display_name"]
            )

    return class_names


# ============================================================
# LOAD YAMNET
# ============================================================

print("\n🔄 Loading YAMNet...")

model = hub.load(YAMNET_URL)

print("✅ YAMNet loaded!")


download_class_map()

class_names = load_class_names()

print(
    f"✅ Loaded {len(class_names)} sound classes."
)


# ============================================================
# CLASSIFY AUDIO CHUNK
# ============================================================

def classify_audio(audio):

    audio = audio.astype(
        np.float32
    )

    scores, embeddings, spectrogram = model(
        audio
    )

    scores = scores.numpy()

    # Average predictions over frames
    mean_scores = np.mean(
        scores,
        axis=0
    )

    top_indices = np.argsort(
        mean_scores
    )[::-1][:5]

    return [
        (
            class_names[index],
            mean_scores[index]
        )
        for index in top_indices
    ]


# ============================================================
# MAIN REAL-TIME LOOP
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("        🔊 SOUNDSCRIPT REAL-TIME DETECTOR")
    print("=" * 60)

    print("\n🎙️ Listening...")
    print("Make sounds around the microphone.")
    print("Press CTRL+C to stop.\n")


    try:

        while True:

            # -----------------------------------------------
            # Record one chunk
            # -----------------------------------------------

            audio = sd.rec(
                CHUNK_SIZE,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            audio = audio.flatten()


            # -----------------------------------------------
            # Classify
            # -----------------------------------------------

            results = classify_audio(
                audio
            )


            # -----------------------------------------------
            # Display
            # -----------------------------------------------

            timestamp = time.strftime(
                "%H:%M:%S"
            )

            print(
                f"\n[{timestamp}] 🔊"
            )

            print("-" * 45)

            for label, confidence in results:

                print(
                    f"{label:32s}"
                    f"{confidence * 100:6.1f}%"
                )


    except KeyboardInterrupt:

        print("\n\n🛑 Detector stopped.")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()