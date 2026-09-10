import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa
import tensorflow_hub as hub
import whisper
import csv
import os


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
DURATION = 5

AUDIO_FILE = "current_audio.wav"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

CLASS_MAP_FILE = "yamnet_class_map.csv"

CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)


# ============================================================
# LOAD YAMNET
# ============================================================

print("Loading YAMNet...")

yamnet = hub.load(YAMNET_URL)

print("✅ YAMNet loaded!")


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper...")

whisper_model = whisper.load_model("base")

print("✅ Whisper loaded!")


# ============================================================
# DOWNLOAD CLASS MAP
# ============================================================

def download_class_map():

    if os.path.exists(CLASS_MAP_FILE):
        return

    import urllib.request

    print("Downloading YAMNet class names...")

    urllib.request.urlretrieve(
        CLASS_MAP_URL,
        CLASS_MAP_FILE
    )

    print("✅ Class names downloaded!")


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
            class_names.append(row["display_name"])

    return class_names


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio():

    print("\n🎙️ Recording...")
    print("Speak or make a sound!")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("✅ Recording finished.")

    audio_int16 = np.int16(audio * 32767)

    wav.write(
        AUDIO_FILE,
        SAMPLE_RATE,
        audio_int16
    )


# ============================================================
# CLASSIFY SOUND
# ============================================================

def classify_sound():

    audio, _ = librosa.load(
        AUDIO_FILE,
        sr=16000,
        mono=True
    )

    audio = audio.astype(np.float32)

    scores, embeddings, spectrogram = yamnet(audio)

    scores = scores.numpy()

    mean_scores = np.mean(
        scores,
        axis=0
    )

    top_indices = np.argsort(
        mean_scores
    )[::-1][:10]

    results = []

    for index in top_indices:

        label = class_names[index]

        confidence = mean_scores[index]

        results.append(
            (label, confidence)
        )

    return results


# ============================================================
# TRANSCRIBE SPEECH
# ============================================================

def transcribe():

    print("\n🗣️ Transcribing...")

    result = whisper_model.transcribe(
        AUDIO_FILE,
        fp16=False
    )

    return result["text"].strip()


# ============================================================
# MAIN
# ============================================================

download_class_map()

class_names = load_class_names()


while True:

    print("\n")
    print("=" * 55)
    print("        🔊 SOUNDSCRIPT AUDIO DETECTOR")
    print("=" * 55)

    record_audio()

    results = classify_sound()

    print("\n🔊 DETECTED SOUNDS")
    print("-" * 55)

    for label, confidence in results[:5]:

        print(
            f"{label:35s} "
            f"{confidence * 100:6.2f}%"
        )


    primary_label = results[0][0]
    primary_confidence = results[0][1]


    print("\n🎯 PRIMARY SOUND")
    print("-" * 55)

    print(
        f"{primary_label} "
        f"({primary_confidence * 100:.2f}%)"
    )


    # --------------------------------------------------------
    # Speech detection
    # --------------------------------------------------------

    speech_detected = False

    for label, confidence in results:

        if "Speech" in label:

            if confidence >= 0.20:

                speech_detected = True

                break


    # --------------------------------------------------------
    # Whisper
    # --------------------------------------------------------

    if speech_detected:

        text = transcribe()

        print("\n📝 TRANSCRIPT")
        print("-" * 55)

        if text:

            print(text)

        else:

            print("(No speech detected)")


    print("\n")

    command = input(
        "Press ENTER to record again "
        "or type Q to quit: "
    )

    if command.lower() == "q":

        break


print("\n👋 SoundScript stopped.")