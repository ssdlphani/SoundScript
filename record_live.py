import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT_FILE = "test.wav"

print("🎙️ Recording...")
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
    OUTPUT_FILE,
    SAMPLE_RATE,
    audio_int16
)

print(f"💾 Saved as {OUTPUT_FILE}")