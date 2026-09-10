import whisper


print("Loading Whisper model...")

model = whisper.load_model("base")

print("Whisper loaded!")


def transcribe_audio(filename):

    print("\n🗣️ Transcribing...")

    result = model.transcribe(
        filename,
        fp16=False
    )

    text = result["text"].strip()

    print("\n📝 TRANSCRIPT")
    print("=" * 45)
    print(text)
    print("=" * 45)

    return text


if __name__ == "__main__":

    transcribe_audio("test.wav")