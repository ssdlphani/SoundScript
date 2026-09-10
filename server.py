import asyncio
import csv
import os
import threading
import time
import urllib.request
from collections import deque

import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.975
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "yamnet_class_map.csv")


# ============================================================
# GLOBAL STATE
# ============================================================

microphone_lock = threading.RLock()
microphone_thread = None
microphone_running = False
microphone_stop_event = threading.Event()
microphone_generation = 0

audio_stream = None
audio_buffer = deque()
audio_buffer_lock = threading.Lock()
audio_ready_event = threading.Event()

connected_clients = set()
clients_lock = threading.Lock()
main_event_loop = None


# ============================================================
# LOAD MODELS AND CLASS MAP
# ============================================================

print("Loading YAMNet...")
yamnet_model = hub.load(YAMNET_URL)
print("YAMNet loaded.")

if not os.path.exists(CLASS_MAP_PATH):
    print("Downloading YAMNet class map...")
    urllib.request.urlretrieve(CLASS_MAP_URL, CLASS_MAP_PATH)
    print("Class map downloaded.")

class_names = []
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        class_names.append(row["display_name"])

print(f"Loaded {len(class_names)} YAMNet classes.")
print("Loading Whisper Base...")
whisper_model = whisper.load_model("base")
print("Whisper loaded.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="SoundScript")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ============================================================
# SOUND CLASSIFICATION
# ============================================================

def get_category(label):
    label_lower = label.lower()

    speech_words = [
        "speech", "conversation", "narration", "talking", "speaking",
        "human voice", "male speech", "female speech", "child speech",
        "whispering", "shout", "yell", "singing", "vocal",
    ]
    dog_words = ["dog", "bark", "bow-wow", "puppy", "canine", "whimper"]
    vehicle_words = [
        "car", "vehicle", "engine", "motorcycle", "motor vehicle", "truck",
        "bus", "train", "aircraft", "helicopter", "vehicle horn", "car horn",
        "road traffic", "tire", "skidding", "brake",
    ]
    alarm_words = [
        "alarm", "siren", "beep", "buzzer", "fire alarm", "smoke detector",
        "emergency",
    ]

    if any(word in label_lower for word in speech_words):
        return "speech"
    if any(word in label_lower for word in dog_words):
        return "dog"
    if any(word in label_lower for word in vehicle_words):
        return "vehicle"
    if any(word in label_lower for word in alarm_words):
        return "alarm"
    return "other"


def classify_audio(audio):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return []

    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    scores, _, _ = yamnet_model(audio)
    scores = scores.numpy()
    mean_scores = np.mean(scores, axis=0) if scores.ndim == 2 else scores
    top_indices = np.argsort(mean_scores)[::-1][:10]

    detections = []
    for index in top_indices:
        score = float(mean_scores[index])
        if score < 0.01 or index >= len(class_names):
            continue

        label = class_names[index]
        detections.append({
            "label": label,
            "confidence": score,
            "category": get_category(label),
        })

    return detections


def get_frontend_sounds(detections):
    result = {"speech": 0.0, "dog": 0.0, "vehicle": 0.0, "alarm": 0.0}

    for detection in detections:
        category = detection["category"]
        if category in result:
            result[category] = max(result[category], detection["confidence"])

    return result


def get_dashboard_detection(detections):
    """Choose a useful dashboard label instead of always showing raw Speech."""
    frontend_sounds = get_frontend_sounds(detections)
    display_labels = {
        "speech": "Speech",
        "dog": "Dog",
        "vehicle": "Vehicle",
        "alarm": "Alarm",
    }

    speech_score = frontend_sounds["speech"]
    non_speech_category = max(
        ("dog", "vehicle", "alarm"),
        key=lambda category: frontend_sounds[category],
    )
    non_speech_score = frontend_sounds[non_speech_category]

    # YAMNet commonly assigns Speech a broad score. Prefer a clearly detected
    # dashboard category when it has a meaningful score relative to Speech.
    if non_speech_score >= 0.08 and non_speech_score >= speech_score * 0.35:
        return {
            "label": display_labels[non_speech_category],
            "confidence": non_speech_score,
            "category": non_speech_category,
        }, frontend_sounds

    if speech_score >= 0.08:
        return {
            "label": "Speech",
            "confidence": speech_score,
            "category": "speech",
        }, frontend_sounds

    return detections[0], frontend_sounds


def transcribe(audio):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) == 0:
        return ""

    try:
        result = whisper_model.transcribe(
            audio,
            fp16=False,
            language="en",
            temperature=0,
            verbose=False,
        )
        return result.get("text", "").strip()
    except Exception as error:
        print(f"Whisper error: {error}")
        return ""


# ============================================================
# MICROPHONE
# ============================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    if microphone_stop_event.is_set():
        return

    try:
        samples = indata[:, 0].copy()
        with audio_buffer_lock:
            audio_buffer.extend(samples.tolist())
            if len(audio_buffer) >= CHUNK_SIZE:
                audio_ready_event.set()
    except Exception as error:
        print(f"Audio callback error: {error}")


def open_microphone():
    global audio_stream

    print("🎙️ Opening microphone...")
    with microphone_lock:
        if audio_stream is not None:
            return

        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=audio_callback,
        )
        audio_stream.start()

    print("🎙️ Microphone active")


def close_microphone():
    global audio_stream

    with microphone_lock:
        stream = audio_stream
        audio_stream = None

    if stream is not None:
        try:
            print("🛑 Closing microphone...")
            stream.stop()
        except Exception as error:
            print(f"Stream stop error: {error}")
        try:
            stream.close()
        except Exception as error:
            print(f"Stream close error: {error}")

    audio_ready_event.set()
    print("🎙️ Microphone closed")


def get_audio_chunk():
    while not microphone_stop_event.is_set():
        if audio_ready_event.wait(timeout=0.05):
            with audio_buffer_lock:
                if len(audio_buffer) >= CHUNK_SIZE:
                    samples = [audio_buffer.popleft() for _ in range(CHUNK_SIZE)]
                    if len(audio_buffer) < CHUNK_SIZE:
                        audio_ready_event.clear()
                    return np.asarray(samples, dtype=np.float32)
                audio_ready_event.clear()
    return None


def process_audio(audio):
    detections = classify_audio(audio)

    if not detections:
        return {
            "sound": "No sound detected",
            "confidence": 0,
            "detections": [],
            "sounds": {"speech": 0, "dog": 0, "vehicle": 0, "alarm": 0},
            "transcript": "",
            "timestamp": time.time(),
        }

    primary, frontend_sounds = get_dashboard_detection(detections)
    transcript = ""

    # Avoid slowing down non-speech detection with unnecessary Whisper work.
    if primary["category"] == "speech" and frontend_sounds["speech"] >= 0.20:
        transcript = transcribe(audio)

    return {
        "sound": primary["label"],
        "confidence": primary["confidence"],
        "detections": detections,
        "sounds": frontend_sounds,
        "transcript": transcript,
        "timestamp": time.time(),
    }


# ============================================================
# WEBSOCKET AND WORKER
# ============================================================

async def broadcast(data):
    dead_clients = []
    with clients_lock:
        clients = list(connected_clients)

    for client in clients:
        try:
            await client.send_json(data)
        except Exception:
            dead_clients.append(client)

    if dead_clients:
        with clients_lock:
            for client in dead_clients:
                connected_clients.discard(client)


def microphone_worker(generation):
    global microphone_running
    print("🎙️ Microphone worker started")

    try:
        open_microphone()

        while not microphone_stop_event.is_set():
            audio = get_audio_chunk()
            if audio is None or microphone_stop_event.is_set():
                break

            try:
                result = process_audio(audio)
            except Exception as error:
                print(f"Processing error: {error}")
                continue

            if microphone_stop_event.is_set() or generation != microphone_generation:
                break

            if main_event_loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(broadcast(result), main_event_loop)
                except Exception as error:
                    print(f"Broadcast error: {error}")
    except Exception as error:
        print(f"Microphone worker error: {error}")
    finally:
        close_microphone()
        with audio_buffer_lock:
            audio_buffer.clear()
        audio_ready_event.clear()

        with microphone_lock:
            if generation == microphone_generation:
                microphone_running = False

        print("🛑 Microphone worker stopped")


def start_microphone():
    global microphone_thread, microphone_running, microphone_generation

    with microphone_lock:
        if microphone_thread is not None and microphone_thread.is_alive():
            return

        microphone_generation += 1
        generation = microphone_generation
        microphone_stop_event.clear()

        with audio_buffer_lock:
            audio_buffer.clear()
        audio_ready_event.clear()

        microphone_running = True
        microphone_thread = threading.Thread(
            target=microphone_worker,
            args=(generation,),
            daemon=True,
        )
        microphone_thread.start()


def stop_microphone():
    global microphone_thread, microphone_running, microphone_generation

    print("🛑 Stop requested")
    with microphone_lock:
        microphone_generation += 1
        microphone_stop_event.set()
        audio_ready_event.set()
        thread = microphone_thread

    close_microphone()

    if thread is not None and thread.is_alive():
        thread.join(timeout=0.5)

    with microphone_lock:
        if thread is None or not thread.is_alive():
            microphone_thread = None
            microphone_running = False

    with audio_buffer_lock:
        audio_buffer.clear()
    audio_ready_event.clear()
    print("✅ Microphone stopped")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global main_event_loop

    await websocket.accept()
    main_event_loop = asyncio.get_running_loop()

    with clients_lock:
        connected_clients.add(websocket)

    print("🌐 WebSocket client connected")
    start_microphone()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("🌐 WebSocket client disconnected")
    except Exception as error:
        print(f"WebSocket error: {error}")
    finally:
        with clients_lock:
            connected_clients.discard(websocket)
            no_clients = len(connected_clients) == 0

        if no_clients:
            await asyncio.to_thread(stop_microphone)


@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    print("\n" + "=" * 55)
    print("        SoundScript Server")
    print("=" * 55)
    print("Server ready.")
    print("Open: http://127.0.0.1:8000")
    print("=" * 55 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 Server shutting down...")
    await asyncio.to_thread(stop_microphone)
    print("✅ Server stopped")
