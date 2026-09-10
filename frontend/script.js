let socket = null;
let listening = false;
let connectionAttempt = 0;

const $ = (id) => document.getElementById(id);

function setConnection(state, text) {
    const dot = $("connectionDot");
    const label = $("connectionText");

    dot.classList.remove("connected", "error");

    if (state === "connected") {
        dot.classList.add("connected");
    } else if (state === "error") {
        dot.classList.add("error");
    }

    label.textContent = text;
    $("statusText").textContent =
        state === "connected" ? "System Active" :
        state === "error" ? "Connection Error" :
        "System Ready";
}

function startListening() {
    if (listening) return;

    listening = true;
    const attempt = ++connectionAttempt;

    $("startButton").disabled = true;
    $("stopButton").disabled = false;
    $("currentSound").textContent = "Listening...";
    $("timestamp").textContent = "Connecting to microphone...";
    $("statusText").textContent = "Starting microphone";

    document.body.classList.add("listening");

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;

    try {
        socket = new WebSocket(`${protocol}//${host}/ws`);

        socket.onopen = () => {
            if (attempt !== connectionAttempt) return;

            setConnection("connected", "Microphone is listening");
            $("timestamp").textContent = "Microphone input active";
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateDetection(data);
            } catch (error) {
                console.error("Invalid server data:", error);
            }
        };

        socket.onclose = () => {
            if (listening && attempt === connectionAttempt) {
                resetAfterStop("Could not connect to the microphone");
            }
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);

            if (attempt === connectionAttempt) {
                setConnection("error", "Unable to connect");
            }
        };
    } catch (error) {
        console.error("WebSocket could not start:", error);
        resetAfterStop("Could not start the connection");
    }
}

function stopListening() {
    if (!listening) return;

    listening = false;
    connectionAttempt += 1;
    document.body.classList.remove("listening");

    if (socket) {
        try {
            socket.close();
        } catch (error) {
            console.error("WebSocket close error:", error);
        }

        socket = null;
    }

    resetAfterStop();
}

function resetAfterStop(errorMessage = "") {
    listening = false;
    document.body.classList.remove("listening");

    $("startButton").disabled = false;
    $("stopButton").disabled = true;

    $("currentSound").textContent = "Waiting...";
    $("confidence").textContent = "0";
    $("confidenceBar").style.width = "0%";
    $("timestamp").textContent =
        errorMessage || "Microphone input waiting for audio...";

    $("transcript").innerHTML =
        '<span class="quote-mark">“</span><p>Speech detected from the microphone will appear here.</p>';

    resetSoundBars();

    setConnection(
        errorMessage ? "error" : "ready",
        errorMessage || "Ready to listen"
    );
}

function updateDetection(data) {
    const sound = data.sound || "Unknown sound";
    const confidence = Number(data.confidence || 0);
    const percentage = Math.max(0, Math.min(100, confidence * 100));

    $("currentSound").textContent = sound;
    $("confidence").textContent = percentage.toFixed(0);
    $("confidenceBar").style.width = `${percentage}%`;
    $("currentIcon").textContent = getSoundIcon(sound);

    if (data.timestamp) {
        $("timestamp").textContent =
            `Detected at ${formatTime(new Date(data.timestamp * 1000))}`;
    } else {
        $("timestamp").textContent = "Sound detected";
    }

    updateSoundList(data.sounds || {});

    if (data.transcript && data.transcript.trim()) {
        $("transcript").innerHTML =
            `<span class="quote-mark">“</span><p>${escapeHtml(data.transcript.trim())}</p>`;
    }

    addEvent(sound, percentage, data.transcript || "");
}

function updateSoundList(sounds) {
    updateSoundBar("speech", sounds.speech);
    updateSoundBar("dog", sounds.dog);
    updateSoundBar("vehicle", sounds.vehicle);
    updateSoundBar("alarm", sounds.alarm);
}

function updateSoundBar(name, value) {
    const number = Math.max(0, Math.min(100, Number(value || 0) * 100));

    const bar = $(`${name}Bar`);
    const label = $(`${name}Value`);

    if (bar) bar.style.width = `${number.toFixed(1)}%`;
    if (label) label.textContent = `${number.toFixed(0)}%`;
}

function resetSoundBars() {
    ["speech", "dog", "vehicle", "alarm"].forEach((name) => {
        updateSoundBar(name, 0);
    });
}

function addEvent(sound, confidence, transcript) {
    const container = $("events");

    const empty = container.querySelector(".empty-events");
    if (empty) empty.remove();

    const item = document.createElement("div");
    item.className = "event-item";

    const safeSound = escapeHtml(sound);
    const safeTranscript = transcript ?
        escapeHtml(transcript) :
        "Sound detected";

    item.innerHTML = `
        <div class="event-icon">${getSoundIcon(sound)}</div>
        <div class="event-main">
            <strong>${safeSound}</strong>
            <span>${safeTranscript}</span>
        </div>
        <div class="event-time">${formatTime(new Date())}<br>${confidence.toFixed(0)}%</div>
    `;

    container.prepend(item);

    while (container.children.length > 10) {
        container.removeChild(container.lastElementChild);
    }
}

function clearEvents() {
    $("events").innerHTML =
        '<div class="empty-events">No events detected yet.</div>';
}

function getSoundIcon(sound) {
    const label = String(sound).toLowerCase();

    if (
        label.includes("speech") ||
        label.includes("voice") ||
        label.includes("conversation") ||
        label.includes("sing")
    ) {
        return "🎙️";
    }

    if (label.includes("dog") || label.includes("bark")) {
        return "🐕";
    }

    if (
        label.includes("car") ||
        label.includes("vehicle") ||
        label.includes("engine") ||
        label.includes("truck") ||
        label.includes("bus") ||
        label.includes("motor")
    ) {
        return "🚗";
    }

    if (
        label.includes("alarm") ||
        label.includes("siren") ||
        label.includes("beep") ||
        label.includes("buzzer")
    ) {
        return "🚨";
    }

    return "🔊";
}

function formatTime(date) {
    return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

$("startButton").addEventListener("click", startListening);
$("stopButton").addEventListener("click", stopListening);

window.addEventListener("beforeunload", () => {
    if (socket) {
        try {
            socket.close();
        } catch (_) {}
    }
});

setConnection("ready", "Ready to listen");