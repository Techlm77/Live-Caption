# Live Captions (Windows, AprilASR)

Fast, simple, and accurate **live captions** on Windows using WASAPI loopback + [AprilASR]. Also supports **microphone mode**. Minimal UI, resizable window, and an optional history log.

> Press **Esc** or close the window to stop.

---

## Features

- 🔊 **Default: output loopback** – captions your current Windows output while you keep listening normally  
- 🎙️ **Mic mode** – caption from any microphone or capture device (`--mic`)  
- 🪟 **Clean overlay** – bottom-aligned single-label window, **width & height resizable**, auto-wrap  
- 🧠 **Low-latency streaming** – AprilASR async session; feeds ~10–40 ms audio blocks  
- 🗂️ **History log** – final lines saved to `history.txt` for later reading  
- 🧱 **Robust loopback** – retries on device changes, handles default output switches  
- 🖥️ **No VB-Cable needed** – uses native WASAPI loopback for system audio  
- ⚙️ Sensible CPU thread defaults (via env vars), overridable by users if needed

---

## Requirements

- **OS:** Windows 10/11 for output loopback (Mic mode also works on Linux/macOS, but loopback is Windows-only for now)
- **Python:** 3.10+
- **Model:** an AprilASR `.april` model file (e.g. `april-english-dev-01110_en.april`)

### Python packages

Create `requirements.txt`:

```
april-asr
onnxruntime
numpy
sounddevice
soundcard
cffi
comtypes ; platform_system == "Windows"
```

Install:

```bash
pip install -r requirements.txt
```

> `tkinter` ships with the standard Python installer on Windows.  
> On Linux you’d install `python3-tk` via your package manager.

---

## Get a Model

Download an AprilASR **`.april`** model and put it next to `start.py`, or pass the path with `--model`.  
(See the AprilASR docs/releases for available English or multilingual models.)

---

## Usage

### 1) Caption “what you hear” (default, Windows)

```bash
python start.py
```

- Captures the **current default Windows output** (speakers/headset) via WASAPI loopback.
- You keep hearing audio normally; this just taps it for captions.

### 2) Microphone mode

```bash
python start.py --mic
```

Optionally target a specific input device index:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
python start.py --mic --device 1
```

### 3) Common options

```bash
--model PATH         # .april model path (default: april-english-dev-01110_en.april)
--font "Segoe UI"    # font family
--font-size 24       # size in points
--opacity 0.96       # 0..1 window opacity
--raw-case           # don’t auto-normalize ALL-CAPS lines
--history-file FILE  # final lines log (default: history.txt)
```

Examples:

```bash
# Larger font & higher opacity
python start.py --font-size 28 --opacity 0.98

# Mic mode with a specific device index
python start.py --mic --device 7

# Disable any capitalization normalization entirely
python start.py --raw-case
```

---

## How it Works (very short)

- Uses `soundcard` to open a **loopback recorder** on the **current default** Windows speaker (no routing hacks).
- Feeds **float32 → PCM16** mono audio frames (~10–40 ms) to an **asynchronous AprilASR session**.
- The session invokes your **handler** for partial & final results.  
  We display **partials immediately** and **append finals** to the window and `history.txt`.

---

## Tips & Troubleshooting

- **No captions / device changed?** Try switching your default output device in Windows once; the loopback will reconnect automatically, or just restart the app.
- **Mic mode silence?** Verify the right input index (`sd.query_devices()`), and that the device has input channels.
- **Soundcard warnings (MediaFoundation)** are filtered by default; they’re harmless data discontinuities.
- **Performance**  
  - The app sets sensible CPU thread env vars automatically.  
  - If you want to override: set `OMP_NUM_THREADS`, `ORT_NUM_THREADS`, etc. before running.
- **GPU?** AprilASR typically runs on ONNX Runtime CPU for stability. If GPU builds are available for your setup, you can experiment by switching the ORT provider—but this project targets CPU by default.

---

## Roadmap (nice-to-haves)

- Optional **per-app capture** (via Windows Audio Session APIs)  
- Toggleable **always-on-top** and quick **position presets**  
- **Latency/CPU** HUD for quick tuning

---

## Acknowledgments

- [AprilASR] for the streaming speech recognition engine  
- `soundcard` and `sounddevice` for straightforward audio I/O

---

**Enjoy!** If you hit any odd Windows audio edge cases, open an issue with your log output and `sd.query_devices()` dump.

[AprilASR]: https://abb128.github.io/april-asr/
