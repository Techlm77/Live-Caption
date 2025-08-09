#!/usr/bin/env python3
import queue
import threading
import time
import sys
import os
import platform
import re
import warnings
import logging
from datetime import datetime
from typing import Optional

def _detect_threads() -> int:
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return max(1, min(n - 1, 12))

def _set_default_env():
    t = str(_detect_threads())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ORT_NUM_THREADS"):
        os.environ.setdefault(var, t)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("APRIL_ORT_PROVIDER", "cpu")

_set_default_env()

import numpy as np
import sounddevice as sd
import soundcard as sc

try:
    from soundcard.mediafoundation import SoundcardRuntimeWarning
    warnings.simplefilter("ignore", SoundcardRuntimeWarning)
except Exception:
    pass

import tkinter as tk
from tkinter import font as tkfont

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("livecaptions")

april = None

_LETTERS = re.compile(r"[A-Za-z]")
_SENT_SPLIT = re.compile(r"([.!?]+)(\s+|$)")

def _mostly_upper(text: str) -> bool:
    letters = _LETTERS.findall(text)
    if not letters:
        return False
    ups = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    return ups / max(1, len(letters)) >= 0.8

def _sentence_case(text: str) -> str:
    parts = _SENT_SPLIT.split(text.strip())
    out = []
    for i in range(0, len(parts), 3):
        chunk = parts[i] or ""
        end = parts[i+1] if i+1 < len(parts) else ""
        space = parts[i+2] if i+2 < len(parts) else ""
        chunk = chunk.strip().lower()
        if chunk:
            chunk = chunk[0].upper() + chunk[1:]
            chunk = re.sub(r"\bi\b", "I", chunk)
        sent = (chunk + end).strip()
        if sent:
            out.append(sent)
        if space:
            out.append(space)
    result = "".join(out).strip()
    return result or text

def normalize_caption(text: str, raw_case: bool) -> str:
    if raw_case:
        return text
    return _sentence_case(text) if _mostly_upper(text) else text

def resample_linear_f32(mono_f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or mono_f32.size == 0:
        return mono_f32
    ratio = dst_sr / float(src_sr)
    dst_len = max(1, int(round(len(mono_f32) * ratio)))
    if dst_len <= 1 or len(mono_f32) <= 1:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, len(mono_f32) - 1, num=dst_len, dtype=np.float32)
    idx_floor = np.floor(src_idx).astype(np.int32)
    idx_ceil = np.minimum(idx_floor + 1, len(mono_f32) - 1)
    frac = src_idx - idx_floor
    y = mono_f32[idx_floor] * (1.0 - frac) + mono_f32[idx_ceil] * frac
    return y.astype(np.float32, copy=False)

class HistoryLogger:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n\n=== Session start: %s ===\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            log.warning("Failed to init history file at %s", self.path, exc_info=True)

    def write_line(self, text: str):
        try:
            with self.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(text.rstrip() + "\n")
        except Exception:
            log.warning("Failed to write history line", exc_info=True)

class AprilSession:
    def __init__(self, model_path: str, ui_q: "queue.Queue[dict]", logger: HistoryLogger, raw_case: bool):
        global april
        if april is None:
            raise RuntimeError("april_asr not imported.")

        self.model = april.Model(model_path)
        self.model_sr = int(self.model.get_sample_rate())
        self.ui_q = ui_q
        self.logger = logger
        self.raw_case = raw_case

        def handler(result_type, tokens):
            s = "".join([t.token for t in tokens])
            s = normalize_caption(s, self.raw_case)
            if result_type == april.Result.PARTIAL_RECOGNITION:
                self.ui_q.put({"type": "partial", "text": s})
            elif result_type == april.Result.FINAL_RECOGNITION:
                line = s.strip()
                if line:
                    self.ui_q.put({"type": "final", "text": line})
                    self.logger.write_line(line)

        self.session = april.Session(self.model, handler, asynchronous=True)

    def feed_f32(self, data_f32: np.ndarray, src_sr: int):
        if data_f32.size == 0:
            return
        np.clip(data_f32, -1.0, 1.0, out=data_f32)
        y = resample_linear_f32(data_f32, src_sr, self.model_sr)
        if y.size:
            y_i16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
            self.session.feed_pcm16(y_i16.tobytes())

    def flush(self):
        try:
            self.session.flush()
        except Exception:
            log.warning("April flush failed", exc_info=True)

class OutputLoopbackWorker(threading.Thread):
    def __init__(self, april_sess: AprilSession, samplerate: int = 48000, blocksize: int = 2048):
        super().__init__(daemon=True)
        self.april = april_sess
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._stop_flag = threading.Event()

    def _com_init(self):
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            ole32 = ctypes.windll.ole32
            ole32.CoInitializeEx(None, 0x0)
        except Exception:
            pass

    def _com_uninit(self):
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass

    def stop(self):
        self._stop_flag.set()

    def run(self):
        self._com_init()
        try:
            while not self._stop_flag.is_set():
                try:
                    sp = sc.default_speaker()
                    if sp is None:
                        raise RuntimeError("No default speaker.")
                    mic = sc.get_microphone(id=str(sp.id), include_loopback=True)
                    log.info("Capturing output (loopback) from: %s", sp.name)
                    with mic.recorder(samplerate=self.samplerate, blocksize=self.blocksize, channels=2) as rec:
                        while not self._stop_flag.is_set():
                            data = rec.record(self.blocksize)
                            if data.ndim == 2 and data.shape[1] > 1:
                                data = data.mean(axis=1)
                            data = data.astype(np.float32, copy=False)
                            np.clip(data, -1.0, 1.0, out=data)
                            self.april.feed_f32(data, self.samplerate)
                except Exception as e:
                    log.warning("Loopback capture error: %s", e)
                    time.sleep(0.5)
        finally:
            self._com_uninit()

class MicInputWorker(threading.Thread):
    def __init__(self, april_sess: AprilSession, device: Optional[int] = None):
        super().__init__(daemon=True)
        self.april = april_sess
        self.device = device
        self.stream: Optional[sd.InputStream] = None
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

    def run(self):
        devinfo = None
        try:
            if self.device is not None:
                devinfo = sd.query_devices(self.device)
        except Exception:
            devinfo = None
        if devinfo is None:
            try:
                default_idx = sd.default.device[0]
                if default_idx is not None and default_idx >= 0:
                    devinfo = sd.query_devices(default_idx)
                    self.device = default_idx
            except Exception:
                devinfo = None

        if devinfo is None or devinfo.get("max_input_channels", 0) <= 0:
            log.error("No usable input device for mic mode.")
            return

        device_sr = int(devinfo.get("default_samplerate", 48000) or 48000)
        extra_settings = None
        if platform.system() == "Windows":
            try:
                hostapi_name_l = sd.query_hostapis(devinfo['hostapi'])['name'].lower()
                if "wasapi" in hostapi_name_l:
                    extra_settings = sd.WasapiSettings(exclusive=False, auto_convert=True)
            except Exception:
                extra_settings = None

        def callback(indata, _frames, _time_info, _status):
            if self._stop_flag.is_set():
                return
            data = indata
            if data.ndim == 2 and data.shape[1] > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32, copy=False)
            np.clip(data, -1.0, 1.0, out=data)
            self.april.feed_f32(data, device_sr)

        try:
            self.stream = sd.InputStream(
                samplerate=device_sr,
                blocksize=0,
                channels=min(2, devinfo["max_input_channels"]),
                dtype="float32",
                callback=callback,
                device=self.device,
                extra_settings=extra_settings,
            )
            self.stream.start()
            log.info("Mic capture started: %s", devinfo.get("name", "Unknown input"))
            while not self._stop_flag.is_set():
                time.sleep(0.2)
        except Exception:
            log.error("Failed to open mic stream.", exc_info=True)
        finally:
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass

class CaptionUI:
    def __init__(self, font_name="Segoe UI", font_size=24, opacity=0.96, width_pct=0.8, height_px=140):
        self.font_name = font_name
        self.font_size = font_size
        self.opacity = float(opacity)
        self.width_pct = width_pct
        self.height_px = height_px

        self.bg_color = "#101010"
        self.fg_color = "#F5F5F5"

        self.root: Optional[tk.Tk] = None
        self.text_var: Optional[tk.StringVar] = None
        self.label: Optional[tk.Label] = None

        self._final_prefix = ""
        self._live_text = ""
        self._queue: "queue.Queue[dict]" = queue.Queue()

    def start(self, on_close):
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.bg_color)
        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = int(sw * self.width_pct)
        wh = self.height_px
        x = int((sw - ww) / 2)
        y = sh - wh - 40
        self.root.geometry(f"{ww}x{wh}+{x}+{y}")

        fnt = tkfont.Font(family=self.font_name, size=self.font_size)
        self.text_var = tk.StringVar(self.root, value="")

        self.label = tk.Label(
            self.root,
            textvariable=self.text_var,
            bg=self.bg_color, fg=self.fg_color,
            justify="left", anchor="sw",
            wraplength=ww - 32,
            font=fnt,
            bd=0, relief="flat", highlightthickness=0,
        )
        self.label.pack(fill="both", expand=True, padx=16, pady=12)

        def _on_cfg(e):
            try:
                inner_w = max(10, e.width - 32)
                self.label.configure(wraplength=inner_w)
            except Exception:
                pass

        self.root.bind("<Configure>", _on_cfg)

        def poll():
            try:
                while True:
                    msg = self._queue.get_nowait()
                    mtype, text = msg.get("type"), msg.get("text", "")
                    if mtype == "final":
                        self._final_prefix += (text.strip() + " ")
                        self._live_text = ""
                    elif mtype == "partial":
                        self._live_text = text
                    self.text_var.set((self._final_prefix + self._live_text).strip())
            except queue.Empty:
                pass
            self.root.after(30, poll)

        self.root.after(30, poll)
        self.root.bind("<Escape>", lambda e: on_close())
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()

    def stop(self):
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass

def _auto_accel():
    provider_map = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    try:
        import onnxruntime as ort
        avail = list(ort.get_available_providers())
        log.info("ONNX Runtime v%s available providers: %s", getattr(ort, "__version__", "?"), avail)
        if platform.system() == "Windows":
            prefs = ["DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]
        elif platform.system() == "Linux":
            prefs = ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
        elif platform.system() == "Darwin":
            prefs = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            prefs = ["CPUExecutionProvider"]
        chosen = "CPUExecutionProvider"
        for ep in prefs:
            if ep in avail:
                chosen = ep
                break
        for k, v in provider_map.items():
            if v == chosen:
                os.environ["APRIL_ORT_PROVIDER"] = k
                break
        log.info("Using APRIL_ORT_PROVIDER=%s", os.environ.get("APRIL_ORT_PROVIDER", "cpu"))
    except Exception:
        os.environ["APRIL_ORT_PROVIDER"] = "cpu"
        log.info("Falling back to CPU")

def _probe_loopback(samplerate: int = 48000, blocksize: int = 512) -> bool:
    try:
        sp = sc.default_speaker()
        if sp is None:
            return False
        mic = sc.get_microphone(id=str(sp.id), include_loopback=True)
        with mic.recorder(samplerate=samplerate, blocksize=blocksize, channels=2) as rec:
            _ = rec.record(blocksize)
        return True
    except Exception:
        return False

def main():
    _auto_accel()

    global april
    try:
        import april_asr as april
    except Exception as e:
        log.error("Failed to import april_asr: %s. Is it installed and compatible with your Python/OS/ORT?", e)
        sys.exit(2)

    model_path = os.environ.get("APRIL_MODEL", "april-english-dev-01110_en.april")
    if not os.path.exists(model_path):
        log.error("Model not found: %s (override with APRIL_MODEL env var)", model_path)
        sys.exit(1)

    try:
        import onnxruntime as ort
        log.info("ONNX Runtime providers: %s", ort.get_available_providers())
    except Exception:
        pass
    log.info("Requested provider: %s (APRIL_ORT_PROVIDER)", os.environ.get("APRIL_ORT_PROVIDER"))

    ui = CaptionUI()
    logger = HistoryLogger("history.txt")
    april_sess = AprilSession(model_path, ui._queue, logger, raw_case=False)

    loop_worker: Optional[OutputLoopbackWorker] = None
    mic_worker: Optional[MicInputWorker] = None

    if _probe_loopback():
        loop_worker = OutputLoopbackWorker(april_sess)
        loop_worker.start()
        log.info("Audio source: system loopback")
    else:
        mic_worker = MicInputWorker(april_sess)
        mic_worker.start()
        log.info("Audio source: microphone (loopback not available)")

    log.info("Live Captions running. Close with Esc or ❌.")

    def on_close():
        try:
            if loop_worker:
                loop_worker.stop()
                loop_worker.join(timeout=1.5)
            if mic_worker:
                mic_worker.stop()
                mic_worker.join(timeout=1.5)
            april_sess.flush()
        finally:
            ui.stop()
            log.info("Live Captions stopped.")

    ui.start(on_close)

if __name__ == "__main__":
    main()
