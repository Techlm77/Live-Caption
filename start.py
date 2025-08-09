#!/usr/bin/env python3
import argparse
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
import april_asr as april
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

def resample_linear_int16(mono_int16: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or mono_int16.size == 0:
        return mono_int16
    x = mono_int16.astype(np.float32)
    ratio = dst_sr / float(src_sr)
    dst_len = max(1, int(round(len(x) * ratio)))
    if dst_len <= 1 or len(x) <= 1:
        return np.zeros(0, dtype=np.int16)
    src_idx = np.linspace(0, len(x) - 1, num=dst_len, dtype=np.float32)
    idx_floor = np.floor(src_idx).astype(np.int32)
    idx_ceil = np.minimum(idx_floor + 1, len(x) - 1)
    frac = src_idx - idx_floor
    y = x[idx_floor] * (1.0 - frac) + x[idx_ceil] * frac
    return np.clip(y, -32768, 32767).astype(np.int16)

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

    def feed_i16(self, data_i16: np.ndarray, src_sr: int):
        if data_i16.size == 0:
            return
        out = resample_linear_int16(data_i16, src_sr, self.model_sr)
        if out.size:
            self.session.feed_pcm16(out.tobytes())

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
        try:
            import ctypes
            ole32 = ctypes.windll.ole32
            COINIT_MULTITHREADED = 0x0
            ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        except Exception:
            pass

    def _com_uninit(self):
        try:
            import ctypes
            ctypes.windll.ole32.CoUninitialize()
        except Exception:
            pass

    def stop(self):
        self._stop_flag.set()

    def run(self):
        if platform.system() != "Windows":
            log.info("Output loopback is Windows-only; skipping.")
            return

        self._com_init()
        try:
            backoff = 0.5
            max_backoff = 5.0
            attempts = 0
            max_attempts = 20

            while not self._stop_flag.is_set():
                try:
                    sp = sc.default_speaker()
                    if sp is None:
                        raise RuntimeError("No default speaker reported by system.")
                    mic = sc.get_microphone(id=str(sp.id), include_loopback=True)
                    log.info("Capturing output (loopback) from: %s", sp.name)
                    with mic.recorder(samplerate=self.samplerate, blocksize=self.blocksize, channels=2) as rec:
                        backoff = 0.5
                        attempts = 0
                        while not self._stop_flag.is_set():
                            data = rec.record(self.blocksize)
                            if data.ndim == 2 and data.shape[1] > 1:
                                data = data.mean(axis=1)
                            else:
                                data = data.reshape(-1)
                            data_i16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
                            self.april.feed_i16(data_i16, self.samplerate)
                except Exception as e:
                    if self._stop_flag.is_set():
                        break
                    attempts += 1
                    log.warning("Loopback capture error (%s). Retrying in %.1fs (attempt %d/%d)...",
                                e, backoff, attempts, max_attempts)
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 1.5)
                    if attempts >= max_attempts:
                        log.error("Loopback retries exceeded. Stopping loopback worker.")
                        break
        finally:
            self._com_uninit()

class SDInputWorker(threading.Thread):
    def __init__(self, april_sess: AprilSession, device: Optional[int]):
        super().__init__(daemon=True)
        self.april = april_sess
        self.device = device
        self.stream: Optional[sd.InputStream] = None

    def run(self):
        _devinfo = None
        if self.device is not None:
            try:
                _devinfo = sd.query_devices(self.device)
            except Exception:
                _devinfo = None
        if _devinfo is None:
            default_idx = sd.default.device[0]
            if default_idx is not None and default_idx >= 0:
                try:
                    _devinfo = sd.query_devices(default_idx)
                    self.device = default_idx
                except Exception:
                    _devinfo = None
        if _devinfo is None or _devinfo.get("max_input_channels", 0) <= 0:
            log.error("No usable input device for mic mode.")
            return

        device_sr = int(_devinfo.get("default_samplerate", 48000) or 48000)
        extra_settings = None
        if platform.system() == "Windows":
            try:
                hostapi_name_l = sd.query_hostapis(_devinfo['hostapi'])['name'].lower()
                if "wasapi" in hostapi_name_l:
                    extra_settings = sd.WasapiSettings(exclusive=False, auto_convert=True)
            except Exception:
                extra_settings = None

        def callback(indata, _frames, _time_info, _status):
            data = indata
            if data.ndim == 2 and data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data.reshape(-1)
            data_i16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
            self.april.feed_i16(data_i16, device_sr)

        try:
            self.stream = sd.InputStream(
                samplerate=device_sr,
                blocksize=0,
                channels=min(2, _devinfo["max_input_channels"]),
                dtype="float32",
                callback=callback,
                device=self.device,
                extra_settings=extra_settings,
            )
            self.stream.start()
            log.info("Mic capture started: %s", _devinfo.get("name", "Unknown input"))
            while self.stream:
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

        self._padx = 16
        self._pady = 12

    def queue_msg(self, msg: dict):
        self._queue.put(msg)

    def _enable_dark_titlebar(self, tk_root):
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            from ctypes import wintypes
            hwnd = tk_root.winfo_id()
            TRUE = ctypes.c_int(1)
            for attr in (20, 19):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    wintypes.HWND(hwnd),
                    wintypes.DWORD(attr),
                    ctypes.byref(TRUE),
                    ctypes.sizeof(TRUE),
                )
                break
        except Exception:
            pass

    def _on_configure(self, event):
        try:
            inner_w = max(10, event.width - (self._padx * 2))
            if self.label:
                self.label.configure(wraplength=inner_w)
        except Exception:
            pass

    def _set_dpi_awareness(self):
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

    def start(self, on_close):
        self._set_dpi_awareness()

        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.bg_color)
        self._enable_dark_titlebar(self.root)
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
        self.root.resizable(True, True)

        min_h = max(80, self.height_px // 2)
        self.root.minsize(320, min_h)

        fnt = tkfont.Font(family=self.font_name, size=self.font_size)
        self.text_var = tk.StringVar(self.root, value="")

        self.label = tk.Label(
            self.root,
            textvariable=self.text_var,
            bg=self.bg_color, fg=self.fg_color,
            justify="left", anchor="sw",
            wraplength=ww - (self._padx * 2),
            font=fnt,
            bd=0, relief="flat", highlightthickness=0,
        )
        self.label.pack(fill="both", expand=True, padx=self._padx, pady=self._pady)

        self.root.bind("<Configure>", self._on_configure)

        def poll():
            try:
                while True:
                    msg = self._queue.get_nowait()
                    mtype = msg.get("type")
                    text = msg.get("text", "")
                    if mtype == "final":
                        self._final_prefix += (text.strip() + " ")
                        self._live_text = ""
                    elif mtype == "partial":
                        self._live_text = text
                    combined = (self._final_prefix + self._live_text).strip()
                    self.text_var.set(combined)
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
                self.root.update_idletasks()
                self.root.destroy()
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Windows Live Captions (April ASR)")
    parser.add_argument("--model", default="april-english-dev-01110_en.april", help="Path to .april model")
    parser.add_argument("--mic", action="store_true", help="Use microphone/capture device instead of output loopback")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input index (only with --mic)")
    parser.add_argument("--font", default="Segoe UI", help="Font family")
    parser.add_argument("--font-size", type=int, default=24, help="Caption font size")
    parser.add_argument("--opacity", type=float, default=0.96, help="Window opacity 0-1")
    parser.add_argument("--raw-case", action="store_true", help="Do NOT auto-fix ALL CAPS to sentence case")
    parser.add_argument("--history-file", default="history.txt", help="Where to save final captions")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        log.error("Model not found: %s", args.model)
        sys.exit(1)

    ui = CaptionUI(font_name=args.font, font_size=args.font_size, opacity=args.opacity)

    logger = HistoryLogger(args.history_file)
    april_sess = AprilSession(args.model, ui_q := ui._queue, logger=logger, raw_case=args.raw_case)

    loop_worker: Optional[OutputLoopbackWorker] = None
    mic_worker: Optional[SDInputWorker] = None

    if args.mic:
        mic_worker = SDInputWorker(april_sess, args.device)
        mic_worker.start()
    else:
        loop_worker = OutputLoopbackWorker(april_sess, blocksize=2048)
        loop_worker.start()

    log.info("Live Captions running. Resize window width/height as you like. Close with Esc or ❌.")

    def on_close():
        try:
            if loop_worker:
                loop_worker.stop()
                loop_worker.join(timeout=1.5)
            april_sess.flush()
        finally:
            ui.stop()
            log.info("Live Captions stopped.")

    ui.start(on_close)

    if loop_worker:
        loop_worker.join(timeout=1.0)
    if mic_worker:
        mic_worker.join(timeout=1.0)

if __name__ == "__main__":
    main()
