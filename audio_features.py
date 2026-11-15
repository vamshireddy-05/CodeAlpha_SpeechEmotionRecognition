"""Shared WAV parsing and MFCC extraction for training and prediction."""

from math import gcd
from pathlib import Path
import warnings

import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
LABELS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")
RATE = 16000
SAMPLES = RATE * 4
FFT_SIZE = 512
WINDOW = 400
HOP = 160
N_MFCC = 20


def mel_filter_bank():
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
    mel_edges = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42)
    edges = 700 * (10 ** (mel_edges / 2595) - 1)
    left, center, right = (array[:, None] for array in (edges[:-2], edges[1:-1], edges[2:]))
    return np.maximum(0, np.minimum(
        (frequencies - left) / (center - left),
        (right - frequencies) / (right - center),
    )).astype(np.float32)


FILTER_BANK = mel_filter_bank()
WINDOW_WEIGHTS = np.hanning(WINDOW).astype(np.float32)


def extract_mfcc(source):
    """Return a fixed-size array with axes (MFCC coefficient, time frame)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Chunk \(non-data\) not understood, skipping it\.",
            category=wavfile.WavFileWarning,
        )
        rate, waveform = wavfile.read(source)
    if not waveform.size:
        raise ValueError("The WAV file contains no audio.")
    if waveform.dtype.kind in "iu":
        limits = np.iinfo(waveform.dtype)
        waveform = waveform.astype(np.float32)
        if limits.min == 0:
            waveform = (waveform - (limits.max + 1) / 2) / ((limits.max + 1) / 2)
        else:
            waveform /= -limits.min
    else:
        waveform = waveform.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = waveform[:rate * 4]
    if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) < 1e-6:
        raise ValueError("Use a WAV file containing audible speech.")
    if rate != RATE:
        divisor = gcd(rate, RATE)
        waveform = resample_poly(waveform, RATE // divisor, rate // divisor)
    waveform = waveform[:SAMPLES]
    waveform = np.pad(waveform, (0, SAMPLES - len(waveform)))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, WINDOW)[::HOP]
    spectrum = np.abs(np.fft.rfft(frames * WINDOW_WEIGHTS, n=FFT_SIZE)) ** 2
    log_mel = np.log(np.maximum(spectrum @ FILTER_BANK.T, 1e-10))
    return dct(log_mel, type=2, norm="ortho", axis=1)[:, :N_MFCC].T.astype(np.float32)
