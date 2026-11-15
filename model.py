"""Compact temporal CNN for MFCC sequences."""
from torch import nn

from audio_features import LABELS, N_MFCC


def make_model():
    return nn.Sequential(
        nn.Conv1d(N_MFCC, 32, 5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(32, 64, 5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(4),
        nn.Conv1d(64, 96, 3, padding=1),
        nn.BatchNorm1d(96),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(4),
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(96 * 4, len(LABELS)),
    )
