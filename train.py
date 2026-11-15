"""Train, validate and test on disjoint RAVDESS actors."""
import argparse
from io import BytesIO
import json
import re
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audio_features import LABELS, ROOT, extract_mfcc
from model import make_model

TEST = {3, 8, 15, 20}
VALIDATION = {5, 10, 17, 22}
TRAIN = set(range(1, 25)) - TEST - VALIDATION
FILENAME = re.compile(r"03-01-(0[1-8])-0[12]-0[12]-0[12]-(\d{2})\.wav")
OUTPUT = ROOT / "outputs"


def load_dataset(archive):
    features, labels, actors, filenames = [], [], [], []
    with ZipFile(archive) as data:
        for member in sorted(data.namelist()):
            name = member.rsplit("/", 1)[-1]
            match = FILENAME.fullmatch(name)
            if match is None:
                continue
            emotion, actor = map(int, match.groups())
            features.append(extract_mfcc(BytesIO(data.read(member))))
            labels.append(emotion - 1)
            actors.append(actor)
            filenames.append(name)
    counts = np.bincount(actors, minlength=25)
    if len(set(filenames)) != 1440 or list(counts) != [0] + [60] * 24:
        raise ValueError("Expected the original RAVDESS audio-only speech ZIP with 60 unique clips per actor.")
    return np.stack(features), np.array(labels), np.array(actors), np.array(filenames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=ROOT / "data/Audio_Speech_Actors_01-24.zip")
    parser.add_argument("--epochs", default=30, type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    OUTPUT.mkdir(exist_ok=True)
    print("Extracting MFCCs from RAVDESS speech recordings...", flush=True)
    x, y, actors, filenames = load_dataset(args.dataset)
    train = np.isin(actors, sorted(TRAIN))
    validation = np.isin(actors, sorted(VALIDATION))
    test = np.isin(actors, sorted(TEST))
    mean = x[train].mean(axis=(0, 2), keepdims=True)
    std = x[train].std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    x = torch.from_numpy((x - mean) / std)
    y_train = torch.from_numpy(y[train])
    batches = DataLoader(TensorDataset(x[train], y_train), batch_size=32, shuffle=True)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    counts = torch.bincount(y_train, minlength=len(LABELS))
    loss_function = nn.CrossEntropyLoss(weight=counts.sum() / (len(LABELS) * counts))
    best_accuracy = -1
    print(f"Train: {train.sum()} | Validation: {validation.sum()} | Test: {test.sum()}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch, targets in batches:
            optimizer.zero_grad()
            loss = loss_function(model(batch), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(x[validation]).argmax(dim=1).numpy()
        accuracy = accuracy_score(y[validation], predicted)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save({
                "weights": model.state_dict(),
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
            }, OUTPUT / "speech_cnn.pt")
        print(f"Epoch {epoch:02d} | validation accuracy {accuracy:.1%}", flush=True)

    checkpoint = torch.load(OUTPUT / "speech_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["weights"])
    with torch.inference_mode():
        predicted = model(x[test]).argmax(dim=1).numpy()
    report = classification_report(y[test], predicted, labels=list(range(len(LABELS))),
                                   target_names=LABELS, zero_division=0)
    (OUTPUT / "test_report.txt").write_text(report, encoding="utf-8")
    with (OUTPUT / "test_predictions.csv").open("w", encoding="utf-8") as stream:
        stream.write("file,actor,expected,predicted\n")
        for name, actor, expected, result in zip(filenames[test], actors[test], y[test], predicted):
            stream.write(f"{name},{actor},{LABELS[expected]},{LABELS[result]}\n")
    summary = {
        "selected_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "test_accuracy": accuracy_score(y[test], predicted),
        "train_actors": sorted(TRAIN),
        "validation_actors": sorted(VALIDATION),
        "test_actors": sorted(TEST),
        "split_sizes": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(figsize=(8, 6), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y[test], predicted, labels=list(range(len(LABELS))), display_labels=LABELS,
        xticks_rotation=35, cmap="Blues", colorbar=False, ax=axes,
    )
    axes.set_title("Speech emotion recognition on unseen speakers")
    figure.savefig(OUTPUT / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(report)
    print(f"Test accuracy: {summary['test_accuracy']:.1%}")


if __name__ == "__main__":
    main()
