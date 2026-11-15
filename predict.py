"""Classify one WAV recording with the supplied CPU model."""
import argparse

import torch

from audio_features import LABELS, ROOT, extract_mfcc
from model import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", default=ROOT / "demo/03-01-05-02-02-01-08.wav")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        mfcc = torch.from_numpy(extract_mfcc(args.wav))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Could not read the WAV file: {error}\n")
    checkpoint = torch.load(ROOT / "outputs/speech_cnn.pt", map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    with torch.inference_mode():
        features = (mfcc - checkpoint["mean"]) / checkpoint["std"]
        scores = model(features).softmax(dim=1)[0]
    ranked_scores, indices = scores.topk(3)
    print(f"Audio: {args.wav}")
    print(f"Predicted expression: {LABELS[indices[0]]}")
    print("Top 3 model scores:")
    for score, index in zip(ranked_scores, indices):
        print(f"  {LABELS[index]:10s} {score:.1%}")


if __name__ == "__main__":
    main()
