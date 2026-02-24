"""
Konfuziona matrica za sentiment model: učitava model i izabrani split (train/val/test),
prikuplja predikcije i crta/snima konfuzionu matricu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_transformer

# Import iz train skripte za isti dataset i tokenizaciju
from amazon_sentiment.train_amazon_sentiment import (
    DATASET_NAME,
    LABEL_COLUMN,
    SOURCE_COLUMN,
    filter_too_long,
    get_row_value,
    load_amazon_splits,
    mean_pool,
    SentimentDataset,
)

RUN_DIR = Path("runs/amazon_sentiment")


def find_latest_checkpoint(weights_dir: Path) -> Path | None:
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(weights_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Konfuziona matrica za sentiment model")
    parser.add_argument("--run-dir", default=RUN_DIR, type=Path, help="Putanja do run foldera")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Koji split koristiti (train/val/test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maksimalan broj uzoraka (default: svi za izabrani split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Fajl za snimljenu sliku (default: run_dir/confusion_matrix_<split>.png)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Samo nacrtaj matricu iz prethodno sačuvanog JSON-a (bez učitavanja modela i predikcije).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()

    labels_names = ["negative", "positive"]
    cm_json_path = run_dir / f"confusion_matrix_{args.split}.json"

    if args.plot_only:
        if not cm_json_path.exists():
            raise FileNotFoundError(
                f"Nema sačuvane matrice: {cm_json_path}. Prvo pokreni skriptu bez --plot-only."
            )
        cm = json.loads(cm_json_path.read_text(encoding="utf-8"))
        cm = [[int(x) for x in row] for row in cm]
        print("Učitana matrica iz", cm_json_path)
        print("              pred_neg  pred_pos")
        for i, name in enumerate(labels_names):
            print(f"  {name:10}  {cm[i][0]:8}  {cm[i][1]:8}")
        print("  Ukupno:", sum(sum(row) for row in cm), "uzoraka")
    else:
        cfg_path = run_dir / "run_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Nema run_config.json u {run_dir}. Prvo pokreni trening.")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        context_size = int(cfg["context_size"])

        tok_path = run_dir / "tokenizers" / "tokenizer_src.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"Nema tokenizer u {run_dir}.")
        tokenizer = Tokenizer.from_file(str(tok_path))

        weights_dir = run_dir / "weights"
        ckpt_path = find_latest_checkpoint(weights_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"Nema checkpoint-a u {run_dir}.")
        print("Checkpoint:", ckpt_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = tokenizer.get_vocab_size()
        model = build_transformer(
            source_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
            source_context_size=context_size,
            target_context_size=context_size,
            model_dimension=int(cfg["model_dimension"]),
            number_of_blocks=int(cfg["number_of_blocks"]),
            heads=int(cfg["heads"]),
            dropout=float(cfg["dropout"]),
            feed_forward_dimension=int(cfg["feed_forward_dimension"]),
        ).to(device)
        classifier = nn.Linear(int(cfg["model_dimension"]), 2).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        model.eval()
        classifier.eval()

        max_train = int(cfg.get("max_train_samples", 80_000))
        max_val = int(cfg.get("max_val_samples", 5_000))
        max_test = int(cfg.get("max_test_samples", 5_000))
        train_raw, val_raw, test_raw = load_amazon_splits(max_train, max_val, max_test)

        if args.split == "train":
            raw = train_raw
        elif args.split == "val":
            raw = val_raw
        else:
            raw = test_raw

        if args.max_samples is not None:
            raw = raw.select(range(min(args.max_samples, len(raw))))
        raw = filter_too_long(raw, tokenizer, SOURCE_COLUMN, context_size)
        dataset = SentimentDataset(raw, tokenizer, SOURCE_COLUMN, LABEL_COLUMN, context_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predikcija"):
                enc_in = batch["encoder_input"].to(device)
                enc_mask = batch["encoder_mask"].to(device)
                labels = batch["label"].to(device).long().view(-1)
                enc_out = model.encode(enc_in, enc_mask)
                pooled = mean_pool(enc_out, enc_mask)
                logits = classifier(pooled)
                pred = logits.argmax(dim=1).cpu().tolist()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred)

        cm = confusion_matrix(y_true, y_pred)
        print("\nKonfuziona matrica (red = stvarno, kolona = predviđeno):")
        print("              pred_neg  pred_pos")
        for i, name in enumerate(labels_names):
            print(f"  {name:10}  {cm[i, 0]:8}  {cm[i, 1]:8}")
        print("  Ukupno:", cm.sum(), "uzoraka")

        cm_json_path.parent.mkdir(parents=True, exist_ok=True)
        cm_json_path.write_text(
            json.dumps([[int(cm[i, j]) for j in range(cm.shape[1])] for i in range(cm.shape[0])]),
            encoding="utf-8",
        )
        print("Matrica sačuvana u", cm_json_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nije instaliran. Instaliraj: pip install matplotlib")
        return

    import numpy as np
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="equal", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_names)
    ax.set_yticklabels(labels_names)
    ax.set_xlabel("Predviđeno")
    ax.set_ylabel("Stvarno")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    plt.colorbar(im, ax=ax, label="Broj uzoraka")
    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = run_dir / f"confusion_matrix_{args.split}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Snimljeno:", out_path)


if __name__ == "__main__":
    main()
