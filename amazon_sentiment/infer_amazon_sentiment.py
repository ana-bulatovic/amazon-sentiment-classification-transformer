"""
Inferenca za sentiment (encoder + classifier): tekst recenzije → positive / negative.
Bez dekodera, samo logits i argmax.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import build_transformer


def mean_pool(encoder_output: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
    mask = encoder_mask.squeeze(1).unsqueeze(-1).float()
    sum_h = (encoder_output * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return sum_h / count


def find_latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(weights_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Nema run_config.json. Prvo pokreni trening.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    context_size = int(args.max_length) if args.max_length is not None else int(cfg["context_size"])

    tokenizers_dir = run_dir / "tokenizers"
    weights_dir = run_dir / "weights"
    tok_path = tokenizers_dir / "tokenizer_src.json"
    if not tok_path.exists():
        raise FileNotFoundError("Nema tokenizer_src.json. Prvo pokreni trening.")

    tokenizer = Tokenizer.from_file(str(tok_path))

    ckpt_path = find_latest_checkpoint(weights_dir)
    if ckpt_path is None:
        raise FileNotFoundError("Nema checkpoint-a. Prvo pokreni trening.")

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

    ids = tokenizer.encode(args.text).ids
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    pad_len = context_size - len(ids) - 2
    if pad_len < 0:
        raise ValueError(f"Tekst predugačak za context_size={context_size}.")

    encoder_input = (
        torch.tensor([sos_id] + ids + [eos_id] + [pad_id] * pad_len, dtype=torch.int64)
        .unsqueeze(0)
        .to(device)
    )
    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()

    with torch.no_grad():
        enc_out = model.encode(encoder_input, encoder_mask)
        pooled = mean_pool(enc_out, encoder_mask)
        logits = classifier(pooled)
        # Samo prvi uzorak: probs[0]=P(negative), probs[1]=P(positive)
        logits_1 = logits.flatten()[:2]
        probs = F.softmax(logits_1.unsqueeze(0), dim=1)[0].cpu()
        p_neg, p_pos = probs[0].item(), probs[1].item()
        pred = 1 if p_pos >= p_neg else 0  # iz istih verovatnoća koje ispisujemo

    sentiment = "positive" if pred == 1 else "negative"
    print("Tekst:", args.text[:200] + ("..." if len(args.text) > 200 else ""))
    print("Sentiment:", sentiment, f"  (P(negative)={p_neg:.3f}, P(positive)={p_pos:.3f})")


if __name__ == "__main__":
    main()
