"""
Sentiment klasifikacija: encoder + linearni sloj (nn.Linear(d_model, 2)).

Recenzija (tekst) → logits za [negative, positive] → argmax.
Bez dekodera, bez greedy decoding, bez causal mask. Brže i jednostavnije.

Dataset: mteb/amazon_polarity. 80k train, batch 64, 3 epohe.
Metrike: accuracy, precision, recall, F1.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import build_transformer


DATASET_NAME = "mteb/amazon_polarity"
SOURCE_COLUMN = "text"
LABEL_COLUMN = "label_text"  # "positive" ili "negative"


def get_row_value(row: Dict[str, Any], key: str) -> str:
    val = row.get(key)
    if val is None:
        return ""
    return val if isinstance(val, str) else str(val)


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    source_column: str
    context_size: int
    model_dimension: int
    number_of_blocks: int
    heads: int
    dropout: float
    feed_forward_dimension: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    seed: int
    max_train_samples: int
    max_val_samples: int
    max_test_samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_amazon_splits(
    max_train: int,
    max_val: int,
    max_test: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    ds = load_dataset(DATASET_NAME)
    train = ds["train"].select(range(min(max_train, len(ds["train"]))))
    test_full = ds["test"]
    n_test = len(test_full)
    val = test_full.select(range(min(max_val, n_test)))
    test = test_full.select(range(max_val, min(max_val + max_test, n_test)))
    return train, val, test


def iter_sentences(dataset: Dataset, column_key: str) -> Iterable[str]:
    for x in dataset:
        yield get_row_value(x, column_key)


def build_or_load_tokenizer(
    path: Path,
    dataset: Dataset,
    column_key: str,
    force_rebuild: bool,
    min_frequency: int = 2,
    vocab_size: int = 50_000,
) -> Tokenizer:
    if path.exists() and not force_rebuild:
        return Tokenizer.from_file(str(path))
    ensure_dir(path.parent)
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=min_frequency,
        vocab_size=vocab_size,
    )
    tok.train_from_iterator(iter_sentences(dataset, column_key), trainer=trainer)
    tok.save(str(path))
    return tok


def filter_too_long(
    dataset: Dataset,
    tokenizer: Tokenizer,
    source_column: str,
    context_size: int,
) -> Dataset:
    def ok(ex):
        ids = tokenizer.encode(get_row_value(ex, source_column)).ids
        return len(ids) <= context_size - 2  # SOS, EOS

    return dataset.filter(ok)


def mean_pool(encoder_output: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
    """
    encoder_output: (B, L, D)
    encoder_mask: (B, 1, 1, L)
    """
    mask = encoder_mask.squeeze(1).squeeze(1).unsqueeze(-1).float()  # (B, L, 1)
    sum_h = (encoder_output * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return sum_h / count


class SentimentDataset(TorchDataset):
    """Samо encoder ulaz + label 0/1. Bez dekodera."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Tokenizer,
        source_column: str,
        label_column: str,
        context_size: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_col = source_column
        self.label_col = label_column
        self.context_size = context_size
        self.sos = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[int(idx)]
        text = get_row_value(row, self.src_col)
        label_str = get_row_value(row, self.label_col).strip().lower()
        label = 1 if label_str == "positive" else 0
        ids = self.tokenizer.encode(text).ids
        pad_len = self.context_size - len(ids) - 2
        if pad_len < 0:
            raise ValueError("Tekst predugačak za context_size.")
        encoder_input = torch.cat([
            self.sos,
            torch.tensor(ids, dtype=torch.int64),
            self.eos,
            torch.full((pad_len,), self.pad.item(), dtype=torch.int64),
        ], dim=0)
        encoder_mask = (encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int()
        return {
            "encoder_input": encoder_input,
            "encoder_mask": encoder_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    writer: Optional[SummaryWriter],
    global_step: int,
    stage: str,
) -> Dict[str, float]:
    model.eval()
    classifier.eval()
    total_loss, total_batches = 0.0, 0
    y_pred: List[int] = []
    y_true: List[int] = []
    for batch in loader:
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        labels = batch["label"].to(device).long().view(-1)
        enc_out = model.encode(encoder_input, encoder_mask)
        pooled = mean_pool(enc_out, encoder_mask)
        logits = classifier(pooled)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item())
        total_batches += 1
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, total_batches)
    metrics: Dict[str, float] = {f"{stage}_loss": avg_loss}
    if y_true:
        metrics[f"{stage}_accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics[f"{stage}_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics[f"{stage}_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics[f"{stage}_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if writer is not None:
        writer.add_scalar(f"{stage}/loss", metrics[f"{stage}_loss"], global_step)
        for k in ("accuracy", "precision", "recall", "f1"):
            if f"{stage}_{k}" in metrics:
                writer.add_scalar(f"{stage}/{k}", metrics[f"{stage}_{k}"], global_step)
        writer.flush()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment: encoder + classifier (Amazon Polarity)")
    parser.add_argument("--run-dir", default="runs/amazon_sentiment")
    parser.add_argument("--force-rebuild-tokenizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-size", type=int, default=256, help="Manji = brži trening; recenzije ~100–200 tokena")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=80_000)
    parser.add_argument("--max-val", type=int, default=5_000)
    parser.add_argument("--max-test", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    cfg = RunConfig(
        dataset_name=DATASET_NAME,
        source_column=SOURCE_COLUMN,
        context_size=args.context_size,
        model_dimension=args.d_model,
        number_of_blocks=args.blocks,
        heads=args.heads,
        dropout=args.dropout,
        feed_forward_dimension=args.d_ff,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        max_train_samples=args.max_train,
        max_val_samples=args.max_val,
        max_test_samples=args.max_test,
    )
    run_dir = Path(args.run_dir)
    tokenizers_dir = ensure_dir(run_dir / "tokenizers")
    weights_dir = ensure_dir(run_dir / "weights")
    tb_dir = ensure_dir(run_dir / "tensorboard")
    save_json(run_dir / "run_config.json", asdict(cfg))
    set_seed(cfg.seed)

    print("Učitavam", DATASET_NAME, "...")
    train_raw, val_raw, test_raw = load_amazon_splits(
        cfg.max_train_samples, cfg.max_val_samples, cfg.max_test_samples,
    )
    print("Train:", len(train_raw), "Val:", len(val_raw), "Test:", len(test_raw))

    tok_path = tokenizers_dir / "tokenizer_src.json"
    tokenizer = build_or_load_tokenizer(tok_path, train_raw, cfg.source_column, args.force_rebuild_tokenizer)

    train = filter_too_long(train_raw, tokenizer, cfg.source_column, cfg.context_size)
    val = filter_too_long(val_raw, tokenizer, cfg.source_column, cfg.context_size)
    test = filter_too_long(test_raw, tokenizer, cfg.source_column, cfg.context_size)
    print("Posle filter — Train:", len(train), "Val:", len(val), "Test:", len(test))

    train_ds = SentimentDataset(train, tokenizer, cfg.source_column, LABEL_COLUMN, cfg.context_size)
    val_ds = SentimentDataset(val, tokenizer, cfg.source_column, LABEL_COLUMN, cfg.context_size)
    test_ds = SentimentDataset(test, tokenizer, cfg.source_column, LABEL_COLUMN, cfg.context_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.get_vocab_size()
    model = build_transformer(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        source_context_size=cfg.context_size,
        target_context_size=cfg.context_size,
        model_dimension=cfg.model_dimension,
        number_of_blocks=cfg.number_of_blocks,
        heads=cfg.heads,
        dropout=cfg.dropout,
        feed_forward_dimension=cfg.feed_forward_dimension,
    ).to(device)
    classifier = nn.Linear(cfg.model_dimension, 2).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=cfg.learning_rate,
        eps=1e-9,
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    writer = SummaryWriter(str(tb_dir))

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Nastavljam od epohe {start_epoch}")

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        classifier.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            labels = batch["label"].to(device).long().view(-1)
            enc_out = model.encode(encoder_input, encoder_mask)
            pooled = mean_pool(enc_out, encoder_mask)
            logits = classifier(pooled)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()),
                    max_norm=args.grad_clip,
                )
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = cfg.learning_rate * lr_scale
            optimizer.step()
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.flush()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        val_metrics = evaluate(model, classifier, val_loader, loss_fn, device, writer, global_step, "val")
        save_json(run_dir / "last_val_metrics.json", val_metrics)
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, weights_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, weights_dir / "best.pt")
        m = val_metrics
        print(f"Epoch {epoch+1} — loss={m['val_loss']:.4f}, acc={m.get('val_accuracy',0):.4f}, P={m.get('val_precision',0):.4f}, R={m.get('val_recall',0):.4f}, F1={m.get('val_f1',0):.4f}")

    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        print("Test na best.pt")
    test_metrics = evaluate(model, classifier, test_loader, loss_fn, device, None, global_step, "test")
    save_json(run_dir / "test_metrics.json", test_metrics)
    print("Gotovo.", run_dir, "Test:", test_metrics)
    print('Inferenca: python amazon_sentiment/infer_amazon_sentiment.py --run-dir "' + str(run_dir) + '" --text "Great product!"')


if __name__ == "__main__":
    main()
