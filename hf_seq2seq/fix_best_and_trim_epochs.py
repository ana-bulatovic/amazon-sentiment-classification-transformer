"""
Jednokratni skript: u run folderu nađe najbolju epohu od 1 do 25 (po val_loss),
sačuva je kao best.pt, i obriše checkpoint-e za epohe 26+.

Pokretanje (iz root foldera projekta):
  python .\hf_seq2seq\fix_best_and_trim_epochs.py --run-dir "runs/hf_seq2seq_samsum_big" --max-epoch 25
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Putanja do run foldera (npr. runs/hf_seq2seq_samsum_big)")
    parser.add_argument("--max-epoch", type=int, default=25, help="Zadrži epohe 0..max_epoch-1 (npr. 25 = epohe 1-25), obriši ostale")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        print(f"Ne postoji folder {weights_dir}")
        return

    # Nađi najbolju epohu u opsegu [0, max_epoch-1]
    best_epoch = None
    best_val_loss = float("inf")
    best_ckpt_path = None

    for e in range(args.max_epoch):
        p = weights_dir / f"epoch_{e:03d}.pt"
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        val_metrics = ckpt.get("val_metrics") or {}
        vl = val_metrics.get("val_loss")
        if vl is not None and vl < best_val_loss:
            best_val_loss = vl
            best_epoch = e
            best_ckpt_path = p

    if best_ckpt_path is None:
        print(f"Nijedan checkpoint za epohe 0..{args.max_epoch - 1} nije pronađen u {weights_dir}")
        return

    # Učitaj pun checkpoint i sačuvaj kao best.pt
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    best_pt = weights_dir / "best.pt"
    ckpt["best_val_loss"] = best_val_loss
    ckpt["_best_epoch"] = best_epoch
    torch.save(ckpt, best_pt)
    print(f"best.pt ažuriran: kopija epohe {best_epoch + 1} (val_loss={best_val_loss:.4f})")

    # Obriši epoch_025.pt, epoch_026.pt, ...
    deleted = []
    for p in sorted(weights_dir.glob("epoch_*.pt")):
        epoch_num = int(p.stem.split("_")[1])
        if epoch_num >= args.max_epoch:
            p.unlink()
            deleted.append(p.name)
    if deleted:
        print(f"Obrisani checkpoint-i (epohe {args.max_epoch}+): {deleted}")
    else:
        print(f"Nema checkpoint-a za epohe >= {args.max_epoch}.")


if __name__ == "__main__":
    main()
