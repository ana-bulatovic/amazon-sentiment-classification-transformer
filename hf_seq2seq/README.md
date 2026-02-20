# HF Seq2Seq Transformer (Encoder–Decoder)

Ovaj projekat uzima **HuggingFace dataset** i rešava **seq2seq** problem. Glavni primer je **apstraktivno sažimanje dijaloga (SAMSum)**.

---

## Problem: sažimanje dijaloga (SAMSum)

- **Dataset**: [SAMSum (knkarthick)](https://huggingface.co/datasets/knkarthick/samsum) – dijalozi iz poruka (chat) sa ljudski napisanim rezimeom.
- **Ulaz (encoder)**: ceo **dijalog**, npr. `"Amanda: I baked cookies. Do you want some?\nRobert: Sure!"`.
- **Izlaz (decoder)**: **kratak rezime**, npr. `"Amanda is offering Robert some cookies and he accepts."`.

Koristi se isti **Transformer** iz `model.py` (encoder + decoder).

---

## Preprocesiranje

1. **Split** – train / validation / test (SAMSum već ima splitove).
2. **Tokenizacija** – WordLevel tokenizer, specijalni tokeni `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`.
3. **Padding** – sve sekvence na fiksnu dužinu `context_size`; predugačke se filtriraju.
4. **Maske** – encoder_mask (sakriva PAD), decoder_mask (PAD + causal).
5. **Teacher forcing** – `decoder_input` i `label` su pomerene sekvence za trening.

---

## Fajlovi

- **`train_hf_seq2seq.py`** – učitava HF dataset (default SAMSum), gradi tokenizer-e, trenira model, validacija i test. Čuva u `runs/...`.
- **`infer_hf_seq2seq.py`** – učitava model, prima dijalog (`--sentence`), ispisuje rezime.

---

## Instalacija

U root folderu projekta:

```bash
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu121
```

---

## Pokretanje (PowerShell, iz root foldera)

### Prvo proveriti da li je GPU dostupan i da li se koristi
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Trening 

```powershell
python .\hf_seq2seq\train_hf_seq2seq.py 
```

### Nastavak od prethodno odradjene epohe
```powershell
python hf_seq2seq\train_hf_seq2seq.py --run-dir "runs/hf_seq2seq_samsum_big --resume-from "runs/hf_seq2seq_samsum_big/weights/epoch_032.pt"  
```
U ovom primeru je epoha 014 - pronalazi se u runs/weights folderu koja je poslednja.

### Brisanje epoha
```powershell
python .\hf_seq2seq\fix_best_and_trim_epochs.py --run-dir "runs/hf_seq2seq_samsum_big" --max-epoch 25
```
### TensorBoard

```powershell
tensorboard --logdir "runs/hf_seq2seq_samsum\tensorboard"
```

### Inferenca (dijalog → rezime)

```powershell
python .\hf_seq2seq\infer_hf_seq2seq.py --run-dir "runs/hf_seq2seq_samsum" --sentence "Amanda: I baked cookies. Do you want some? Robert: Sure!"
```
