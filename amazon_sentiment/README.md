# Amazon Sentiment (encoder + klasifikator)

**Binarna klasifikacija sentimenta**: ulaz = tekst recenzije, izlaz = **positive** ili **negative**. Koristi se **samo encoder** Transformer-a + **linearni sloj** `nn.Linear(d_model, 2)` — bez dekodera, bez greedy decoding, bez causal mask. Brže i jednostavnije od seq2seq pristupa.

---

## Šta je potrebno za pokretanje

- **Python** 3.9+ (preporučeno 3.10+).
- **Radni direktorijum:** sve komande se pokreću iz **korena repozitorijuma** (`transformer`), npr. `cd c:\...\transformer`.
- **Zavisnosti:** iz korena projekta:
  ```bash
  pip install -r requirements.txt
  ```
  Za sentiment modul posebno treba: `torch`, `datasets`, `tokenizers`, `scikit-learn`, `tqdm`, `tensorboard`, `gradio`, `numpy`. Za konfuzionu matricu (slika): `matplotlib`.
- U root-u mora postojati `model.py` (koristi se `build_transformer` i `.encode()`).

---

## Opis problema

- **Ulaz:** recenzija proizvoda (tekst).
- **Izlaz:** jedna od dve klase — `positive` (1) ili `negative` (0).
- **Model:** encoder obrađuje tekst; izlaz encodera se **mean-pooluje** (prosek samo nad ne-pad tokenima) → vektor dimenzije `d_model` → linearni sloj → 2 logita → argmax daje predikciju.
- **Cilj:** jednostavan i brz trening, manje memorije nego encoder–decoder.

---

## Skup podataka

**[mteb/amazon_polarity](https://huggingface.co/datasets/mteb/amazon_polarity)** — recenzije sa Amazona, kolone `text` i `label` (0 = negative, 1 = positive). U projektu: 80k train, 5k val, 5k test.

---

## Parametri treninga (podrazumevano)

| Parametar      | Vrednost | Opis |
|----------------|----------|------|
| `--max-train`  | 80 000   | Broj trening uzoraka |
| `--batch-size` | 64       | Veličina batch-a |
| `--epochs`     | 3        | Broj epoha |
| `--context-size` | 256   | Maks. dužina u tokenima (manji = brži trening; recenzije su obično ~100–200 tokena) |

Metrike: **accuracy**, **precision**, **recall**, **F1** (macro). Evaluacija na kraju svake epohe.

---

## Kako pokrenuti

Iz korena repozitorijuma (`transformer`):

### Trening

```bash
python amazon_sentiment/train_amazon_sentiment.py --run-dir runs/amazon_sentiment
```

### Resume (nastavak od epohe)

```bash
python amazon_sentiment/train_amazon_sentiment.py --run-dir runs/amazon_sentiment --epochs 6 --resume-from runs/amazon_sentiment/weights/epoch_002.pt
```

### Inferenca (CLI)

```bash
python amazon_sentiment/infer_amazon_sentiment.py --run-dir runs/amazon_sentiment --text "Great product, fast delivery!"
```

Izlaz: `positive` ili `negative` i verovatnoće P(negative), P(positive).

### Web aplikacija (Gradio)

Otvara lokalni interfejs u browseru za unos recenzije i prikaz sentimenta + verovatnoća.

```bash
python amazon_sentiment/app_sentiment.py
```

- Podrazumevano učitava model iz `runs/amazon_sentiment`. Drugi folder: `--run-dir putanja`.
- Za javni link (npr. za test sa drugog uređaja): `--share`.
- Otvori u browseru: **http://127.0.0.1:7860**.

Zavisnost: `gradio` (`pip install gradio` ako nije u requirements).

### Konfuziona matrica

Računa predikcije nad train/val/test splitom, ispisuje konfuzionu matricu i (ako je instaliran matplotlib) snima sliku.

```bash
python amazon_sentiment/confusion_matrix_sentiment.py --run-dir runs/amazon_sentiment
```

- Podrazumevano: `--split val`. Može i `--split train` ili `--split test`.
- Ograničenje uzoraka: `--max-samples 5000`.
- Samo crtanje (bez ponovnog učitavanja modela), ako već postoji sačuvana matrica:
  ```bash
  python amazon_sentiment/confusion_matrix_sentiment.py --plot-only
  ```
- Slika se snima u `run_dir/confusion_matrix_<split>.png` (ili `--out putanja.png`).

Zavisnost za sliku: `matplotlib` (`pip install matplotlib`).

### Provera balansa labela (opciono)

Prikazuje koliko ima negativnih/pozitivnih u trening skupu (dataset `mteb/amazon_polarity`).

```bash
python amazon_sentiment/check_balance.py
```

Zavisnost: `datasets`, `numpy`.

---

## Struktura run foldera

| Staza | Opis |
|-------|------|
| `run_config.json` | Konfiguracija (context_size, d_model, itd.) |
| `tokenizers/tokenizer_src.json` | Samo jedan tokenizer (ulazni tekst) |
| `weights/epoch_*.pt`, `best.pt` | model_state_dict (encoder), classifier_state_dict, optimizer_state_dict |

---

## Ostale opcije treninga

- `--force-rebuild-tokenizer` — ponovo napravi tokenizer (inače se učitava iz run foldera)
- `--max-train`, `--max-val`, `--max-test` — broj uzoraka po split-u
- `--lr`, `--context-size`, `--d-model`, `--blocks`, `--heads`, `--dropout`, `--d-ff` — hiperparametri modela

---

## Zavisnosti po fajlu

| Fajl | Zavisnosti |
|------|------------|
| `train_amazon_sentiment.py` | `torch`, `datasets`, `tokenizers`, `scikit-learn`, `tqdm`, `tensorboard`, `numpy` |
| `infer_amazon_sentiment.py` | `torch`, `tokenizers` |
| `app_sentiment.py` | `torch`, `tokenizers`, `gradio` |
| `confusion_matrix_sentiment.py` | `torch`, `datasets`, `tokenizers`, `scikit-learn`, `tqdm`, `numpy`; za sliku i `matplotlib` |
| `check_balance.py` | `datasets`, `numpy` |

U root-u projekta obavezno: **`model.py`** (koristi se `build_transformer` i `.encode()`). Za sentiment se ne koristi `dataset.py`.
