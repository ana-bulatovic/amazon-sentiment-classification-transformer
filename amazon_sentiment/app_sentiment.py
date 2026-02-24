"""
Moderan web interfejs za sentiment analizu (Gradio).
Pokretanje: python amazon_sentiment/app_sentiment.py
           ili: python amazon_sentiment/app_sentiment.py --run-dir runs/amazon_sentiment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import build_transformer

try:
    import gradio as gr
except ImportError:
    print("Instaliraj Gradio: pip install gradio")
    sys.exit(1)


RUN_DIR = Path("runs/amazon_sentiment")
device = None
model = None
classifier = None
tokenizer = None
context_size = None


def mean_pool(encoder_output: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
    mask = encoder_mask.squeeze(1).unsqueeze(-1).float()
    sum_h = (encoder_output * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return sum_h / count


def load_model(run_dir: Path) -> None:
    global device, model, classifier, tokenizer, context_size
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Nema run_config.json u {run_dir}. Prvo pokreni trening.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    context_size = int(cfg["context_size"])

    tok_path = run_dir / "tokenizers" / "tokenizer_src.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Nema tokenizer u {run_dir}. Prvo pokreni trening.")
    tokenizer = Tokenizer.from_file(str(tok_path))

    weights_dir = run_dir / "weights"
    ckpt_path = weights_dir / "best.pt"
    if not ckpt_path.exists():
        ckpts = sorted(weights_dir.glob("epoch_*.pt"))
        ckpt_path = ckpts[-1] if ckpts else None
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Nema checkpoint-a u {run_dir}. Prvo pokreni trening.")

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


def predict(text: str) -> str:
    if not text or not text.strip():
        return '<div class="result-card result-placeholder">Unesi recenziju.</div>'
    global device, model, classifier, tokenizer, context_size
    ids = tokenizer.encode(text.strip()).ids
    sos_id = tokenizer.token_to_id("[SOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    pad_len = context_size - len(ids) - 2
    if pad_len < 0:
        return '<div class="result-card result-error">Tekst je predugačak. Skrati recenziju.</div>'

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
        logits_1 = logits.flatten()[:2]
        probs = F.softmax(logits_1.unsqueeze(0), dim=1)[0].cpu()
        p_neg, p_pos = probs[0].item(), probs[1].item()
        pred = 1 if p_pos >= p_neg else 0

    label = "Pozitivna"
    emoji = "😊"
    if pred == 0:
        label = "Negativna"
        emoji = "😞"
    return (
        f'<div class="result-card">'
        f'<div class="result-sentiment">{emoji} {label}</div>'
        f'<div class="result-probs">P(negativna) = {p_neg:.1%}  ·  P(pozitivna) = {p_pos:.1%}</div>'
        f'</div>'
    )


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
    ).set(
        body_background_fill="linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#cbd5e1",
        block_radius="12px",
        input_background_fill="#f1f5f9",
        button_primary_background_fill="linear-gradient(90deg, #0d9488 0%, #0f766e 100%)",
    )
    with gr.Blocks(
        title="Sentiment analiza recenzija",
        theme=theme,
        css="""
        .gradio-container { max-width: 640px !important; margin: auto; }
        .result-card {
            text-align: center;
            width: 100%;
            padding: 1rem 1.25rem;
            margin: 0;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            box-sizing: border-box;
        }
        .result-sentiment { font-size: 1.65rem; font-weight: 600; margin: 0; line-height: 1.3; }
        .result-probs {
            font-size: 0.95rem; color: #64748b; margin-top: 0.35rem; margin-bottom: 0;
        }
        .result-placeholder { color: #94a3b8; }
        .result-error { color: #b91c1c; }
        """
    ) as demo:
        gr.Markdown(
            """# Sentiment analiza recenzija
            Unesi tekst recenzije — model predvida da li je **pozitivna** ili **negativna**."""
        )
        with gr.Row():
            inp = gr.Textbox(
                label="Tekst recenzije",
                placeholder="Nalepi recenziju... npr. Great product, fast delivery!",
                lines=4,
                max_lines=12,
            )
        with gr.Row():
            submit_btn = gr.Button("Analiziraj sentiment", variant="primary", size="lg")
        out_result = gr.HTML(label="Rezultat", value="")
        examples = gr.Examples(
            examples=[
                ["Great product, amazing quality. I would buy again!"],
                ["Terrible experience. Broke after one day. Do not recommend."],
                ["Fast delivery, exactly as described. Very satisfied."],
                ["Waste of money. Poor quality and rude customer service."],
                ["Love it! Works perfectly and arrived on time. Five stars."],
                ["Complete junk. Stopped working after a week. Refund requested."],
                ["Good value for money. Does what it says on the box. Happy with purchase."],
                ["Disappointed. The description was misleading and quality is poor."],
                ["Best purchase I've made this year. Highly recommend to everyone."],
                ["Never again. Customer support was useless and the product broke immediately."],
                ["Solid build, looks nice, easy to use. No complaints at all."],
                ["Cheap materials, feels flimsy. Would not buy from this seller again."],
                ["Exactly what I needed. Fast shipping and well packaged. Thank you!"],
                ["Overpriced for what you get. Save your money and look elsewhere."],
                ["Impressed by the quality. My whole family is happy with this product."],
                ["Fake product, not as advertised. Very angry and want my money back."],
                ["Works great so far. Good instructions and quick setup. Recommended."],
                ["Damaged on arrival. Packaging was torn and item is scratched. Returning."],
            ],
            inputs=inp,
            label="Primeri (klik da ispuniš)",
        )
        submit_btn.click(fn=predict, inputs=inp, outputs=out_result)
        inp.submit(fn=predict, inputs=inp, outputs=out_result)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=RUN_DIR, type=Path, help="Putanja do run foldera (weights, tokenizer)")
    parser.add_argument("--share", action="store_true", help="Kreira Gradio share link (javni URL)")
    args = parser.parse_args()
    run_dir = args.run_dir if isinstance(args.run_dir, Path) else Path(args.run_dir)
    run_dir = run_dir.resolve()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    print("Učitavam model iz:", run_dir)
    load_model(run_dir)
    print("Model učitan. Pokrećem interfejs...")
    demo = build_ui()
    demo.launch(share=args.share, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
