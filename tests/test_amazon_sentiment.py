"""
Test primeri za sentiment: 10 recenzija; za svaku se ispisuje da li je pozitivna ili negativna.
Pokreće infer_amazon_sentiment.py za svaki primer.
"""

from subprocess import run
from pathlib import Path

RUN_DIR = "runs/amazon_sentiment"

# 10 primer recenzija (mix pozitivnih i negativnih)
REVIEWS = [
    "Great product, amazing quality. I would buy again!",
    "Real great, broke instantly"
    # "Terrible experience. Broke after one day. Do not recommend.",
    # "Fast delivery, exactly as described. Very satisfied.",
    # "Waste of money. Poor quality and customer service was rude.",
    # "Best purchase I have made this year. Exceeded my expectations.",
    # "Not worth the price. Stopped working after a week.",
    # "Love it! Simple to use and does everything I needed.",
    # "Disappointed. The item was damaged when it arrived.",
    # "Excellent value for money. Would recommend to friends.",
    # "Cheap material, fell apart. Save your money and look elsewhere.",
]

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = "python"


def get_sentiment(text: str) -> None:
    """Pokreće infer_amazon_sentiment.py za dati tekst i ispisuje izlaz."""
    cmd = [
        str(PYTHON),
        str(ROOT / "amazon_sentiment" / "infer_amazon_sentiment.py"),
        "--run-dir", RUN_DIR,
        "--text", text,
    ]
    result = run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
    if result.returncode != 0:
        print("(exit code:", result.returncode, ")")


if __name__ == "__main__":
    print("Run dir:", RUN_DIR)
    print("Sentiment za 10 recenzija\n" + "=" * 50)
    for i, review in enumerate(REVIEWS, 1):
        print(f"\n--- Recenzija {i} ---")
        get_sentiment(review)
