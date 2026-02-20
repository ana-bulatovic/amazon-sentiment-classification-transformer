"""
Test primeri za Code → Docstring: pokretanjem skripte proveriš
generisane opise za zadate delove koda (koristi infer_code_comment.py).
"""

from subprocess import run
from pathlib import Path

# Folder u kojem je run sa weights i tokenizers (nakon train_code_comment.py)
RUN_DIR = "runs/code_comment"

# Primeri Python koda (funkcije) za koje želimo opis (docstring)
CODE_EXAMPLES = [
    "def add(a, b): return a + b",
    "def get_user_by_id(user_id): return db.query(User).filter_by(id=user_id).first()",
    "def is_even(n): return n % 2 == 0",
    "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "def reverse_string(s): return s[::-1]",
    "def max_of_three(a, b, c): return max(a, max(b, c))",
    "def count_vowels(text): return sum(1 for c in text.lower() if c in 'aeiou')",
    "def clamp(value, low, high): return max(low, min(high, value))",
    "def safe_divide(a, b): return a / b if b != 0 else None",
    "def flatten(lst): return [item for sublist in lst for item in sublist]",
]

# Putanja do Python interpretora (venv); prilagodi ako treba
ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = "python"  # fallback na sistemski python


def get_docstring(code: str) -> None:
    """Pokreće infer_code_comment.py za dati kod i ispisuje izlaz."""
    cmd = [
        str(PYTHON),
        str(ROOT / "code_comment" / "infer_code_comment.py"),
        "--run-dir", RUN_DIR,
        "--code", code,
    ]
    result = run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
    if result.returncode != 0:
        print("(exit code:", result.returncode, ")")


if __name__ == "__main__":
    print("Run dir:", RUN_DIR)
    print("Primeri: Code → Docstring\n" + "=" * 50)
    for i, code in enumerate(CODE_EXAMPLES, 1):
        print(f"\n--- Primer {i} ---")
        print("Kod:", code[:80] + ("..." if len(code) > 80 else ""))
        print("Opis:")
        get_docstring(code)
