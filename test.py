from subprocess import run
from pathlib import Path

# Folder u kojem je tvoj run sa weights i tokenizers
RUN_DIR = "runs/hf_seq2seq_samsum_big"

# Lista dijaloga za test
DIALOGUES = [
    "Amanda: I baked cookies. Robert: Sure!",
    "Alice: Did you finish the report? Bob: Not yet, I need one more hour.",
    "John: Can you pick up groceries? Mary: Yes, after work.",
    "Emma: Are you coming to the party tonight? Liam: I think so.",
    "Sophia: I lost my keys. Ethan: Did you check your bag?",
    "Olivia: I booked the tickets for the movie. Noah: Great!",
    "Mia: How was your day? Lucas: Pretty good, thanks!",
    "Charlotte: Are you hungry? Aiden: Yes, very.",
    "Amelia: Can we meet tomorrow? Logan: Sure, what time?",
    "Harper: I need help with this project. Mason: I can assist you.",
    "Evelyn: Did you call the plumber? James: Not yet, I will.",
    "Abigail: I finished the homework. Benjamin: Good job!",
    "Ella: Are we still on for dinner? Elijah: Absolutely!",
    "Scarlett: Can you send me the files? Henry: Already sent.",
    "Victoria: Did you see the latest episode? Jackson: Not yet, no spoilers!"
]

PYTHON = r"C:\Users\rcadmin\Desktop\transformer\dialogue-summarization-transformer\venv\Scripts\python.exe"

def summarize_dialogue(dialogue: str):
    cmd = [
        PYTHON, "hf_seq2seq/infer_hf_seq2seq.py",
        "--run-dir", RUN_DIR,
        "--sentence", dialogue
    ]
    result = run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

if __name__ == "__main__":
    for i, dlg in enumerate(DIALOGUES, 1):
        print(f"\n--- Dijalog {i} ---")
        summarize_dialogue(dlg)