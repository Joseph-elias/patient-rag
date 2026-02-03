import subprocess
import sys

patients = ["P001", "P005", "P010", "P012"]
questions = [
    "What adverse events are mentioned?",
    "What treatment did the patient receive?",
    "What is the diagnosis?",
    "Is pneumonitis present?"
]

py = sys.executable  # IMPORTANT: uses the current venv python

for pid in patients:
    for q in questions:
        print("\n" + "="*80)
        print("PATIENT:", pid, "| Q:", q)
        cmd = [
            py, "-m", "src.app.ask",
            "--question", q,
            "--patient_id", pid
        ]
        subprocess.run(cmd, check=False)
