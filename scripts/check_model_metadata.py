"""Print the metadata stored inside the saved Keras model file.

Usage (from the project root):
    python scripts/check_model_metadata.py
"""

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "balanced_best_phase3.keras"


def main():
    if not MODEL_PATH.exists():
        print("Model not found:", MODEL_PATH)
        return

    with zipfile.ZipFile(MODEL_PATH, "r") as archive:
        print("Archive files:", archive.namelist())

        config = json.loads(archive.read("config.json"))
        metadata = json.loads(archive.read("metadata.json"))

    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))
    print("\nModel name:", config.get("config", {}).get("name"))
    print("Saved Keras version:", metadata.get("keras_version"))


if __name__ == "__main__":
    main()
