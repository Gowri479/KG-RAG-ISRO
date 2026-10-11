"""Extract the generator's tokenizer from its Ollama GGUF blob.

The committed ``config/tokenizer/mistral-7b-instruct-q4_K_M`` was produced by
this script; rerun it only if the Ollama model changes. Requires ``gguf``.

    python -m src.generator.extract_tokenizer
"""

from __future__ import annotations

import json
from pathlib import Path

from src.generator.ollama_api import MODEL_NAME
from src.generator.token_budget import TOKENIZER_PATH

OLLAMA_MODELS = Path.home() / ".ollama" / "models"


def model_blob(model: str = MODEL_NAME) -> Path:
    name, tag = model.split(":", 1)
    manifest = OLLAMA_MODELS / "manifests" / "registry.ollama.ai" / "library" / name / tag
    layers = json.loads(manifest.read_text(encoding="utf8"))["layers"]
    digest = next(layer["digest"] for layer in layers if layer["mediaType"].endswith(".model"))
    return OLLAMA_MODELS / "blobs" / digest.replace(":", "-")


def main() -> None:
    from transformers import AutoTokenizer

    blob = model_blob()
    tokenizer = AutoTokenizer.from_pretrained(str(blob.parent), gguf_file=blob.name)
    tokenizer.save_pretrained(str(TOKENIZER_PATH.parent))
    print(f"saved {TOKENIZER_PATH.parent} from {blob.name}")


if __name__ == "__main__":
    main()
