"""Write the OpenAPI spec and a real example response into docs/frontend/.

    python -m src.api.export_docs
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app

OUT = Path(__file__).resolve().parents[2] / "docs" / "frontend"
EXAMPLE_QUESTION = "Which organization developed a payload carried by Aditya-L1?"


def main() -> None:
    os.environ["KG_RAG_MOCK_DELAY_MS"] = "0"
    OUT.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app("mock")) as client:
        while client.get("/api/health").json()["status"] != "ready":
            time.sleep(0.05)
        (OUT / "openapi.json").write_text(json.dumps(client.app.openapi(), indent=2) + "\n", encoding="utf8")
        example = client.post("/api/ask", json={"question": EXAMPLE_QUESTION}).json()
        example["kg_paths"] = example["kg_paths"][:3]  # keep the sample readable
        (OUT / "example_ask_response.json").write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
