"""Download the curated Phase 1 source registry without modifying pilot data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "data" / "corpus" / "document_registry.json"
ALLOWED_HOSTS = {"www.isro.gov.in", "www.issdc.gov.in"}
logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_path_to_file(local_path: str) -> Path:
    return ROOT / Path(local_path.replace("/", os.sep).replace("\\", os.sep))


def validate_response(document: dict, body: bytes, content_type: str) -> None:
    expected_type = document["file_type"]
    if expected_type == "pdf" and not body.startswith(b"%PDF-"):
        raise ValueError(
            f"{document['document_id']} expected a PDF, received "
            f"{content_type!r} and {body[:40]!r}"
        )
    if expected_type == "html":
        sample = body[:8192].decode("utf-8", errors="replace").lower()
        if "<html" not in sample and "<!doctype html" not in sample:
            raise ValueError(
                f"{document['document_id']} expected HTML, received "
                f"{content_type!r} and {body[:40]!r}"
            )


def download_one(document: dict) -> tuple[str, str]:
    url = document["source_url"]
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"Unapproved source host for {document['document_id']}: {host}")

    request = Request(
        url,
        headers={"User-Agent": "ISRO-KG-RAG-Phase1-Corpus/1.0"},
    )
    with urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise HTTPError(url, response.status, "Unexpected HTTP status", response.headers, None)
        content_type = response.headers.get("Content-Type", "")
        body = response.read()

    validate_response(document, body, content_type)
    digest = hashlib.sha256(body).hexdigest()
    output_path = registry_path_to_file(document["local_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")
    try:
        temporary_path.write_bytes(body)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(output_path.relative_to(ROOT)).replace("\\", "/"), digest


def download_registry(registry_path: Path = DEFAULT_REGISTRY) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    documents = registry["documents"]
    errors: list[str] = []
    seen_hashes: dict[str, tuple[str, str]] = {}

    for document in documents:
        local_path = registry_path_to_file(document["local_path"])
        existed_before = local_path.exists()
        try:
            if local_path.exists():
                digest = sha256_file(local_path)
                validate_response(
                    document,
                    local_path.read_bytes()[:8192],
                    "existing local document",
                )
                relative_path = str(local_path.relative_to(ROOT)).replace("\\", "/")
            else:
                relative_path, digest = download_one(document)
                local_path = registry_path_to_file(relative_path)

            duplicate = seen_hashes.get(digest)
            if duplicate and duplicate[0] != document["document_id"]:
                duplicate_path, duplicate_id = duplicate[1], duplicate[0]
                if not existed_before and local_path.is_relative_to(ROOT / "data" / "corpus" / "raw"):
                    local_path.unlink(missing_ok=True)
                document["local_path"] = duplicate_path
                document["status"] = "duplicate_content_reused"
                document["notes"] += f" Identical SHA-256 to {duplicate_id}; shared its local file."
                logger.info("%s reuses identical bytes from %s", document["document_id"], duplicate_id)
            else:
                document["local_path"] = relative_path
                document["status"] = (
                    "existing_pilot_reused"
                    if document["status"] == "existing_pilot_reused"
                    else "downloaded"
                )
                seen_hashes[digest] = (document["document_id"], relative_path)

            document["checksum_sha256"] = digest
            document["retrieval_date"] = registry.get("retrieval_date") or date.today().isoformat()
            logger.info("%s: %s (%s)", document["document_id"], document["status"], digest)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            document["status"] = "download_failed"
            document["notes"] += f" Download failure: {type(exc).__name__}: {exc}"
            errors.append(f"{document['document_id']}: {type(exc).__name__}: {exc}")
            logger.error("Failed to collect %s: %s", document["document_id"], exc)
        finally:
            registry_path.write_text(
                json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    errors = download_registry(args.registry)
    if errors:
        logger.error("Collection finished with %d source error(s):\n%s", len(errors), "\n".join(errors))
        return 1
    logger.info("Collected all %d registered sources.", len(json.loads(args.registry.read_text(encoding="utf-8"))["documents"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
