"""Plain JSON/NPZ/HTML evidence and checksums, with no UI or pickle dependency."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import uuid
from pathlib import Path, PurePosixPath


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def value_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value, name="SHA-256") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"Invalid {name}")
    return value


def local_path(root, relative) -> Path:
    """Resolve a declared POSIX relative path, including protection on Windows."""
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
        or PurePosixPath(relative).is_absolute()
        or any(p in (".", "..") for p in relative.split("/"))
    ):
        raise ValueError(f"Unsafe relative artifact path: {relative!r}")
    root = Path(root).resolve()
    result = root.joinpath(*PurePosixPath(relative).parts)
    if not result.resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes its declared root")
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return parse_json(stream.read())


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [parse_json(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, value):
    """Atomic pointer replacement; Windows cannot promise directory-fsync durability."""
    path = Path(path)
    stage = path.with_name(f".{uuid.uuid4().hex[:16]}.partial")
    try:
        write_json(stage, value)
        os.replace(stage, path)
        sync_directory(path.parent)
    finally:
        stage.unlink(missing_ok=True)


def sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_bytes(row).decode("utf-8") + "\n")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def software():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "tokenizers",
        "numpy",
        "safetensors",
        "sqlglot",
        "huggingface-hub",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def source_identity():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def redact_error(value):
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "[REDACTED]", str(value))
    return re.sub(r"(?i)([?&](?:token|access_token)=)[^&\s]+", r"\1[REDACTED]", text)


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def seal(directory, *, status="complete", metadata=None):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Output symlinks are not supported")
        if path.is_file() and path != directory / "manifest.json":
            files[path.relative_to(directory).as_posix()] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    write_json(
        directory / "manifest.json",
        {
            "format": "remove-reset-artifacts-v1",
            "status": status,
            "created_utc": now(),
            "files": files,
            **(metadata or {}),
        },
    )


def verify_bundle(directory, *, complete=True):
    directory = Path(directory).resolve(strict=True)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format") != "remove-reset-artifacts-v1":
        raise ValueError("Unsupported artifact manifest")
    if complete and manifest["status"] != "complete":
        raise ValueError("Artifact is not complete")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Artifact file set changed")
    for name, binding in manifest["files"].items():
        path = local_path(directory, name)
        if (
            path.is_symlink()
            or path.stat().st_size != binding["bytes"]
            or sha256(path) != digest(binding["sha256"])
        ):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def save_result(directory, result, *, status="complete", error=None):
    """Generic shape, numeric arrays separate, standalone inspectable HTML.

    Figures, when supplied, are plain serializable specifications, not a UI
    callback. No scores are invented for failed or incomplete computations.
    """
    from dataclasses import asdict, is_dataclass

    doc = asdict(result) if is_dataclass(result) else dict(result)
    arrays = doc.pop("arrays", {})
    if arrays:
        import numpy as np

        for key, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"Arrays must be finite numeric values: {key}")
        np.savez_compressed(Path(directory) / "arrays.npz", **arrays)
        doc["arrays"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }
    else:
        doc["arrays"] = {}
    doc = {"format": "remove-reset-result-v1", "status": status, "created_utc": now(), **doc}
    write_json(Path(directory) / "result.json", doc)
    if error is not None:
        write_json(
            Path(directory) / "error.json",
            {
                "type": type(error).__name__,
                "message": redact_error(error),
                "status": "failed",
                "current_partial_update_resumable": False,
            },
        )
    rendered = html.escape(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False))
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Remove or reset: local artifacts</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Remove or reset: local artifacts</h1>"
        "<p>Read the scope, controls, exclusions and raw values before interpretation.</p>"
        f"<pre>{rendered}</pre></html>"
    )
    with (Path(directory) / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    seal(directory, status=status)
    return doc
