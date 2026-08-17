from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "sapientinc/HRM-Text-data-io-cleaned-20260515"
REVISION = "main"
ROOT = Path(__file__).resolve().parent / "official_source_subset"
FILES = (
    "data/no_robots.jsonl",
    "data/gsm8k_train.jsonl",
    "data/Platypus/openbookqa.jsonl",
    "data_clustered/flan/flan_fsopt_data__ai2_arc_ARC-Challenge_1.0.0.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = []
    for remote_path in FILES:
        destination = ROOT / remote_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded_path = "/".join(urllib.parse.quote(part) for part in remote_path.split("/"))
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{encoded_path}?download=true"
        if not destination.exists():
            print(f"download {remote_path}", flush=True)
            urllib.request.urlretrieve(url, destination)
        records.append(
            {
                "path": remote_path,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "url": url,
            }
        )
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
