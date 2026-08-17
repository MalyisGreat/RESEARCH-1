from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "sapientinc/HRM-Text-data-io-cleaned-20260515"
BASE = "https://datasets-server.huggingface.co/rows"
ROOT = Path(__file__).resolve().parent
TARGETS = {"train": 2000, "validation": 200, "test": 200}
PAGE_SIZE = 100


def request_rows(split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}
    )
    url = f"{BASE}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def evenly_spaced_offsets(total: int, pages: int) -> list[int]:
    if pages == 1:
        return [max(0, (total - PAGE_SIZE) // 2)]
    maximum = max(0, total - PAGE_SIZE)
    return [round(index * maximum / (pages - 1)) for index in range(pages)]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": DATASET,
        "config": "default",
        "api": BASE,
        "sampling": "deterministic evenly spaced 100-row pages across each split",
        "splits": {},
    }
    combined_rows = []
    for split, target in TARGETS.items():
        first = request_rows(split, 0, 1)
        total = int(first["num_rows_total"])
        pages = max(1, target // PAGE_SIZE)
        offsets = evenly_spaced_offsets(total, pages)
        selected = []
        for page_index, offset in enumerate(offsets, start=1):
            payload = request_rows(split, offset, PAGE_SIZE)
            for item in payload["rows"]:
                row = item["row"]
                selected.append(
                    {
                        "split": split,
                        "row_idx": int(item["row_idx"]),
                        "instruction": row["instruction"],
                        "response": row["response"],
                        "condition": row["condition"],
                    }
                )
            print(f"{split}: page {page_index}/{pages} offset={offset} rows={len(selected)}", flush=True)
        output = ROOT / f"official_cleaned_{split}_{len(selected)}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        output.write_text(content, encoding="utf-8")
        manifest["splits"][split] = {
            "total_rows": total,
            "sample_rows": len(selected),
            "offsets": offsets,
            "file": output.name,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "instruction_chars": sum(len(row["instruction"]) for row in selected),
            "response_chars": sum(len(row["response"]) for row in selected),
            "conditions": sorted({row["condition"] for row in selected}),
        }
        combined_rows.extend(selected)
    combined = ROOT / "official_cleaned_sample_2400.jsonl"
    combined_content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows)
    combined.write_text(combined_content, encoding="utf-8")
    manifest["combined"] = {
        "file": combined.name,
        "rows": len(combined_rows),
        "bytes": len(combined_content.encode("utf-8")),
        "sha256": hashlib.sha256(combined_content.encode("utf-8")).hexdigest(),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
