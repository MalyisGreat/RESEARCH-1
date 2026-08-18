#!/usr/bin/env python3
"""Run gman with the locally stored, workspace-scoped service token."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    token_file = Path(
        os.environ.get(
            "GMAN_SERVICE_TOKEN_FILE",
            Path.home() / ".config" / "gman" / "research1-service-token.json",
        )
    )
    payload = json.loads(token_file.read_text(encoding="utf-8"))
    token = payload.get("token")
    if not token:
        raise RuntimeError(f"service token missing from {token_file}")
    executable = Path.home() / ".local" / "bin" / "gman"
    environment = os.environ.copy()
    environment["GMAN_TOKEN"] = token
    os.execvpe(str(executable), [str(executable), *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()
