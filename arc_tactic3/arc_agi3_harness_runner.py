from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

HARNESS_SRC = Path(__file__).resolve().parents[1] / "arc-agi-3-benchmarking" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from arcagi3.runner import AgentRunner  # noqa: E402

from .arc_agi3_harness_agent import definition as tactic_definition


def main_cli(cli_args: list[str] | None = None) -> None:
    load_dotenv()
    runner = AgentRunner()
    runner.register(tactic_definition)
    runner.run(cli_args)


if __name__ == "__main__":
    main_cli()
