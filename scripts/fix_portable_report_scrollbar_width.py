from __future__ import annotations

import sys
from pathlib import Path


STYLE = """<style id="portable-report-scrollbar-width-fix">
.analytics-top-bar {
  width: 100% !important;
  margin-right: 0 !important;
  margin-left: 0 !important;
}
</style>
"""


def main() -> None:
    report = Path(sys.argv[1])
    html = report.read_text(encoding="utf-8")
    if "portable-report-scrollbar-width-fix" not in html:
        html = html.replace("</head>", f"{STYLE}</head>", 1)
        report.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
