"""
Preview the deck and rebuild it when an included section changes

This works around a Quarto preview issue: https://github.com/quarto-dev/quarto-cli/issues/2795

"""

import os
import sys
from pathlib import Path
from subprocess import Popen
from time import sleep


def main() -> int:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    def section_times() -> dict[Path, int]:
        try:
            # cost_model.py writes _variables.yml as Quarto's pre-render step
            paths = [*Path("sections").glob("*.qmd"), Path("cost_model.py")]
            return {path: path.stat().st_mtime_ns for path in paths}
        except FileNotFoundError:
            return {}

    previous = section_times()
    with Popen(["quarto", "preview", "slides.qmd", *sys.argv[1:]]) as preview:
        try:
            while preview.poll() is None:
                sleep(0.5)
                current = section_times()
                if current != previous:
                    Path("slides.qmd").touch()
                    previous = current
        except KeyboardInterrupt:
            preview.terminate()
            preview.wait()
            return 0
        return preview.wait()


if __name__ == "__main__":
    raise SystemExit(main())
