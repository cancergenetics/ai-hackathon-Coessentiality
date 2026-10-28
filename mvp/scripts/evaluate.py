"""Thin wrapper around coessential.cli.evaluate_main (run with --help for options)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coessential.cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    evaluate_main()
