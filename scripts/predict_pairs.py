"""Thin wrapper around coessential.cli.predict_pairs_main (run with --help for options)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coessential.cli import predict_pairs_main  # noqa: E402

if __name__ == "__main__":
    predict_pairs_main()
