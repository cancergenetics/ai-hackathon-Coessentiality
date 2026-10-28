"""Thin wrapper around coessential.all_pairs.main (run with --help for options)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coessential.all_pairs import main  # noqa: E402

if __name__ == "__main__":
    main()
