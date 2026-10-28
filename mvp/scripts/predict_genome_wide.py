"""Thin wrapper around coessential.cli.genome_wide_main (run with --help for options)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coessential.cli import genome_wide_main  # noqa: E402

if __name__ == "__main__":
    genome_wide_main()
