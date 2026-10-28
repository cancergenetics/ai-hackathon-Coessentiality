"""Project-wide constants. Every stochastic step must draw seeds from SEEDS."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
CACHE = ROOT / "data_cache"      # derived from restricted/large data; gitignored
RESULTS = ROOT / "results"       # committed

SEEDS = [0, 1, 2, 3, 4]          # fixed seed list; do not change after first use
PRIMARY_SEED = SEEDS[0]

POS_FILE = DATA / "HuMAP" / "Train_positive_pairs.txt"   # prompt names it Train_positive_pairs_pairs.txt; on disk it is this
NEG_FILE = DATA / "HuMAP" / "Train_negative_pairs.txt"
HGNC_FILE = DATA / "HGNC" / "hgnc_complete_set.txt"
DEPMAP = DATA / "DepMAP"

# Scorable universe (all scorable HuMAP train pairs, commit 9bc81a2; asserted against data in tests/evaluate).
N_POS_SCORABLE = 3191
N_NEG_SCORABLE = 593406
NEG_POS_RATIO = N_NEG_SCORABLE / N_POS_SCORABLE   # target ratio for uniform prevalence reweighting (§3.5)
N_FOLDS = 5
