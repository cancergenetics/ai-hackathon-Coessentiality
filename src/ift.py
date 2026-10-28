"""Internal final test (IFT) loader. ONLY scripts/final_eval.py may import this module
(enforced by tests/test_process.py::test_only_final_eval_imports_ift, an AST scan of src/ and scripts/).

Returns a universe with dev AND IFT pairs (side column), so that supervised candidates can be fit on dev pairs and
scored on IFT pairs; cross pairs are excluded. Also returns the unscorable (unmapped-protein) IFT pairs for the
§3.10 report (minimum score rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CACHE
from src.universe import PairUniverse, _build


def load_ift_universe() -> tuple[PairUniverse, pd.DataFrame]:
    from src.clusters import load_assignment
    u = _build(("dev", "ift"), "dev+ift")
    assign = load_assignment()
    m = assign[assign.mapped]
    u.meta["f_ift"] = float((m.side == "ift").mean())
    u.meta["f_dev"] = float((m.side == "dev").mean())
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    side = assign.set_index("protein").side
    un = allp[~allp.scorable].copy()
    un["side"] = np.where(side.reindex(un.a).values == side.reindex(un.b).values, side.reindex(un.a).values, "cross")
    return u, un[un.side == "ift"].reset_index(drop=True)
