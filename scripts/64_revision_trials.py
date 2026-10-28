"""Revision F12: count trial records in results/trials.jsonl (start / end records, end records by status, starts
without an end) and the IFT look log. Writes results/revision/trials_counts.json.

The counts are given twice: for the log as it was before the revision analyses (records with time earlier than
``--before``, default = the first revision record) and for the full log.

Usage: ./.env/bin/python scripts/64_revision_trials.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS  # noqa: E402

REV_GRIDS = ("revision_",)


def is_revision(r):
    tag = str(r.get("tag", ""))
    grid = str((r.get("extra") or {}).get("grid", ""))
    return tag.startswith(("revision_", "onion_")) or grid.startswith(REV_GRIDS)


def count(recs):
    starts = [r for r in recs if r.get("event") == "start"]
    ends = [r for r in recs if r.get("event") == "end"]
    end_ids = {r["trial_id"] for r in ends}
    start_ids = {r["trial_id"] for r in starts}
    return dict(n_records=len(recs), n_start=len(starts), n_end=len(ends),
                end_by_status=dict(Counter(r.get("status", "?") for r in ends).most_common()),
                n_unique_trial_ids=len(start_ids | end_ids),
                n_start_without_end=len(start_ids - end_ids), n_end_without_start=len(end_ids - start_ids),
                start_without_end_tags=sorted(str(r.get("tag")) for r in starts if r["trial_id"] not in end_ids),
                by_universe=dict(Counter(str(r.get("universe")) for r in starts).most_common()),
                first=recs[0]["time"] if recs else None, last=recs[-1]["time"] if recs else None)


def main():
    allrecs = [json.loads(l) for l in (RESULTS / "trials.jsonl").read_text().splitlines() if l.strip()]
    notes = [r for r in allrecs if r.get("event") not in ("start", "end")]
    recs = [r for r in allrecs if r.get("event") in ("start", "end")]
    rev_ids = {r["trial_id"] for r in recs if is_revision(r)}
    pre = [r for r in recs if r["trial_id"] not in rev_ids]
    looks = [json.loads(l) for l in (RESULTS / "ift_looks.jsonl").read_text().splitlines() if l.strip()]
    out = dict(source="results/trials.jsonl", before_revision=count(pre), revision_only=count(
        [r for r in recs if r["trial_id"] in rev_ids]), full_log=count(recs),
        n_lines=len(allrecs), other_records=[dict(event=r.get("event"), text=str(r.get("text", ""))[:300]) for r in notes],
        ift_looks=dict(n_records=len(looks), looks=[dict(look=l["look"], status=l.get("status"), what=l["what"][:120])
                                                    for l in looks]),
        note="one trial = one evaluation call (grid point x seed), logged by src.evaluate.run_dev_cv / log_trial; "
             "'ok_not_written' = evaluated and logged without writing a result JSON (e.g. nested selection steps, "
             "screen probes); revision records = tags 'revision_*' / 'onion_*' or grid 'revision_*'")
    (RESULTS / "revision" / "trials_counts.json").write_text(json.dumps(out, indent=1))
    for k in ("before_revision", "revision_only", "full_log"):
        c = out[k]
        print(k, c["n_start"], c["n_end"], c["end_by_status"], "no-end:", c["n_start_without_end"])


if __name__ == "__main__":
    main()
