"""Identifier mapping, unmapped handling and the command-line scripts on toy files."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from coessential import cli, data
from toy import modules

HGNC_ROWS = [  # symbol, status, entrez_id, uniprot_ids
    ("G0", "Approved", "100", "P00100"),
    ("G1", "Approved", "101", "P00101"),
    ("G2", "Approved", "102", "P00102|P00999"),     # two accessions on one gene
    ("H1", "Approved", "201", "P00201"),
    ("H2", "Approved", "202", "P00201"),            # P00201 encoded by two genes in DepMap -> ambiguous
    ("ABS", "Approved", "900", "P00900"),           # not a DepMap gene
    ("OLD", "Withdrawn", "103", "P00103"),          # withdrawn rows ignored
]


@pytest.fixture()
def toy_files(tmp_path):
    X, lab = modules(n_lines=60, n_modules=4, size=12, n_noise=200, seed=2)
    G = X.shape[1]
    ent = [str(100 + i) if i < 3 else str(201 + i - 3) if i < 5 else str(1000 + i) for i in range(G)]
    cols = [f"S{e} ({e})" for e in ent]
    df = pd.DataFrame(X, index=[f"ACH-{i:06d}" for i in range(X.shape[0])], columns=cols)
    df.index.name = "ModelID"
    dep = tmp_path / "CRISPRGeneEffect.csv"
    df.to_csv(dep)
    rows = HGNC_ROWS + [(f"S{e}", "Approved", e, f"Q{e}") for e in ent[5:]]
    hg = tmp_path / "hgnc.txt"
    pd.DataFrame(rows, columns=["symbol", "status", "entrez_id", "uniprot_ids"]).assign(hgnc_id="x").to_csv(
        hg, sep="\t", index=False)
    return dict(dep=dep, hgnc=hg, X=X, lab=lab, ent=ent, tmp=tmp_path)


def test_uniprot_mapping(toy_files):
    X, genes = data.load_depmap(toy_files["dep"])
    hg = data.load_hgnc(toy_files["hgnc"])
    u = data.build_uniprot_map(["P00100", "P00102", "P00999", "P00201", "P00900", "P00103", "NOPE"], genes, hg)
    assert u.status.to_dict() == {"NOPE": "not_in_hgnc", "P00100": "mapped", "P00102": "mapped",
                                  "P00103": "not_in_hgnc", "P00201": "ambiguous", "P00900": "not_in_depmap",
                                  "P00999": "mapped"}
    assert u.gene_idx["P00102"] == u.gene_idx["P00999"] == 2 and u.gene_idx["P00201"] == -1
    assert X.dtype == np.float32 and X.shape == toy_files["X"].shape


def _write_pairs(path, pairs):
    path.write_text("".join(f"{a}\t{b}\n" for a, b in pairs) + "\n")


def test_predict_pairs_cli_unmapped(toy_files):
    t = toy_files
    e = t["ent"]
    pairs = [(f"Q{e[10]}", f"Q{e[11]}"), (f"Q{e[11]}", f"Q{e[10]}"), (f"Q{e[10]}", f"Q{e[60]}"),
             ("P00100", "P00101"),
             ("P00102", "P00999"),            # same gene
             ("P00201", "P00100"),            # ambiguous
             ("P00900", "P00100"),            # not in DepMap
             ("NOPE", "P00100")]              # not in HGNC
    pf = t["tmp"] / "pairs.txt"
    _write_pairs(pf, pairs)
    common = ["--pairs", str(pf), "--depmap", str(t["dep"]), "--hgnc", str(t["hgnc"]),
              "--k", "0", "--q", "5", "--dim", "8", "--null-pairs", "2000"]
    out = t["tmp"] / "s.tsv"
    cli.predict_pairs_main(common + ["--out", str(out), "--components", "--add-gls"])
    d = pd.read_csv(out, sep="\t")
    assert list(d.columns) == ["uniprot_a", "uniprot_b", "score", "mapped", "z_B", "z_E", "z_F", "gls_t"]
    assert d.mapped.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert d.uniprot_a.tolist() == [p[0] for p in pairs]                      # input order and orientation
    assert d.score[0] == d.score[1]
    floor = d.score[:4].min() - 1
    assert np.allclose(d.score[4:], floor) and d[["z_B", "gls_t"]][4:].isna().all().all()
    assert np.allclose(d.score[:4], d[["z_B", "z_E", "z_F"]][:4].mean(axis=1), atol=1e-4)
    out2 = t["tmp"] / "s2.tsv"
    cli.predict_pairs_main(common + ["--out", str(out2), "--method", "gls", "--unmapped", "na"])
    d2 = pd.read_csv(out2, sep="\t")
    assert np.allclose(d2.score[:4], d.gls_t[:4], rtol=1e-5) and d2.score[4:].isna().all()


def test_genome_wide_cli(toy_files):
    t = toy_files
    out = t["tmp"] / "gw.tsv"
    d = cli.genome_wide_main(["--depmap", str(t["dep"]), "--hgnc", str(t["hgnc"]), "--top", "100",
                              "--k", "0", "--q", "5", "--dim", "8", "--null-pairs", "2000", "--out", str(out)])
    assert len(d) == 100 and d.zavg_BEF.is_monotonic_decreasing and (d["rank"] == np.arange(1, 101)).all()
    lab = t["lab"]
    ga = np.array([t["ent"].index(str(x)) for x in d.entrez_a])
    gb = np.array([t["ent"].index(str(x)) for x in d.entrez_b])
    assert ((lab[ga] == lab[gb]) & (lab[ga] >= 0)).mean() > 0.8          # top pairs = planted modules (chance ~1%)


def test_evaluate(toy_files):
    rng = np.random.default_rng(0)
    n = 400
    ids = [f"P{i:05d}" for i in range(n)]
    cl = rng.integers(0, 40, n)
    a, b = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    keep = a < b
    a, b = a[keep], b[keep]
    y = (cl[a] == cl[b]).astype(int)
    s = y + rng.standard_normal(len(y))
    sc = pd.DataFrame(dict(uniprot_a=[ids[i] for i in b], uniprot_b=[ids[i] for i in a], score=s, mapped=1))
    sc = sc.drop_duplicates(["uniprot_a", "uniprot_b"]).reset_index(drop=True)
    lab = pd.DataFrame(dict(uniprot_a=[ids[i] for i in a], uniprot_b=[ids[i] for i in b], label=y))
    lab = lab.drop_duplicates(["uniprot_a", "uniprot_b"]).reset_index(drop=True)
    asg = pd.DataFrame(dict(protein=ids, cluster=cl))
    r = cli.evaluate(sc, lab, assignment=asg, f=0.25)
    key = dict(zip(zip(sc.uniprot_b, sc.uniprot_a), sc.score))
    ss = np.array([key[(p, q)] for p, q in zip(lab.uniprot_a, lab.uniprot_b)])
    assert np.isclose(r["ap"], average_precision_score(lab.label, ss))
    w = np.where(cl[[ids.index(p) for p in lab.uniprot_a]] == cl[[ids.index(q) for q in lab.uniprot_b]], 1.0, 4.0)
    assert np.isclose(r["ap_ht"], average_precision_score(lab.label, ss, sample_weight=w))
    # CLI with pos / neg files and a side restriction
    tmp = toy_files["tmp"]
    sc.to_csv(tmp / "sc.tsv", sep="\t", index=False)
    _write_pairs(tmp / "pos.txt", lab[lab.label == 1][["uniprot_a", "uniprot_b"]].values)
    _write_pairs(tmp / "neg.txt", lab[lab.label == 0][["uniprot_a", "uniprot_b"]].values)
    asg.assign(side=np.where(np.arange(n) % 2 == 0, "test", "dev"), mapped=True).to_csv(tmp / "asg.csv", index=False)
    res = cli.evaluate_main(["--scores", str(tmp / "sc.tsv"), "--pos", str(tmp / "pos.txt"), "--neg",
                             str(tmp / "neg.txt"), "--assignment", str(tmp / "asg.csv"), "--side", "test",
                             "--out", str(tmp / "r.json")])
    assert res["ht_f"] == 0.5 and json.loads((tmp / "r.json").read_text())["n_pos"] == res["n_pos"]
