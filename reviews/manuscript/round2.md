# Review round 2

Independent agent, 2026-10-27. Evidence: `results/`, DATA/DESIGN/RESEARCH_TREE, release README/AGENTS.md, git log.

**Overall.** Most round-1 items are fixed and nearly every number matches its source. Four new factual errors were introduced. Recommendation: minor revision.

## A. MUST-FIX

**N1. The Discussion contradicts Fig 3.** It says "Two choices had the largest effects: standardising genes … and averaging z-scores". In Fig 3, the neighbour cosine adds +0.036 and row-z adds +0.034, but averaging adds only +0.021 (the same as the ridge). Write instead: "The largest steps were gene standardisation (+0.061) and the neighbourhood score with row-z (+0.070)."

**N2. Results, Baselines: "the same search budget as our methods" is false.** The baselines had 2 or 8 variants each (S2 §1); our methods had about 300. Write: "selected by the same rule, from 2–8 variants".

**N3. Methods, Candidate selection: "the best single score (B)" is wrong.** E scored 0.463 against B's 0.448. B was kept because E's gain was not significant under the replacement rule (S2 §5). State this.

**N4. "Before any method development" (Results) and "before method development" (Introduction) are inaccurate.** The M1 baseline run and the reviewers' whitening and PC-removal probes (S2 Tables D–E) came before the split. Write "before the method screen".

**N5. The agent setup is described inconsistently.**
- AGENTS.md:
  - "about 1,100 trials" should be 845 started and 844 completed.
  - "pre-registered" should be "pre-specified".
  - "Everything after that was done by agents" omits the "continue" message and the manuscript prompt.
  - "about 25" sub-agents should be 33 (S3).
- Introduction: "humans set the task in one prompt" contradicts Methods, which lists two prompts.

**N6. M15 is still open.** DATA.md lists hu.MAP 3.0 as "assumed; to confirm" and the HGNC download date as "unknown". The release README states the hu.MAP files' CC BY 4.0 licence as fact. Confirm these, or flag them as unconfirmed.

## B. Round-1 items

| Item | Status | Remaining |
|---|---|---|
| M1, M3, M4, M6, M7, M8, M9, M10, M11, M13, M14 | fixed | M7: see N3 |
| M2 | partly | Runtime bases are mixed: B's 28 s is the old tool total; zavg_BEF is 118 s (process) vs 137 s (tool); only one GLS run is shown (19 s; the other was 24 s); E alone (125 s) exceeds zavg_BEF (118 s) and is not explained |
| M5 | partly | "pre-registered" remains in S2 §5, §6, §7.2 and in AGENTS.md |
| M12 | fixed | The main-text list omits agents 32–33 |
| M15 | partly | N6 |
| M16 | fixed | Placeholders remain: authors, funding, competing interests, URLs, ref 6 DOI |
| B1 | partly | The PC-removed Pearson reference is not in Table 2; caveat missing (D1) |
| B2–B4, B6, B7, B10, B11, B13, B15 | fixed | B7: numbers differ from S2 (C) |
| B5 | mostly | No reason given for Louvain or for 5 folds |
| B8 | partly | S2 §1 still says "carried no signal" |
| B9 | partly | "This reflects…" is stated as fact |
| B12 | not fixed | N1 |
| B14 | partly | Retrospectives were held at only 2 milestones, although the prompt asked for one at each; not disclosed |

## C. Factual discrepancies

| Location | Stated | Correct (source) |
|---|---|---|
| Discussion ¶2 | averaging is a "largest effect" | +0.021 < +0.036, +0.034 (Fig3 CSV) |
| Results, Baselines | "same search budget" | 2/8 vs ~300 (S2 §1) |
| Methods, Candidate selection | "best single score (B)" | E 0.463 > B 0.448 (phaseB_v2) |
| Controls | monotone LGBM 0.145 | S2 Table M: mean 0.146 |
| Supervision; Discussion | ~1,800 training positives per fold | 1,808–1,981 (clusters_summary); S2 §8 says 1,900 |
| S2 §7.4 | "about 2.5×" | 2.6× (0.024/0.0091) |
| S2 §5 | E macro AP 0.394 vs 0.341 | Table 2 gives 0.406 vs 0.351; state the basis |
| Methods, IFT draw | "20% of the total" | 20% of the Kish n_eff of all clusters (25.2) |

Correct: Abstract, Author summary, Tables 1–2, all IFT numbers, post-hoc values, S2 Table O, controls, coverage, trial and S3 counts.

## D. Overstatements, caveats, credit

1. **Post-hoc reference.** "Above GLS" needs a qualifier. Its macro AP is below GLS on both sets (dev 0.342 vs 0.349; IFT 0.320 vs 0.369), and its CI against GLS excludes 0 in only 2 of 5 seeds. Add this to Results and to S2 §9.
2. **Author summary.** "Concentrated in larger complexes" needs "in development data". On the IFT, the gain without the largest component (+0.120) exceeded the gain on all pairs.
3. **Results.** "This reflects…" → "This probably reflects…".
4. **Discussion.** "Caught most of them" → "caught several".
5. **Methods.** "Each headline result" was independently verified, but the revision analyses were not. Say so.
6. **AI framing** is clear from the title onward (except N5).
7. **Credit.** The credit for each method component matches `references_verified.md`. The Kish section is unverified. Optionally, cite Bengio et al. 2004 for the Nyström out-of-sample extension.

## E. PLOS compliance

- The abstract has 300 words, exactly at the limit; trim about 10. The author summary has 198.
- Captions are placed correctly. Figures, tables and S1→S3 are cited in order. References 1–37 are in Vancouver style, numbered in order. No `{key}` placeholders remain.
- Abbreviations: HGNC is not defined, and AP is never expanded.
- The Fig 2C right panel and the Table 2 "without top-10" column are pooled-OOF values; their captions should say so.
- Cosmetic: overlapping labels in Fig 3 and Fig 5A.

## F. Clarity edits

| Location | Original | Pattern | Replacement |
|---|---|---|---|
| Abstract ¶4 | "In a single evaluation on an internal test set of held-out complexes, specified in advance, it scored…" | 6 | "In one pre-specified evaluation on held-out complexes, it scored 0.345 against 0.242 (GLS), 0.146 (Spearman) and 0.134 (Pearson)." |
| Results, AUROC | "It sharpens the top of the ranking at the cost of the global ranking." | 1, 2 | "It ranks the highest-scoring pairs better but the full list worse." |
| Results, IFT | "Counting the 9 positive and 2,352 negative IFT pairs with an unmapped protein at the minimum score, AP was 0.332" | 6, 4 | "When the 9 positive and 2,352 negative pairs that involve an unmapped protein were added at the minimum score, AP was 0.332" |
| Results, verification | "with its own code; they were identical to 10⁻¹⁶" | 4 | "with its own code and obtained values identical to within 10⁻¹⁶" |
| Discussion ¶1 | "A label-free change of perspective, from direct co-essentiality to shared co-essentiality neighbourhoods, improved…" | 1, 2, 6 | "Scoring shared co-essentiality neighbourhoods instead of direct co-essentiality improved, without labels, …" |
| Methods, IFT | "Draws whose effective number of clusters fell below 20% of the total were rejected." | 1 | "A draw was rejected if its effective number of clusters was below 20% of that of all clusters (25.2)." |
