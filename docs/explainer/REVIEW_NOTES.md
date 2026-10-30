# Review notes for the explainer

These are the discrepancies, open questions and judgement calls found while writing the explainer. Sources: the session transcripts, the git history, the results files, and a fresh run of the code.

## Discrepancies between record files and code or results

| # | Record says | Code, results or transcript say | Where the explainer stands |
|---|---|---|---|
| 1 | REPORT §9: "about 1,100 trials" | `results/trials.jsonl` holds 822 trial starts at report time (845 now). The ~1,100 figure counts JSONL lines, which include both start and end records. | Uses 822 (research phase) |
| 2 | REPORT §2: "Connected components … form 247 clusters"; DESIGN §2: "232 components" | Components: 232 over all 1,151 proteins, 241 over the 1,114 mapped ones. Clusters: 247 after splitting the giant component into 7 communities. | Uses 247 clusters |
| 3 | DESIGN v3 text: 116 cross-community positive edges | `clusters_summary.json`: 121. Probably an earlier Louvain run, from before the hash-seed fix. | Not stated |
| 4 | RESEARCH_TREE D1: the early reviewer probe only made the whitening prior "IFT-informed" | The same probe also ranked sparse second-order (0.310) and z-averaging (0.314) best. The PI forwarded these numbers to the round-2 reviewers, whose advice became the "label-free ensemble default". | JOURNEY turning point 2 describes this broader influence |
| 5 | Tree and REPORT §3 present "top-20 PC removal" as part of the gain | Removing components adds +0.004 to F on seed 0; over five seeds F scores 0.395 with or without it. It helps B slightly (0.401 vs 0.389). | Says this explicitly |
| 6 | REPORT §3: B's neighbours are the "top 0.25% … under F" | The code selects neighbours by the symmetrised per-gene z-score of F, weighted by max(V, 0). | Follows the code |
| 7 | D3: choose the combination "by 5-seed mean of nested held-out HT-AP" | `decision.json` ranks by the in-sample 5-seed mean. Nested values exist only for the selection procedure as a whole. BEF (0.4696) and BEHF (0.4685) are effectively tied either way. | Not detailed. Turning point 8 reports the nested check (0.468) |
| 8 | DESIGN never defines "macro AP" | B's held-out fail verdict depends on the definition. With all pairs touching a cluster, B fails; with within-cluster pairs only, B would pass. zavg_BEF passes under both. | Not detailed; the manuscript covers it |
| 9 | REPORT §5: supervised stacking "never better" | True for HT-AP. But the stackers had higher dev macro AP (0.415–0.418 vs 0.395). | Not mentioned in the explainer |
| 10 | Runtimes for zavg_BEF: 125 s / 137 s / 118 s; for GLS: 7.5 s / 24 s / 19 s | These come from different sessions and different definitions (cached matrix vs raw CSV). | Says "about 120–140 s" |
| 11 | LESSONS #7: three supervised families were null "after full 5-seed runs" | Phase B v1 was run on seed 0 only. | Not repeated |
| 12 | S3 Text / AGENTS.md: the only human input was the prompt and one "continue" | The transcript also has "what is the method name specifically" (18:08) and questions at 20:18–20:24, all after the report and with no effect on the method. | JOURNEY mentions them |
| 13 | docs/AGENTS.md: "process logs … stay in the private research repository" | The public release includes `reviews/` and the record files (RESEARCH_TREE, LESSONS). | Fixed in AGENTS.md |
| 14 | Retro M3a cites commit "047dd58" | The actual hash is 47dd58b. | Uses 47dd58b |
| 15 | JOURNEY dead ends (first version): "Avana-only cell lines … Worse" | `round1_data_biology.md`: Avana-flagged lines (1,048) tied with all lines after component removal; only Avana-only lines (827) were worse. Re-run with zavg_BEF on dev ([results/library_subsets](../../results/library_subsets/README.md)): tie (−0.003) and −0.039 on matched pairs. | Row corrected |

## Open questions and uncertain choices

1. **Whether the turning points are the right ones.** Two were merged and one was reframed against the original story:
   - The graph embedding (E) was launched 6 minutes *before* the verifier reported that B's gain was concentrated in large clusters. That concentration finding instead motivated the metric-learning attempt, which failed but brought F back.
   - JOURNEY says this plainly.
2. **Whether D1 belongs in the method story.** The contamination is presented as a turning point because it changed priorities. A reader might see this as overstating it. All concrete settings were chosen on dev.
3. **Choice of worked example.**
   - SMC5/6 is a development-set cluster. It illustrates the mechanism but is not evidence.
   - The hard pair, NSMCE2–SLF1, was chosen because it shows B's mechanism clearly, so it is a favourable case.
   - Pearson does better than GLS on this complex. The example therefore makes GLS look weaker than its average performance.
   - The BBSome is included as an honest failure.
4. **Mechanistic explanations are marked "likely" or "believed".** These include why gene standardisation helps and why E helps small complexes. None was tested directly.
5. **The SUMO-pathway remark in the worked example** (shared partners of SMC5 and SMC6 include SAE1, UBA2, UBE2I, RNF4 and NFATC2IP) is an observation. No enrichment test was run.
6. **Links from the HTML page to repository files** point to `github.com/cancergenetics/ai-hackathon-Coessentiality/blob/main/…`. The `docs/explainer/` files do not exist there until the release is rebuilt and pushed, so those links are dead for now. (Resolved: the release with `docs/explainer/` was pushed on 2026-10-29.)
7. **Times** are UTC, from the transcript and commits. S3 Text calls them "local times"; Irish time equalled UTC on that date, so both are right.
