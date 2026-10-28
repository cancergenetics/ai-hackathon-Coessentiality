# Round 2 review: external PI (statistics), DESIGN v2

I computed no label-based metrics. I computed only the structure of the positive graph (scratch script, not committed):
- The giant component (144 proteins, 848 positives) splits into 7 Louvain communities.
- 116 of its 848 positive edges cross communities.
- After the split there are 238 clusters. The largest holds 9.8% of positives, and Kish n_eff is about 25 (about 10 before the split).

## 1. Round-1 proposals
Most are adopted. The gaps:
- **Reweighting (Stats P1-2), only half adopted.** Uniform reweighting of negatives leaves within-cluster negatives enriched about 5× (see §2.2).
- **Missing statements:**
  - an intersection–union test across the 3 baselines, with no correction (Stats P1-4);
  - "the declared method is reported even if it fails on IFT; no swap" (ExtPI C3);
  - a refit on all train pairs.
- **Rejected without a logged reason:** RPCA/onion+SNF and an RPCO baseline (Prior P2/P3), lambdarank (Alg P5), r vs Fisher-z across NaN patterns (Alg P2), and gene-effect rescaling. Rejecting them on cost is defensible; log one line each.
- **Wrong unit in the power check.** It counts positives (≥700). Power depends on cluster n_eff and on paired correlation (see §2.4).

## 2. New problems in v2
**2.1 Louvain clustering vs hu.MAP.**
- Only one component exceeds 40 proteins. Say "the giant component is split into seeded Louvain communities", pin the networkx version, and commit the cluster file and its hash.
- With a hard partition, "proteins in both sides" cannot occur. Delete that sentence.
- The IFT is protein-disjoint, which is stricter than hu.MAP's complex split, where shared proteins appear in both train and test. As a result the IFT under-rewards protein propensity. That is conservative, but state it.

**2.2 Prevalence reweighting is mis-specified.**
- The target composition is the full train universe, because the withheld test is built the same way.
- Inclusion probabilities are 1/K (IFT: 0.25) for within-cluster pairs and about 1/K² (IFT: 0.0625) for between-cluster pairs.
- Use Horvitz–Thompson weights: weight between-cluster pairs, including the 116 cross positives, by K relative to within-cluster pairs (4 on IFT), then normalise.
- Apply the same rule to dev and IFT. The weights depend only on the frozen clusters.

**2.3 Bootstrap.**
- The weights m_c (within) and m_c1·m_c2 (between) preserve the expected composition. Acceptable.
- On IFT, about 60 clusters (n_eff ≈ 6–8) and a ratio statistic mean percentile CIs will under-cover.
- Before the IFT look, check coverage on the planted toy with matched cluster sizes. If coverage is below 90%, use BCa or a cluster jackknife.
- Pre-register the one-sided p as the share of bootstrap ΔAP ≤ 0.

**2.4 "Clearly beats": feasibility and power.**
- (ii) State that the ≥10% margin applies to the point estimate and the CI to Δ > 0. A lower bound above 10% would be infeasible.
- The power script should take the paired ΔAP variance from dev cluster resamples (the probe gains of +18–23% are optimistic) and report P(pass ii) at true gains of 10%, 15% and 20%. If P < 0.8 at 15%, enlarge the IFT to 30–35% now.
- Holm puts the first candidate at α = 0.025. Use **fixed-sequence** testing instead: test the declared primary at 0.05, and test the second only if the primary passes.
- (iii) "Macro per-cluster AP not worse" fails about 50% of the time between equal methods, because many clusters have 1–3 positives. Use non-inferiority: the one-sided 95% lower bound of Δ is above −5% relative.
- (iii) Define "without the largest component" as "without the giant component's communities".
- (i) is only a stability check: the seed repeats share one dev set.

**2.5 IFT draw.**
- Sequential random fill can put 2–3 of the top clusters into the IFT, giving n_eff ≈ 3.
- Stratify by size tier: split the top 10 clusters about 1:3, and assign the rest at random.
- Accept the draw only if IFT n_eff ≥ 0.2 × total n_eff, and freeze it before any scoring.

**2.6 Comparability.** Label-free methods are selected on pooled AP and supervised ones on per-fold mean AP. Use the HT-weighted per-fold mean for every cross-family comparison and when choosing the primary.

## 3. Method priorities
GLS reaches P@R0.1 = 0.87 but P@R0.5 = 0.03. AP is earned at recall 0.1–0.4, in essential or mixed pairs.
1. **High: Phase A sweep, then a frozen label-free ensemble.**
   - Components: ridge τ (± top-k), genome-wide row-z, and sparse second-order similarity at q ∈ {0.5, 1, 2, 5}%. Compute all on the *whitened* matrix.
   - Combine by z-averaging, with weights set on dev.
   - Declare it the primary candidate.
   - Compute z-moments genome-wide and fix them, never from the scored file, so batch invariance holds.
2. **High: Phase B as reliability-aware stacking.**
   - Inputs: a few layers, plus min/max of gene SD, mean effect and NaN fraction. Optionally add per-gene replicate noise from the SGE models screened more than once (label-free).
   - Monotone constraints; select by HT-AP.
3. **Medium: Phase D.** Stack 1 and 2 within CV as the secondary candidate.
4. **Low: Phase C.** Learned spectral filter only, after first trying a finer τ grid. Defer the InfoNCE projection.
5. **Remove from the critical path:** A2 and A5.

## 4. Priority
High: §2.2, §2.4, §2.5, §3, and the missing statements in §1. Medium: §2.3. Low: logged rejection reasons and the §3.1 wording.
