# Phase 2 — RQ3 (residual identity errors): annotated bibliography and evidence pack

Retrieved and verified 2026-09-08. Every entry fetched (arXiv/PMLR/NeurIPS/ACL/bioRxiv) and PDFs converted locally. Code claims cite ms-pred commit ed8311f (2026-09-04). Cosine conventions differ per paper and are stated per entry. COI: entries 2, 4, 6, 7, 8, 15 are Coley-lab papers benchmarking against their own reimplementations; entries 10–12 share authors (Sugiyama/Niu/Feng/Lv).

## A. Annotated bibliography (APA 7)

1. **Murphy, M., Jegelka, S., Fraenkel, E., Kind, T., Healey, D., & Butler, T. (2023).** Efficiently predicting high resolution mass spectra with graph neural networks. *PMLR, 202*, 25549–25562. https://arxiv.org/abs/2301.11419 — T1. Vocabulary F̂(P)=P̂ ∪ (P−L̂): K=10⁴ product+loss formulas chosen greedily by summed peak height (Alg. 1), explaining 98% of ion counts in NIST-20 train (193,577 product / 348,692 loss formulas observed). Products+losses beat either alone (Fig. 2, qualitative). Loss = peak-marginal CE (Eq. 5): −Σ yᵢₙ log Σ_{f∈F̂ᵢₙ} ŷ_f — intensity-weighted marginal over formulas within tolerance, the closest published analogue of our bag likelihood. Metric: matchms CosineHungarian 0.05 Da; NIST-20 test .70 vs NEIMS .60, CFM-ID .52; all methods "struggle" with large molecules.

2. **Goldman, S., Bradshaw, J., Xin, J., & Coley, C. W. (2023).** Prefix-tree decoding for predicting mass spectra from molecules. *NeurIPS 36*. https://arxiv.org/abs/2303.06470 — T1. Labels: MAGMa depth-3 bank, each peak → nearest formula within 20 ppm (ms-pred default 10 ppm with 200-Da floor: `data_scripts/forms/01_assign_subformulae.py:77,272-283`). Table 1 coverage@10/30/300/1000, NIST20: Frequency .173/.275/.659/.830; LSTM autoregressive .204/.262/.309/.317; SCARF-D (difference-only) .248/.425/.839/.941; SCARF-F (forward-only) .249/.476/.855/.943; SCARF (gated both, Eq. 3) .308/.552/.907/.968. Table 2 (0.1-Da bins, top-100): NIST20 SCARF .726 > FixedVocab .704; **NPLIB1 FixedVocab .568 > SCARF .536**; retrieval NIST20 SCARF .187 vs FixedVocab .172, NPLIB1 FixedVocab .193 vs SCARF .135. Table A8: SCARF retrieval falls from .191 (0–200 Da) to ~.165 (≥300 Da) while FixedVocab rises to .220 (600–700 Da). Inference: per-element beam, top-500 prefixes (`scarf_model.py:484,582,596`).

3. **Young, A., Wang, F., Wishart, D. S., Wang, B., Greiner, R., & Röst, H. (2025).** FraGNNet. *TMLR*. https://arxiv.org/abs/2404.02360 — T1. Classification over an enumerated heavy-atom fragmentation DAG (d=4) with H range ±4; median support 679 formulae. Out-of-support term (Eq. 7–8, 10 ppm): true OS mass .098/.110; predicted within δ_TV .057/.067. Table 1 (NIST20 [M+H]+, CE-merged): C_BIN(0.01 Da) D4 .736/.678, ICEBERG .707/.636, GrAFF-MS reimpl. .596/.520. Annotation (scaffold): FraGNNet AR .81/AWR .90/AP .98; GrAFF-MS AR .95/AP .61; ICEBERG AR .66/AP .99. ICEBERG+OptFrag .732 vs .707 ⇒ autoregressive fragment generation is an error source.

4. **Goldman, S., Xin, J., Provenzano, J., & Coley, C. W. (2024).** MIST-CF. *JCIM, 64*(7), 2421–2431. https://doi.org/10.1021/acs.jcim.3c01082 — T1. Energy-based ranking over ≤256 candidates (FastFilter recovers the true formula 99% in top-256; >15% of spectra have >5,000 candidates at 10 ppm). Top-1 NPLIB1 .741 vs Transformer .639, FFN .635. "All models decrease in accuracy as the masses of compounds increase … growing number of plausible candidates". Peaks 1/5/10/20/50 → .673/.729/.754/.756/.774 (weak peaks add little).

5. **Wang, Y., et al. (2026).** MolSpecFlow. *bioRxiv*. https://doi.org/10.64898/2026.01.28.702438 — T2. Graph-free (randomized SMILES tokens). Expected-formula loss (Eq. 9–10) and expected-mass loss (Eq. 11) on the molecule-token distribution. MassSpecGym simulation cosine .63 vs FraGNNet .52. Ablation (de novo only): w/o formula embedding top-1 3.11→1.85%, w/o mass embedding →2.15%.

6. **Goldman, S., Li, J. X., & Coley, C. W. (2024).** ICEBERG. *Anal. Chem., 96*(8), 3419–3428. — T1. H-shift head: 13 intensities {0, ±1…±6 H} per fragment restricted by bonds broken (`common/misc_utils.py:277`, `iceberg/inten_model.py:239`). Cosine (0.1 Da) NIST20 .727/.699 vs SCARF .726/.669. Coverage .754 < SCARF .807: "rigid fragment-grounding causes our model to miss certain lower intensity peaks". No H-shift ablation.

7. **Wang, R., Manjrekar, M., … Coley, C. W. (2025).** ICEBERG 2.0. *bioRxiv*. https://doi.org/10.1101/2025.05.28.656653 — T2. Spectral-entropy loss ("Emphasizing low-intensity peaks aids the model"); no numeric entropy-vs-cosine ablation. ±1 H per bond break, ≤6. NIST20 top-1 40.0% random / 33.5% scaffold.

8. **Wang, R.-X., Wang, R., & Coley, C. W. (2026).** GLACIER. *arXiv:2606.29161*. — T2. DETR-style queries + Hungarian (`glacier/joint_model.py:1235`), 13 H states, breakpoints k≤3. MAGMa teacher forcing decayed by λ; λ=0 "fails to converge"; λ=0.9 test loss .177 vs .243 constant. Retrieval top-1 MassSpecGym 69.95 vs ICEBERG 2.0 63.95, MolSpecFlow 55.32, FraGNNet 46.64; NIST20 scaffold .460 vs .316.

9. **Min, S., Chen, D., Hajishirzi, H., & Zettlemoyer, L. (2019).** A discrete hard EM approach for weakly supervised question answering. *EMNLP-IJCNLP*, 2851–2864. https://aclanthology.org/D19-1284/ — T1. MML −log Σ P(zᵢ) vs hard-EM (min over Z); annealing: MML w.p. min(t/τ,1), τ=4–20K steps. Table 3 (First/MML/Hard): TriviaQA F1 64.4/64.8/66.9; NarrativeQA 55.3/55.8/58.1; DROP-BERT 42.9/39.7/52.8; NQ-open 23.6/26.6/28.8. "Gap … marginal when |Z|≤1, gradually increases as |Z| grows"; gain "particularly large when |Z|>3"; annealing +2.2 F1.

10. **Lv, J., Xu, M., Feng, L., Niu, G., Geng, X., & Sugiyama, M. (2020).** Progressive identification of true labels for partial-label learning (PRODEN). *PMLR, 119*, 6500–6510. — T1. Model-weighted candidate CE with weights renormalised over the bag each epoch; algorithmically identical to RC.

11. **Feng, L., Lv, J., Han, B., Xu, M., Niu, G., Geng, X., An, B., & Sugiyama, M. (2020).** Provably consistent partial-label learning. *NeurIPS 33*. — T1. CC = −log Σ_{y∈S} p(y|x) (our loss); RC = weighted. Table 1: RC vs CC MNIST 98.00/97.87, KMNIST 89.38/88.83, CIFAR-10 77.93/75.78; under mismatched candidate generation (Table 7) CIFAR-10 68.18/56.13, KMNIST 92.81/86.67. "RC significantly outperforms CC when deep neural networks are used."

12. **Wang, W., Wu, D.-D., Wang, J., Niu, G., Zhang, M.-L., & Sugiyama, M. (2025).** Realistic evaluation of deep partial-label learning algorithms (PLENCH). *ICLR 2025 (spotlight)*. https://arxiv.org/abs/2502.10184 — T1. PLCIFAR10-Aggregate (avg 4.87 candidates): PRODEN 85.95 vs CC 80.66 vs PiCO 79.20; Vaguest: 74.95 vs 71.78.

13. **Ludwig, M., et al. (2020).** ZODIAC. *Nat. Mach. Intell., 2*, 629–641. — T1. Reranks SIRIUS top-50 via shared fragments/losses across a dataset: NIST1950 error 8.51→5.32%, largest gains >700 Da. Not applicable per-spectrum; loss-frequency prior adaptable.

14. **Dührkop, K., et al. (2019).** SIRIUS 4. *Nat. Methods, 16*, 299–302. — T1. Mass decomposition + RDBE/valence filters + fragmentation trees. Pruning rules adaptable.

15. **Coley group. (2026).** ms-pred [code], commit ed8311f. https://github.com/coleygroup/ms-pred — T3. `use_reverse` 3-way (forward, gate, reverse-indexed by precursor−count): `scarf_pred/scarf_model.py:57,202,466-476`; `ffn_pred/ffn_model.py:22,137-145`; `massformer_pred/massformer_model.py:42,212-220`; `autoregr_gen/autoregr_model.py:52,246-253`. Released configs: SCARF/FFN/GNN/MassFormer `use-reverse: true`. README:467: the GrAFF-MS baseline "replace[s] the marginal peak loss with a cosine similarity loss".

16. **Martin, M. R., & Hassoun, S. (2025).** GIF. *arXiv:2511.09571*. — T2. Prompted LLMs produce peak-labelled spectra at cosine .36. Cautionary graph-free baseline only.

## B. Evidence table

| Mechanism | Source | Reported delta | Setting | Applicability |
|---|---|---|---|---|
| Fixed vocab (products+losses) vs generation | #2, #3 | NIST20 .704 vs .726 (−.022); NPLIB1 .568 vs .536 (+.032); FraGNNet's GrAFF-MS .596 vs .736 | different reimpl. | Adaptable |
| Reverse/difference count head | #2 T1 | coverage@30 .476→.552, @300 .855→.907 (F-only→gated); D-only .425/.839 | NIST20, MAGMa labels | Direct (`scarf_model.py:466-476`) |
| Diff embedding (precursor−fragment) input | #2, #6, #8 | no isolated ablation | — | Direct |
| Bounded ±k H-shift head | #6, #3, #8 | no ablation; #3 AR .81 vs ICEBERG .66 | NIST20 | Adaptable |
| Marginal (CC/MML) → hard-EM / RC/PRODEN | #9, #11, #12 | +2–13 pts, growing with |Z|>3; RC−CC +0.1–2.2 (matched), +6–12 (mismatched); PRODEN−CC +5.3 | QA / images, bag sizes 2–8 | Direct (our loss is CC/MML) |
| Intensity-weighted identity loss | #1 Eq. 5, #3, #7 | no ablation; #6 loses low-intensity coverage; #4 peaks>20 add +.018 | — | Direct, unproven |
| Out-of-support mass | #3 | P(M^OS)≈.10–.11; δ_TV .057 | NIST20 | Adaptable |
| Enumerated candidate rescoring | #4, #3 | 99% recall @256; median support 679 | formula-from-spectrum | Adaptable |
| Per-level top-k beam | #2 code (500/level) | coverage@300 .907 vs frequency .659 | NIST20 | Direct |
| Expected-mass penalty | #5 | de novo only | MassSpecGym | Adaptable |
| Dataset-level loss prior | #13 | −3.2 pt error; largest >700 Da | LC-MS datasets | Not applicable per-spectrum |
| MAGMa teacher forcing | #8 | λ=0 fails to converge | DETR fragment masks | Not applicable — warning |

## C. Contradictions and gaps

- Vocab vs generation flips by dataset (NIST20 favours SCARF, NPLIB1 favours FixedVocab), and the two GrAFF-MS reimplementations differ by 0.11 cosine. No clean comparison exists.
- Metric conventions are incompatible across papers; only #2/#6/#7 numbers are loosely comparable to our ms-pred-style cos@100.
- Label ambiguity handling is never ablated on spectra: nearest-mass hard label (#2), uniform split (#1), marginal CE (#1) coexist without a head-to-head. PLL/hard-EM gains are from QA/vision with bag sizes 2–8; our per-peak bag-size distribution was unmeasured at retrieval time.
- Weak-peak identity: #7 asserts entropy loss helps with no numbers; #6 shows rigid grounding loses them; #4 shows peaks beyond top-20 barely change formula accuracy.
- Large precursors: #1, #2, #4, #13 agree accuracy falls with mass; remedies are loss-parametrisation (#1, #2 SCARF-D) and dataset priors (#13).
- Graph-free setting untested: no 2024–2026 paper trains a formula decoder on a frozen LM; #16 (prompting) reaches .36.

## D. Ranked interventions

1. **Gated reverse (neutral-loss) count head + diff embedding** (#2 T1; #1 §4.2). Per element, predict forward count and loss count indexed by precursor−count, gate as in `scarf_model.py:466-476`; feed counts(precursor − prefix) as input. Confirm: valid bag_hit by precursor-mass quartile, unique vs ambiguous, cos@100.
2. **Annealed hard-EM / RC-weighted bag objective** (#9, #11, #12; domain-transfer caveat). MML for τ steps, then min-over-bag or PRODEN weights renormalised within the bag. Confirm: ambiguous-bag hit (baseline .09) by bag size {1, 2–3, 4–10, >10} and mass; unique-bag hit ≥ .45; cos@100.
3. **Constrained per-level beam with valence/RDBE pruning and bag snapping** (#2, #4, #14). Top-k prefixes per element, RDBE pruning, rescoring of enumerated subformulas within tolerance for peaks the beam misses. Confirm: coverage@k, bag_hit before/after snapping, cos@100, latency vs mass.

Intensity-weighted identity loss (#1 Eq. 5) is a cheap fourth candidate without ablation evidence.
