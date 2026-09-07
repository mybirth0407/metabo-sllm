# Phase 2 — RQ1 (generalization): annotated bibliography and evidence pack

Retrieved and verified 2026-09-08 by the bibliography/source-verification agent. Every entry was fetched at its primary page (arXiv abstract/HTML, PMLR, PMC, bioRxiv, HF/GitHub). Where a journal page blocked fetching, the arXiv/PMC mirror was used and the venue stated only if confirmed. Excluded as unverifiable: "Efficient Ensemble..." (Li et al. 2025) and the ChemBERTa-3 *Digital Discovery* journal version (only the ChemRxiv/Zenodo record confirmed). Tiers: T1 peer-reviewed, T2 preprint, T3 gray.

## A. Annotated bibliography (APA 7)

1. **Biderman, D., Portes, J., Gonzalez Ortiz, J. J., et al. (2024).** LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2405.09673 — T1. Code/math LoRA vs full FT on Llama-2-7B. Target-domain accuracy rises monotonically with rank: code IFT HumanEval r=16 0.358, r=64 0.417, r=256 0.498 (full FT 0.497); code CPT r=16/64/256 = 0.162/0.196/0.224 vs 0.263. Full FT with weight decay (5e-5, 1e-4) or attention dropout (0.05, 0.1) did not beat LoRA on the learn/forget trade-off. Performance kept improving with more tokens/epochs. Recommends all modules, α=2r, LR sweep 1e-5–5e-4 taking the highest stable value; LoRA is "more sensitive to learning rates," optimal LR ~10× full FT. Corroborated (T3, vendor blog) by Schulman/Thinking Machines Lab (2025), "LoRA Without Regret," https://thinkingmachines.ai/blog/lora/: adapters "fall off the minimum-loss curve when the adapter runs out of capacity"; optimal LR roughly rank-independent; MLP-layer LoRA matters most.

2. **Kalajdzievski, D. (2023).** *A rank stabilization scaling factor for fine-tuning with LoRA* (arXiv:2312.03732). https://arxiv.org/abs/2312.03732 — T2. With 1/r scaling, LoRA models converge to a similar loss irrespective of rank (gradient collapse); with 1/√r (rsLoRA) loss improves with rank (LoRA r=4 tuned 1.863 ppl vs rsLoRA r=2048 1.836). No generalization analysis. We use rsLoRA r=64: rank is not gradient-throttled.

3. **Lin, Y., Ma, X., Chu, X., Jin, Y., Yang, Z., Wang, Y., & Mei, H. (2024).** *LoRA Dropout as a sparsity regularizer for overfitting control* (arXiv:2404.09610). — T2. Random masking of LoRA input/output neurons (p=0.5) with a sparsity-based generalization bound; test-time ensemble N=4. DeBERTaV3-base GLUE avg 88.50→89.54 (+1.04; RTE +2.89); SQuAD v1.1 EM 86.6→88.2. Only NLU evidence.

4. **Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024).** DoRA: Weight-decomposed low-rank adaptation. *PMLR 235* (ICML). https://proceedings.mlr.press/v235/liu24bn.html — T1. Rank sweep (Table 15, fixed hyperparameters): LoRA r=4/8/16/32/64 = 39.5/40.7/70.9/74.7/65.8 vs DoRA 61.9/77.9/77.5/78.4/72.1 — both degrade from r=32 to r=64. The only verified hint that r=64 can underperform r=32.

5. **Lee, Y.-A., Ko, C.-Y., Chen, P.-Y., & Yeh, M.-Y. (2026).** *Learning rate matters: Vanilla LoRA may suffice for LLM fine-tuning* (arXiv:2602.04998). — T2. "Once learning rates are properly tuned, all methods achieve similar peak performance (within 1–2%)" for rsLoRA, DoRA, LoRA+, PiSSA. Same conclusion in He et al. (2026), *A unified study of LoRA variants* (arXiv:2601.22708, T2). LoRA+ (Hayou, Ghosh & Yu, 2024, *PMLR 235*, https://proceedings.mlr.press/v235/hayou24a.html, T1) reports 1–2% from η_B = λη_A with λ task-sensitive. We have not swept LR.

6. **Goldman, S., Bradshaw, J., Xin, J., & Coley, C. W. (2023).** Prefix-tree decoding for predicting mass spectra from molecules. *NeurIPS 2023* (arXiv:2303.06470). — T1. NIST20: 35,129 spectra / 24,403 structures; collision energies pooled per compound-adduct; sqrt intensities. Cosine 0.726 (random). Optuna (50 trials): SCARF-Thread dropout 0.3, weight decay 1e-6; SCARF-Weave dropout 0.2, weight decay 0.

7. **Goldman, S., Li, J., & Coley, C. W. (2024).** Generating molecular fragmentation graphs with autoregressive neural networks. *Analytical Chemistry, 96*(8), 3419–3428. https://arxiv.org/abs/2304.13136 — T1. NIST20 random→scaffold cosine: ICEBERG 0.727→0.699, SCARF 0.726→0.669, MassFormer 0.721→0.682, FixedVocab 0.704→0.658, CFM-ID 0.412→0.411. Graph models lose 0.03–0.06 on scaffold split.

8. **Murphy, M., Jegelka, S., Fraenkel, E., Kind, T., Healey, D., & Butler, T. (2023).** Efficiently predicting high resolution mass spectra with graph neural networks. *PMLR 202* (ICML). https://proceedings.mlr.press/v202/murphy23a.html — T1. K=10⁴ formulas (from 188,349 product-ion and 351,165 neutral-loss formulas) explain 98% of ion counts in the structure-disjoint test split. Covariates (CE, instrument, adduct) embedded; dropout 0.1, weight decay 1e-5, 100 epochs, best-validation checkpoint. Fragment formulas recur across structures.

9. **Young, A., Röst, H., & Wang, B. (2024).** Tandem mass spectrum prediction for small molecules using graph transformers. *Nature Machine Intelligence, 6*, 404–416 (arXiv:2111.04824). — T1. NIST: 375,406 spectra / 22,105 compounds, 11.33 NCEs per compound; NCE as covariate; each CE a separate example. Graphormer pretraining needed for stability at scale. LR/weight decay/dropout tuned; early stopping on 10% validation; 20 epochs linear decay.

10. **Young, A., Wang, F., Wishart, D. S., Wang, B., Greiner, R., & Röst, H. (2025).** FraGNNet: A deep probabilistic model for tandem mass spectrum prediction. *Transactions on Machine Learning Research* (arXiv:2404.02360). — T1. NIST20 InChIKey→scaffold cosine: FraGNNet 0.736→0.678, ICEBERG 0.707→0.636, MassFormer 0.639→0.562, GrAFF-MS 0.596→0.520. CE via Fourier features; dropout 0.1, weight decay 1e-5, LR 1e-3 cosine, 150 epochs, best-val checkpoint; annotation recall 0.81 / precision 0.98. Scaffold drop of 0.06–0.08 is normal.

11. **Nowatzky, Y., et al. (2025).** FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events. *Nature Communications*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11889238/ — T1. 74,401 spectra / 10,692 compounds; CE continuous covariate; loss weighted 1/(#spectra per compound); LR reduce-on-plateau, 200 epochs, lowest-val-loss checkpoint. Test cosine 0.81 vs ICEBERG 0.72; CASMI-22 0.29 vs CFM-ID 0.38; quality declines linearly for Tanimoto < 0.6.

12. **Wang, R., Manjrekar, M., et al. (2025).** *Neural spectral prediction for structure elucidation with tandem mass spectrometry* (bioRxiv 2025.05.28.656653). — T2. ICEBERG 2.0: NIST20 top-1 25.1%→40.0%. Ablation (top-1 / cosine, full 40.0 / 0.785): remove CE positional encoding 38.4 / 0.765; sqrt normalization 39.2 / 0.779; spectral-entropy loss 38.8 / 0.772; contrastive FT 37.2 / 0.756. Dropout 0.2 / 0.1.

13. **Wang, R.-X., Wang, R., & Coley, C. W. (2026).** *GLACIER: Rethinking mass spectrum prediction as an object detection problem* (arXiv:2606.29161). — T2. DETR-style learned queries + Hungarian matching over Graphormer. NIST20 cosine random 0.838 (0.802 with contrastive FT) vs ICEBERG 2.0 0.773; scaffold 0.788 (0.756) vs 0.733; MassSpecGym 0.58 vs 0.47. "If we rely on the intensity objective alone, the model does not converge at all"; MAGMa supervision weighted from 1 with 0.9 decay (test loss 0.177) beats always-MAGMa (0.243).

14. **Bushuiev, R., Bushuiev, A., et al. (2024).** MassSpecGym: A benchmark for the discovery and identification of molecules. *NeurIPS 2024 Datasets & Benchmarks* (arXiv:2410.23326). — T1. 231k spectra / 29k molecules; MCES≥10 split. Simulation baselines: FraGNNet 0.52, fingerprint FFN 0.25, GNN 0.19. MolSpecFlow (Wang, Y., et al., 2026, bioRxiv 10.64898/2026.01.28.702438, T2), SMILES-token flow model pretrained on 100M molecules + 42M spectra, reports 0.63.

15. **Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., & Das, P. (2022).** Large-scale chemical language representations capture molecular structure and properties. *Nature Machine Intelligence* (arXiv:2106.09553). — T1. Bidirectional MLM, linear attention + rotary, 1.1B SMILES. MoleculeNet scaffold ROC-AUC (BBBP/Tox21/ClinTox/HIV/BACE/SIDER): XL 93.7/84.7/94.8/82.2/88.2/69.0 vs MolCLR 73.6/79.8/93.2/80.6/89.0/68.0; public 10%+10% checkpoint (46.8M, Apache-2.0, HF `ibm-research/MoLFormer-XL-both-10pct`) 91.5/84.5/94.6/81.3/86.6/68.9. Other verified public bidirectional encoders: SMI-TED 289M (Soares et al., 2025, *Communications Chemistry, 8*, 193; arXiv:2407.20267) and ChemBERTa-3 (Singh et al., 2025, ChemRxiv 10.26434/chemrxiv-2025-4glrl).

16. **Kristiadi, A., Strieth-Kalthoff, F., Skreta, M., Poupart, P., Aspuru-Guzik, A., & Pleiss, G. (2024).** A sober look at LLMs for material discovery. *PMLR 235* (ICML). https://arxiv.org/abs/2402.05015 — T1. "Chemistry-focused features (T5-Chem, MolFormer, and even fingerprints) are better than general-purpose LLM features" (T5, GPT-2, LLaMA-2-7B); LoRA fine-tuning of the LLM helps on average but inconsistently.

17. **BehnamGhader, P., Adlakha, V., Mosbach, M., Bahdanau, D., Chapados, N., & Reddy, S. (2024).** LLM2Vec: Large language models are secretly powerful text encoders. *COLM 2024* (arXiv:2404.05961). — T1. Enabling bidirectional attention without training hurts (S-LLaMA-1.3B 86.10→76.50); adding masked-next-token-prediction recovers and surpasses causal (90.51).

18. **Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D., & Hestness, J. (2025).** Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs. *ICLR 2025* (arXiv:2502.15938). — T1. Linear decay-to-zero beats 10× decay. Hägele et al. (2024, *NeurIPS 2024* spotlight, arXiv:2405.18392, T1): cosine's endpoint is tied to its planned length; constant LR + cooldown produces a sharp late loss drop; SWA helps. Neither tests adapters.

Also verified: Lee, C. E., et al. (2025). SimSon. *Bioinformatics, 41*(5) — randomized-SMILES contrastive positives beat masked SMILES by 11.51% avg. Brinkmann, H., et al. (2025). *Digital Discovery, 4*(10), 2752 — SMILES enumeration "always in the top-two" augmentations (T1).

## B. Evidence table

| Mechanism | Source | Reported effect | Task | Applicability |
|---|---|---|---|---|
| Higher adapter rank (rsLoRA) | 1, 2 | r16→r256: +0.14 HumanEval; rsLoRA loss improves with rank | LLM code | adaptable |
| Rank non-monotonic | 4 | LoRA r32 74.7 → r64 65.8 (fixed hparams) | commonsense | adaptable (confound check) |
| LR sweep > variant choice | 5 (+He, Hayou) | variants within 1–2% once LR tuned | LLM | direct |
| LoRA dropout + TTA | 3 | +1.04 GLUE, +1.6 SQuAD EM (p=0.5, N=4) | NLU | adaptable |
| Weight decay on adapters | 1, 6, 8, 10 | no isolated delta; MS models use 1e-5–1e-6 or 0 | MS/LLM | adaptable, size unknown |
| Schedule length / decay-to-zero | 18 (+Hägele) | D2Z 80 TPP < 10×-decay 200 TPP loss | LM pretrain | adaptable |
| CE as continuous encoding | 12, 10, 9, 11 | −1.6 pt top-1, −0.020 cosine without CE encoding | MS | direct |
| Per-compound loss weighting | 11 | used; no ablation | MS | adaptable |
| Sqrt intensity / entropy loss | 12 | −0.006 / −0.013 cosine when removed | MS | direct |
| Scaffold-split drop baseline | 7, 10, 13 | 0.03–0.08 cosine drop for graph SOTA | MS | direct (calibration) |
| Formula sparsity | 8 | 10⁴ formulas explain 98% ion counts, structure-disjoint | MS | direct |
| Heuristic fragment supervision | 13 | intensity-only slot decoder does not converge | MS | not applicable (MAGMa banned) |
| Chemistry-pretrained encoder | 15, 16 | MoLFormer > GNNs on 3/6 scaffold tasks; chem features > general LLM features | property/BO | adaptable |
| Bidirectionalization | 17 | naive: −9.6 pts; with MNTP: +4.4 pts | NLP | adaptable |
| SMILES randomization | SimSon, Brinkmann | +11.5% vs masking; top-two augmentation | property/generative | adaptable |
| Early stopping on val | 8, 9, 10, 11 | universal practice; no delta | MS | direct |

## C. Contradictions and gaps

- **Rank.** Biderman/rsLoRA/Thinking Machines: higher rank = more learning, plateau = capacity. DoRA Table 15: LoRA and DoRA both drop from r=32 to r=64. Lee and He: apparent variant/rank effects are largely LR confounds. No source measures rank vs held-out scaffold generalization on a scientific structured task.
- **Regularization.** Lin reports +1 pt from heavy LoRA dropout; Biderman finds weight decay/dropout weaker than LoRA itself for forgetting; MS predictors pick dropout 0–0.3 and weight decay 0–1e-5 by search, none ablates them. No source isolates adapter weight decay as a generalization lever.
- **CE handling.** SCARF/ICEBERG-1 merged CEs and still reached 0.727; ICEBERG 2.0, FraGNNet, MassFormer, FIORA use CE-as-input. Only quantified delta ~0.02 cosine.
- **Encoder.** MoLFormer/Kristiadi favor chemistry-pretrained bidirectional encoders; MolSpecFlow exceeds FraGNNet on MassSpecGym but with 42M-spectrum pretraining — no source tests encoder choice for MS prediction at fixed data.
- **Unanswered for our setting:** adapter rank vs generalization with ~16k precursor formulas; "formula seen in train" vs unseen stratification (GrAFF-MS's 98% coverage is the only proxy); whether extending a cosine schedule moves an adapter plateau; whether a set decoder trained without heuristic fragment labels converges to the supervision ceiling.

## D. Ranked interventions

1. **LR sweep and schedule-horizon test before regularizing** (sources 5, 1, 18, Hägele). lr ∈ {0.75, 1.5, 3}×1e-4 with (a) the current cosine and (b) a 2× horizon or constant+cooldown. Confirm: valid bag_hit/cos@100 at epochs 3–6 differ across arms and the 2×-horizon arm's epoch-6 point is not on the same plateau. If valid bag_hit moves >0.02 while train bag_hit does not, the plateau is schedule-limited, not overfit.
2. **Adapter regularization: LoRA dropout p∈{0.1, 0.3} (+N=4 test-time ensemble) and weight decay 1e-5 on adapters and decoder; rank {32, 128} as confound control** (sources 3, 4, 8, 10). Confirm: train−valid bag_hit gap narrows from 0.70−0.385 with valid bag_hit ≥ 0.385 and cos@100 ≥ 0.51.
3. **Data/representation levers** (sources 12, 11, 15, 16, SimSon). (a) Per-compound loss weighting 1/(#CEs) plus randomized-SMILES input augmentation; (b) an encoder-swap ablation with MoLFormer-XL-both-10pct (frozen + LoRA, graph-free) — the only lever with Tier-1 evidence of scaffold generalization per se. Confirm on the scaffold valid fold with bag_hit stratified by precursor-formula seen/unseen in train.
