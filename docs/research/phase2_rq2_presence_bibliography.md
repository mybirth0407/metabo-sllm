# Phase 2 — RQ2 (presence recall and slot competition): annotated bibliography and evidence pack

Retrieved and verified 2026-09-08. All sources verified against primary pages (arXiv abs/HTML, CVF/NeurIPS/PMC) and locally extracted PDF text. Excluded as unverifiable: FIORA (Nature Comms behind auth redirect; bioRxiv 403) and Group DETR's CVF PDF (403; arXiv v3 camera-ready used instead). Tiers: T1 peer-reviewed, T2 preprint.

## A. Annotated bibliography (APA 7)

**DETR-family set prediction**

1. **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020).** End-to-end object detection with transformers. *ECCV 2020*. https://arxiv.org/abs/2005.12872 — T1. 100 learned queries, Hungarian matching, auxiliary Hungarian loss after every decoder layer. Presence is a softmax "no-object" class whose log-probability is **down-weighted by a factor 10**. Our balanced BCE (equal weight) is stricter on negatives than DETR's 0.1 ∅-weight.

2. **Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L. M., & Shum, H.-Y. (2023).** DINO: DETR with improved DeNoising anchor boxes. *ICLR 2023*. https://arxiv.org/abs/2203.03605 — T1. Contrastive denoising (CDN): lightly-noised positive query must reconstruct the GT, more-noised negative must predict "no object"; sigmoid focal loss. CDN +0.5 AP (47.4→47.9), +1.3 AP small objects. Adaptable: presence-negatives from perturbed true formulas (±H, ±CH₂).

3. **Chen, Q., et al. (2023).** Group DETR: Fast DETR training with group-wise one-to-many assignment. *ICCV 2023*. https://arxiv.org/abs/2207.13085 — T1. K=11 query groups, o2o inside each group, **self-attention separately per group**; inference uses one group. Conditional-DETR 32.6→37.6 (+5.0); DINO 49.4→50.1 (+0.7). Table 7: naive o2m with shared self-attention collapses to **8.4 mAP**; group-wise o2m without separate SA 34.8, with separate SA 37.6.

4. **Jia, D., et al. (2023).** DETRs with hybrid matching. *CVPR 2023*. https://arxiv.org/abs/2207.13080 — T1. Auxiliary branch of T=1500 queries matched to GT repeated K=6 times, discarded at inference. Deformable-DETR R50 47.0→48.7 (+1.7, 12 ep); K<3 degrades. Motivation: "less than 30 queries from a pool of 300" matched — our 23-of-64 regime. Lower false-negative rates (oLRP).

5. **Zong, Z., Song, G., & Liu, Y. (2023).** DETRs with collaborative hybrid assignments training. *ICCV 2023*. https://arxiv.org/abs/2211.12860 — T1. Parallel o2m auxiliary heads plus customized positive queries. Deformable-DETR 37.1→42.9 (+5.8); DINO 49.4→51.2. Encoder heads are anchor-based (vision); the decoder half transfers.

6. **Hu, Z., Sun, Y., Wang, J., & Yang, Y. (2023).** DAC-DETR: Divide the attention layers and conquer. *NeurIPS 36*. — T1. Cross-attention gathers many queries onto one object, self-attention disperses them. Auxiliary decoder sharing all weights but **without self-attention**, trained o2m; +3.4 AP over Deformable DETR (12 ep). Cheapest o2m variant (no extra parameters).

7. **Liu, S., et al. (2023).** Detection transformer with stable matching. *ICCV 2023*. https://arxiv.org/abs/2304.04742 — T1. Unstable matching traced to a "multi-optimization path" problem; supervise positives' classification score with a positional quality and add it to the matching cost. DINO 49.0 → 49.8 (PSL) → 50.2 (+PMC). Our analogue of IoU is intensity/mass-proximity agreement; our matching cost is detached.

8. **Zhang, C.-B., Zhong, Y., & Han, K. (2025).** Mr. DETR: Instructive multi-route training for detection transformers. *CVPR 2025*. https://arxiv.org/abs/2412.10028 — T1. One o2o + two o2m routes (K=6, learned instruction tokens). Deformable-DETR++ 47.0→49.5 (+2.5), DINO 49.0→50.9 (+1.9). Sharing all decoder components between routes costs −6.0; any one independent component recovers +1.6–2.1.

9. **Huang, S., et al. (2025).** DEIM: DETR with improved matching for fast convergence. *CVPR 2025*. https://arxiv.org/abs/2412.04234 — T1. Dense O2O (more targets per image via mosaic/mixup) and Matchability-Aware Loss MAL(p,q,y)= −qᵞlog p −(1−qᵞ)log(1−p) for y=1, −pᵞlog(1−p) for y=0 (γ=1.5). RT-DETRv2-R50: 53.4 → 53.6 (Dense O2O) → 53.9 (+MAL). O2O yields <10 positives/image.

10. **Lee, C., Koh, S., Jeon, Y., & Kim, J. (2026).** MDS-DETR: DETR with masked duplicate suppressor. *arXiv:2605.23507*. — T2. Early layers o2m-supervised, last layer o2o; last-layer self-attention uses a **confidence-sorted causal mask** so duplicates learn to suppress themselves. 47.7 → 49.8 (+2.1); soft mask only +0.3; +0.3 over Mr. DETR with no auxiliary decoder.

Also verified: MS-DETR (Zhao et al., CVPR 2024, https://arxiv.org/abs/2401.03989; o2m on primary queries, 48.8 AP); Align-DETR (Cai et al., BMVC 2024, https://arxiv.org/abs/2304.07527; IoU-aware BCE, +0.6); LoRA-DETR (Zhang, Y. et al., 2026, https://arxiv.org/abs/2601.09247, T2; +0.8 on Relation-DETR, abstract only).

**Presence/objectness losses and calibration**

11. **Mukhoti, J., et al. (2020).** Calibrating deep neural networks using focal loss. *NeurIPS 33*. https://arxiv.org/abs/2002.09437 — T1. Focal loss = upper bound on KL minus an entropy regulariser → less overconfident. ECE CIFAR-10 ResNet-50 4.35 (CE) → 1.48 (FL-3); CIFAR-100 17.52 → 5.13. Predicts *lower* presence probabilities — at a fixed 0.5 threshold a recall risk unless re-tuned.

12. **Li, X., et al. (2020).** Generalized focal loss. *NeurIPS 33*. https://arxiv.org/abs/2006.04388 — T1. Quality Focal Loss with continuous target y∈[0,1]. ATSS R50: 38.0 → 39.9. Adaptable: presence target = matched intensity rather than 1.

13. **Zhang, H., Wang, Y., Dayoub, F., & Sünderhauf, N. (2021).** VarifocalNet. *CVPR 2021 (oral)*. https://arxiv.org/abs/2008.13367 — T1. VFL: positives weighted by target q, negatives α·pᵞ-weighted. 39.0 → 40.1 (+1.1); removing q-weighting −0.4. DEIM reports its weakness on low-quality matches.

14. **Munir, M. A., et al. (2023).** Cal-DETR: Calibrated detection transformer. *NeurIPS 36*. https://arxiv.org/abs/2311.03570 — T1. Focal-loss-trained DETRs remain miscalibrated: D-ECE Deformable-DETR 12.8%, DINO 15.5%; variance of class logits across decoder layers as uncertainty → 8.4% / 11.7%.

**Spectrum models: matching, over-generation, losses**

15. **Goldman, S., Bradshaw, J., Xin, J., & Coley, C. W. (2023).** SCARF. *NeurIPS 36*. https://arxiv.org/abs/2303.06470 — T1. Thread (BCE, multi-label) emits top-300 formulae; Weave (cosine loss) assigns intensities. Coverage 0.907 @300, 0.968 @1000; cosine 0.726, peak coverage 0.807. 300 candidates for ~23 peaks.

16. **Murphy, M., et al. (2023).** GrAFF-MS. *ICML 2023, PMLR 202*. https://arxiv.org/abs/2301.11419 — T1. Vocabulary K=10⁴ explains 98% of ion counts; softmax over vocabulary with peak-marginal cross-entropy; cosine .70. Hungarian appears only as an evaluation metric (matchms CosineHungarian).

17. **Goldman, S., Li, J., & Coley, C. W. (2024).** ICEBERG. *Analytical Chemistry, 96*(8), 3419–3428. — T1. Generate keeps top-100 fragments; Score = sigmoid intensities, cosine loss. Cosine 0.727 ≈ SCARF 0.726, but coverage 0.754 < SCARF 0.807: "rigid fragment-grounding causes our model to miss certain lower intensity peaks" — a documented recall deficit in a committed-fragment model.

18. **Young, A., et al. (2025).** FraGNNet. *TMLR*. https://arxiv.org/abs/2404.02360 — T1. Distribution over median 679 enumerated formulae; NLL with out-of-support term. Annotation precision/recall (NIST20 scaffold): FraGNNet 0.81/0.98, ICEBERG 0.66/0.99, GrAFF-MS 0.95/0.61.

19. **Wang, R., et al. (2025).** ICEBERG 2.0. *bioRxiv*. https://doi.org/10.1101/2025.05.28.656653 — T2. Normalised spectral-entropy loss "to better emphasize low-intensity peaks"; top-1 retrieval 0.203 → 0.288 from the loss switch alone. Sinkhorn used only for soft ranking in contrastive fine-tuning. Coverage 0.809 → 0.856.

20. **Wang, R.-X., Wang, R., & Coley, C. W. (2026).** GLACIER. *arXiv:2606.29161*. — T2; COI: same group as 15, 17, 19. DETR on the molecular graph. Hungarian cost = CE(breakpoint alignment) + CE(cardinality); per-layer auxiliary losses with 0.9 depth decay; MAGMa curriculum (without it "Not Converged"). **No presence head:** every query emits a fragment, duplicates removed by set-uniqueness, a sigmoid intensity head suppresses implausible fragments. NIST20 [M+H]⁺ random: cosine 0.838, coverage 0.887 vs ICEBERG 2.0 0.794/0.856 (scaffold 0.788 vs 0.735). n_query not stated; no ablation of cost terms or query count.

Also verified: Bushuiev et al. (2024), MassSpecGym, *NeurIPS 2024 D&B* — simulation baselines' training losses not specified.

## B. Evidence table

| Mechanism | Source | Reported delta | Setting | Applicability |
|---|---|---|---|---|
| Group-wise o2m + separate self-attention | Group DETR (3) | +5.0 (C-DETR), +0.7 (DINO); naive o2m → 8.4 mAP | COCO 12 ep | Direct |
| Aux o2m branch, GT repeated K=6 | H-DETR (4) | +1.7 | Deformable-DETR | Direct |
| Aux decoder w/o self-attention, o2m | DAC-DETR (6) | +3.4 | Deformable-DETR | Direct |
| Multi-route o2o+o2m | Mr. DETR (8) | +2.5 / +1.9; shared-all −6.0 | Def-DETR++/DINO | Direct |
| o2m on primary queries | MS-DETR | 48.8 vs 47.0 | same | Direct |
| Confidence-masked SA duplicate suppressor | MDS-DETR (10) | +2.1; soft mask +0.3 | Def-DETR++ | Direct (sort by presence logit) |
| Aux heads + custom positive queries | Co-DETR (5) | +5.8 / +2.4 | 12 ep | Adaptable (decoder half) |
| Contrastive denoising | DINO (2) | +0.5 AP, +1.3 APs | 12 ep | Adaptable (formula perturbations) |
| Quality-supervised cls + cost | Stable-DINO (7) | +0.8 / +0.4 | DINO | Adaptable |
| Dense O2O + MAL | DEIM (9) | +0.2 / +0.3 | RT-DETRv2 | Adaptable |
| VFL vs focal | VFNet (13) | +1.1 | dense detector | Adaptable |
| QFL vs centerness | GFL (12) | +0.7 | ATSS | Adaptable |
| Focal loss calibration | Mukhoti (11) | ECE 4.35→1.48 | classification | Adaptable, recall direction ambiguous |
| Layer-variance uncertainty gating | Cal-DETR (14) | D-ECE 12.8→8.4 | Def-DETR | Adaptable |
| Over-generate then weight | SCARF, ICEBERG, FraGNNet | coverage 0.907@300; recall 0.66–0.95 | NIST20 | Adaptable |
| No presence gate, sigmoid intensity suppresses | GLACIER (20) | coverage 0.887 vs 0.856; cos 0.838 vs 0.794 | NIST20 [M+H]⁺ | Direct (cross-model, not ablation) |
| Entropy loss instead of cosine | ICEBERG 2.0 (19) | top-1 0.203→0.288 | NIST20 retrieval | Direct |
| Hungarian/Sinkhorn as training peak matcher | 16, 18, 19 | none — used as metric / ranking | — | Not applicable as stated |

## C. Contradictions and gaps

- **Premise correction:** no verified spectrum paper trains with Sinkhorn/Hungarian unbinned peak matching. GLACIER matches queries to MAGMa fragment patterns (identity CE + cardinality CE, no intensity/mass term); ICEBERG 2.0's Sinkhorn is a soft-ranking device; GrAFF-MS/FraGNNet use CosineHungarian for evaluation.
- **Presence head vs none:** GLACIER has no presence output and lets intensity carry suppression; it reports the best coverage. Our design gates on presence and multiplies by intensity, double-penalising low-confidence slots. Not separable from its graph encoder/MAGMa curriculum.
- **Focal-type losses:** Mukhoti shows focal loss lowers confidence — for a fixed 0.5 gate a recall risk; VFNet/GFL/DEIM use focal-type losses for ranking with IoU targets; Cal-DETR shows those detectors remain miscalibrated. None reports recall at a fixed threshold.
- **Committed vs over-generated fragments:** ICEBERG (100 committed) has higher cosine but lower coverage than SCARF (300); FraGNNet's table shows the same trade-off. No paper measures per-slot presence recall.
- **Gaps:** no result for ~25 positives among 64 slots; GLACIER's n_query unreported; o2m gains are AP with recall evidence only indirect; Group DETR's 8.4-mAP collapse warns that o2m on shared self-attention without grouping/masking may degrade duplicates.

## D. Ranked interventions

1. **Auxiliary one-to-many route with separated self-attention** (H-DETR + Group DETR + DAC-DETR; Mr. DETR ablation on what to share). Replicate the 64 queries into G groups or repeat each supervised peak K=6 times against a 64·G pool, block-diagonal self-attention, Hungarian per group, aux route dropped at inference; cheapest = DAC-style aux decoder without self-attention. Prediction: matched-presence recall ↑ (train 0.70 baseline), active-slot precision ≥ 0.47, duplicates ≤ 5.7%, valid cos@100 vs 0.51. Evidence: five T1 papers, +1.7 to +5.8 AP.
2. **Quality-aware asymmetric presence loss plus hard-negative denoising** (VFL/MAL with q = matched intensity or mass/identity agreement; DINO CDN with perturbed-formula negatives; Stable-DINO quality term in the matching cost). Confirm: presence recall at 0.5 and a reliability curve; active-slot precision and cos@100. Evidence moderate (+0.5 to +1.1 AP).
3. **Remove or soften the presence gate; move suppression into intensity; entropy-type spectral loss** (GLACIER; ICEBERG 2.0). Render from intensity alone (Huber on unmatched → 0), keep presence as auxiliary; replace cosine with normalised entropy loss. Confirm post hoc first: sweep thresholds *below* 0.5 with intensity-only rendering, then coverage, duplicate rate, cos@100. Evidence: cross-model comparison, not ablation.
