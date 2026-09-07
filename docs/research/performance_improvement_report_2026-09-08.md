# metabo-sllm 성능 향상 방향 연구 보고서

*Deep-research (full mode), 2026-09-08. Graph-free MS/MS 예측 모델(Qwen3-0.6B-Base + LoRA, 64-slot formula decoder)의 남은 오류를 무엇이 가장 크게 줄이는가.*

---

## Abstract

이 보고서는 metabo-sllm — 분자 그래프 인코더 없이 Qwen3-0.6B-Base 언어모델(LoRA r=64)로 SMILES·메타데이터 텍스트를 인코딩하고 64-slot DETR형 decoder로 fragment formula·ion state·presence·intensity를 예측하는 NIST23 MS/MS 예측 모델 — 의 성능을 높일 개입을, 1차 자료(수정된 파이프라인의 진단)와 2차 자료(검증된 문헌 52편)로 순위화한다. 2026-09-07에 두 학습 결함(DDP rank 미동기화, bf16 autocast 캐시로 인한 formula decoder gradient 소실)이 수정된 뒤 full split(train 891,038 / valid 109,817 spectra) 6 epoch 학습으로 valid canonical cos@100은 0.342 → **0.515**(supervision 상한 0.9725의 53%)에 도달했고, checkpoint 독립 평가가 in-run 값을 소수점 넷째 자리까지 재현한다. 진단 결과 잔여 오류의 지배 요인은 **일반화**다: 같은 checkpoint가 train 표본에서 0.808(상한의 83%), precursor formula가 train에 있는 valid에서 0.599, 없는 valid에서 0.453이다. identity 오류는 far miss가 29% → 8%로 붕괴한 뒤 "무거운 원자 하나가 다른 근접 formula"(active slot의 23%)가 지배하고, 후보 bag의 81%가 singleton이라 모호성 해소 objective의 상한은 작다. presence recall(train 0.70)은 임계값 조정으로는 풀리지 않는다. 문헌과 대조해 권고 순서는 (1) lr·스케줄 지평 스윕, (2) adapter 정규화 + SMILES 무작위화 + 분자당 가중, (3) neutral-loss(reverse) count head + diff embedding, (4) one-to-many 보조 라우트로 presence recall, (5) 화학 사전학습 양방향 인코더 사이드채널이다. hard-EM/RC objective, 어휘/열거-후-점수 재설계, decoder 용량 확대는 근거상 후순위로 내린다.

---

## 1. Introduction

### 1.1 맥락

metabo-sllm은 분자 구조를 그래프가 아니라 텍스트로 받는 tandem 질량 스펙트럼 예측기다. 동결된 Qwen3-0.6B-Base(LoRA r=64, rsLoRA, q/k/v/o/gate/up/down)가 `SMILES | formula | adduct | collision_energy | instrument | precursor_mz`를 인코딩하고, 64개 학습 query를 가진 transformer decoder가 slot마다 fragment의 **molecular formula**(원소별 count 자기회귀, precursor 공급량으로 제한), ion state, presence, sqrt 공간 intensity를 낸다. 학습은 Hungarian 매칭 후 candidate-bag likelihood `−log Σ_{z∈bag} p(z)`(bag = 허용오차 안의 subformula 전부, 원자가 필터), balanced presence BCE(가중 0.2), Huber intensity, 렌더된 spectrum의 cosine으로 구성된다. 설계 전제 — graph-free, MAGMa 라벨 없음, Qwen 전체 fine-tuning 없음, test fold 봉인 — 는 튜닝 대상이 아니다.

### 1.2 문제

프로젝트 초기 보고값 0.25는 후술할 결함 때문에 실제 모델 성능이 아니었고, 결함 수정 후 0.515에 이르렀으나 상한 0.9725의 절반이 남았다. "무엇이 남은 격차를 만드는가"는 수정 이전의 진단으로는 답할 수 없다 — 그 진단들은 학습이 불가능하던 decoder 위에서 이루어졌기 때문이다.

### 1.3 연구 질문

**주 RQ**: bmscaffold_1에서 상한의 53%에 있는 현재 모델의 남은 오류를 어떤 학습·모델링 개입이 가장 큰 폭으로 줄이는가?

- **RQ1 (일반화)**: train–valid 격차는 정규화, 데이터 사용, 인코더 표현 중 무엇으로 가장 잘 닫히는가?
- **RQ2 (presence)**: 진짜 fragment의 30%가 presence < 0.5인 recall 결손은 어떤 목적·구조로 정밀도 손실 없이 회복되는가?
- **RQ3 (identity 잔차)**: 정상 학습된 decoder의 miss는 어떤 유형이며 그에 맞는 개입은 무엇인가?

Phase 1 반론 점검에서 세 가지가 수정됐다: "과적합"은 측정 전엔 가설이다(lr 감쇠 구간과 겹침), Phase 2 문헌은 수정 후 진단에 대고 매핑한다, 세 RQ 중 잔차가 가장 큰 축을 우선한다.

---

## 2. Background: 시스템의 측정된 상태

### 2.1 두 결함과 정정된 수치

2026-09-07, "checkpoint가 자기 in-run 검증값을 재현하지 못한다"는 어긋남에서 출발해 두 결함을 찾았다(커밋 `876aa41`).

1. **rank 미동기화.** 학습 forward가 DDP 래퍼가 아닌 raw 모듈로 호출되어 gradient all-reduce가 무장되지 않았다. 4개 rank가 각자 데이터의 1/4로 독립 학습했고, in-run 검증은 네 모델의 혼합, checkpoint는 rank 0이었다. in-run 예측 parquet을 rank 순서로 나누면 rank 0 구간만 checkpoint와 100% 일치했다. V1 subset run의 네 rank는 0.350 / 0.098 / 0.345 / 0.210으로, 보고된 0.2505는 그 평균이었고 저장된 모델은 **0.3481**이었다.
2. **formula decoder의 Linear 가중치 gradient 소실.** two-pass scorer가 같은 bf16 autocast 구간에서 decoder를 `no_grad`(매칭 비용)로 먼저 돌리고 grad로 다시 돌렸는데, autocast가 첫 캐스트에서 만든 bf16 복사본을 캐시하고 그 복사본에는 autograd 이력이 없어 두 번째 패스의 gradient가 fp32 원본에 닿지 않았다. fp32로 도는 LayerNorm·embedding만 학습됐다. 격리 재현과 회귀 테스트(`tests/test_autocast_two_pass.py`)로 고정했고, `cache_enabled=False`로 수정했다.

이후 모든 수치는 (a) checkpoint의 독립 평가로 재현되고(smoke: 0.1015 = 0.1015; best: 0.5151 = 0.5151), (b) 저장 시 rank 간 파라미터 일치가 단언된다.

### 2.2 수정 후 run

`qwen_formula_slots_v1_full_fixed_seed0`: V1 rank-0 가중치에서 warm-start, lr 1.5e-4 cosine(진짜 유효 batch 256), presence 0.2, full split, 8 epoch 계획 중 사용자 요청으로 epoch 6 후 중단.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| valid canonical cos@100 | 0.443 | 0.464 | 0.483 | 0.502 | 0.508 | 0.510 | **0.515** |
| valid bag_hit | 0.328 | 0.343 | 0.370 | 0.385 | 0.382 | 0.365 | 0.388 |
| train bag NLL (epoch 평균) | 1.785 | 1.358 | 1.147 | 0.977 | 0.829 | — | — |
| train bag_hit | 0.518 | 0.585 | 0.628 | 0.666 | 0.702 | — | — |

step-0(학습 전, V1 rank-0 그대로)은 0.3418이었다. full valid fold의 상한은 subset과 동일하다: peak_copy 0.9914, support_mask 0.9765, **formula_rendered 0.9725**, slot_capacity 0.9684. 두 valid fold의 분자량 분포도 같다(p50 357 vs 360 Da).

---

## 3. Method

**패러다임**: pragmatist — 실험적 근거로 순위를 정한다.

**데이터 전략.** 1차 자료 = 수정된 파이프라인의 best checkpoint(epoch 6)에 대한 진단: (i) checkpoint 독립 재현, (ii) train 표본 4,000건 자유 예측, (iii) precursor formula의 train 포함 여부·분자량 층화, (iv) identity 오류 유형(근접/far/수소/원자가/어휘), (v) 후보 bag 크기 분포, (vi) presence 임계값 sweep, (vii) counterfactual decomposition(8 condition × 4 intensity 공간)과 bag mass 통계. 2차 자료 = Phase 2에서 세 RQ 축별로 검색·검증한 문헌(각 팩은 `docs/research/phase2_rq{1,2,3}_*.md`; 모든 출처를 1차 페이지에서 확인하고 미확인 후보는 제외).

**분석 틀.** 잔차 분해 → 각 잔차에 문헌 기제 매핑 → (기대이득 × 근거 강도) / 비용 순위 → 성공 기준·비용·순서를 가진 실험 계획.

**타당성.** 모든 모델 수치는 checkpoint 기준 독립 평가로 재현 가능해야 한다(in-run 값 단독 사용 금지). 문헌 수치는 metric 관례(bin 폭, sqrt, top-k, CE 병합 여부)와 함께 인용한다. 이해충돌: RQ2·RQ3 팩의 spectrum 논문 다수가 한 연구실(Coley)의 자기 재구현 비교이며, PLL 논문들은 같은 연구진의 결과다.

---

## 4. Findings

### 4.1 1차 진단 — 수정 후 best checkpoint

**(a) 일반화가 지배적 잔차다.** 같은 checkpoint의 자유 예측:

| | train 표본 (4,000) | valid (109,817) |
|---|---|---|
| canonical cos@100 | **0.808** | **0.515** |
| bag_hit | 0.613 | 0.388 |
| unique-bag hit | 0.671 | 0.457 |
| active slot 정밀도 | 0.635 | 0.466 |

train에서는 이미 상한의 83%다. valid 안에서도 **precursor formula가 train에 있는 42.6%는 0.599, 없는 57.4%는 0.453**이다 — formula 신규성 하나로 0.146의 격차가 생긴다. 분자량별로는 0.638(<200 Da) → 0.578 → 0.501 → 0.460(400–600) → 0.463(≥600)로 단조 감소 후 평탄. 분포는 넓다(중앙값 0.549, p10 0.068, p90 0.889).

**(b) identity 잔차는 근접 formula 판별이다.** valid active slot 2.75M 기준:

| 유형 | 수정 전 (subset, V1) | **수정 후 (epoch 6)** |
|---|---|---|
| hit | 28.3% | **54.0%** |
| 수소만 차이 | 5.9% | 5.0% |
| 무거운 원자 1개 차이 | 21.9% | **22.9%** |
| 무거운 원자 2개 | 14.9% | 9.9% |
| far | 29.1% | **8.3%** |
| miss의 m/z 간격 중앙값 | 3.97 Da | 2.0 Da |

원자가 하드 위반은 hit·miss 모두 0%(v2 bag), 소프트 RDBE(precursor+1 초과)는 hit 6.5% vs miss 6.4%로 무신호, miss formula의 97.5%가 train에 20회 이상 등장한다. 출력 공간·제약이 아니라 **CH₂↔O↔N 치환·±C 수준의 판별력**이 문제다.

**(c) bag은 압도적으로 singleton이다.** supervised peak 중 valid 80.7%(세기 83.5%)가 |Z|=1, 14.2%가 2–3, 5.1%가 ≥4; train은 88.9%가 singleton. 모호성 해소를 겨냥한 objective가 건드릴 수 있는 peak은 5% 안팎이다. unique-bag hit 0.457 vs ambiguous 0.101이지만, ambiguous가 전부 unique 수준으로 올라도 bag_hit은 ~0.07 오른다.

**(d) presence는 임계값 문제가 아니다.** epoch 4 예측의 사후 sweep: 0.5 → 0.5079, 0.6 → 0.5012, 0.7 → 0.4895, 0.9 → 0.4229 — 올리면 단조 손해. presence가 낮은 슬롯은 gating(presence×amplitude)으로 세기가 작아 잘라낸 노이즈보다 잃는 진짜 peak이 많다. 남는 문제는 **recall**(train matched recall 0.70)이며 학습 쪽 변경이 필요하다. 저장된 예측이 active 슬롯만 담아 0.5 아래는 사후 검증이 불가능하다(다음 run에서 저장 범위를 넓혀야 한다).

**(e) 손실별 상태(epoch 0→4).** identity(bag NLL) 1.79 → 0.83으로 주 동력이자 여전히 하강 중; presence 0.58 → 0.48로 가장 느림(무작위 0.69); intensity 0.017 → 0.009로 미미; spectrum 0.224 → 0.140(매칭이 주어진 렌더링 cosine 0.86). 매칭이 주어진 0.86과 자유 예측 valid 0.515 사이가 presence 오류 + identity 오류 + 일반화의 합이다.

**(f) counterfactual decomposition.**

valid 109,817건, canonical cos@100. 각 condition은 예측의 한 부분만 supervision의 정답으로 바꾼다.

| condition | cos@100 | 기준선 대비 |
|---|---|---|
| all_predicted (실제 성능) | **0.5151** | — |
| ion state 정답 | 0.5152 | 0.000 |
| presence만 matched 슬롯에 강제 | 0.5398 | +0.025 |
| identity(위치) 정답, weight(presence×intensity) 모델 | 0.5784 | **+0.063** |
| identity 모델, weight 정답 | 0.6229 | **+0.108** |
| identity·weight 모두 정답 (sqrt 세기) | 0.9688 | 상한 |

수정 전 pilot에서는 identity 정답의 이득(0.218 → 0.426)이 weight 정답(→ 0.302)보다 컸으나, 정상 학습된 모델에서는 **weight 정답의 이득이 더 크다**. 그리고 한쪽만 고치면 0.58/0.62에 머무르고 둘 다 고쳐야 0.97에 이른다 — 남은 오류는 같은 슬롯에서 identity와 세기가 함께 틀리는 형태로 얽혀 있어, 성분별 이득의 합(0.17)이 전체 격차(0.45)에 크게 못 미친다. 이 결과는 §4.1(e)의 "intensity 항은 작다"는 loss 값 기준 판단을 정정한다: Huber 값은 작지만 **렌더된 세기 오류의 metric 비용은 크며**, 현재 spectrum loss가 매칭된 슬롯만 렌더하므로 false positive 슬롯과 자유 예측의 세기 보정은 직접 최적화되지 않는다.

**(g) bag mass 층화.**

Hungarian으로 매칭된 (slot, peak) 쌍 기준 `−log Σ_bag p`의 기하평균 mass와 greedy argmax가 bag 안에 드는 비율.

| 그룹 | valid: 쌍 수 | geo. mass | argmax-in-bag | train 표본: argmax-in-bag |
|---|---|---|---|---|
| 전체 | 3,303,224 | 0.053 | 0.453 | 0.761 |
| unique bag (\|Z\|=1) | 2,666,143 | 0.091 | 0.524 | 0.799 |
| ambiguous (\|Z\|≥2) | 637,081 | 0.0055 | 0.156 | 0.579 |
| precursor <200 Da | 143,068 | 0.363 | 0.763 | 0.957 |
| 200–300 | 734,347 | 0.144 | 0.584 | 0.826 |
| 300–400 | 1,046,176 | 0.055 | 0.435 | 0.705 |
| 400–600 | 1,046,641 | 0.026 | 0.361 | 0.685 |
| ≥600 | 332,992 | 0.020 | 0.375 | 0.616 |

수정 전 pilot의 unique-bag mass 0.016 / argmax 0.249에서 0.091 / 0.524로 올랐다. train–valid 격차는 모든 층에서 크고(전체 0.761 vs 0.453), 분자량 의존은 두 fold 모두에 있다. ambiguous bag(19%의 쌍)의 0.156은 낮지만, §4.1(c)대로 이 층이 전부 unique 수준이 되어도 전체 이득은 ~0.07이다.

### 4.2 2차 자료 — 검증된 문헌의 요지

**RQ1 (일반화).** LoRA 변종·정규화보다 **lr 스윕이 먼저**다: lr을 맞추면 rsLoRA/DoRA/LoRA+ 차이가 1–2% 안이고(Lee et al., 2026; He et al., 2026), LoRA는 lr 민감도가 크며 최적 lr이 full FT의 ~10배(Biderman et al., 2024). rank는 단조가 아닐 수 있다(DoRA Table 15: LoRA r=32 74.7 → r=64 65.8; Liu et al., 2024). adapter weight decay를 일반화 지렛대로 분리 측정한 출처는 없고, MS 모델들은 dropout 0–0.3·wd 0–1e-5를 탐색으로 골랐다(Goldman et al., 2023; Murphy et al., 2023; Young et al., 2025). cosine의 종점은 계획 길이에 묶이며(Hägele et al., 2024) 선형 0-감쇠가 낫다(Bergsma et al., 2025). scaffold split에서 그래프 SOTA도 0.03–0.08 떨어진다(ICEBERG 0.727 → 0.699; FraGNNet 0.736 → 0.678). CE 인코딩 제거 시 −0.02(ICEBERG 2.0). 화학 사전학습 양방향 인코더가 범용 LLM 특징보다 낫다는 Tier-1 근거(Kristiadi et al., 2024; Ross et al., 2022, MoLFormer scaffold ROC-AUC). SMILES 무작위화는 강한 증강이다(SimSon, 2025; Brinkmann et al., 2025).

**RQ2 (presence).** DETR 원조도 ∅ 클래스를 0.1 가중으로 뒀다(Carion et al., 2020) — 우리 balanced BCE는 음성에 더 엄격하다. one-to-many 보조 라우트가 Tier-1 5편에서 +1.7~+5.8 AP(H-DETR, DAC-DETR, Mr. DETR, Co-DETR, Group DETR), 단 self-attention을 분리하지 않으면 8.4 mAP로 붕괴(Chen et al., 2023). confidence-mask self-attention +2.1(MDS-DETR, 2026). GLACIER는 presence head가 없고 sigmoid intensity가 억제를 맡으며 coverage 최고(0.887; Wang et al., 2026). committed-fragment 모델은 coverage가 낮다(ICEBERG 0.754 vs SCARF 0.807; Goldman et al., 2024). focal 계열은 신뢰도를 낮춰 고정 임계값에서 recall 위험(Mukhoti et al., 2020). 검증된 spectrum 논문 중 Sinkhorn/Hungarian을 *학습* 매칭에 쓴 것은 GLACIER뿐이다.

**RQ3 (identity).** SCARF ablation(Goldman et al., 2023, Table 1): forward-only → **reverse(neutral-loss) head 게이트 결합**으로 coverage@30 0.476 → 0.552, @300 0.855 → 0.907; 이유는 질량이 커질수록 fragment 공간은 폭발하나 loss 공간은 작기 때문(Murphy et al., 2023 §4.2). 우리 loss는 partial-label learning의 CC이며 RC/PRODEN·annealed hard-EM이 |Z|>3에서 크게 낫다(Feng et al., 2020; Wang et al., 2025; Min et al., 2019) — 단 우리 bag은 81% singleton. vocab vs 생성은 데이터셋마다 뒤집힌다(NIST20 SCARF 0.726 > FixedVocab 0.704; NPLIB1 0.536 < 0.568). 약한 peak에 대한 entropy loss 효과는 수치 없음. 모든 방법이 분자량 증가에 따라 정확도가 떨어진다(Murphy; Goldman; Goldman et al., 2024 MIST-CF; Ludwig et al., 2020).

### 4.3 잔차 → 기제 매핑

| 측정된 잔차 | 크기 | 맞는 기제 (근거) | 맞지 않는 기제 (근거) |
|---|---|---|---|
| train 0.81 vs valid 0.52; unseen formula 0.45 vs seen 0.60 | 최대 | lr/스케줄 스윕(T2 2026×2, T1); adapter dropout/wd/rank 대조(T1/T2); SMILES 무작위화(T1); 분자당 가중; 화학 인코더(T1) | decoder 용량 확대(train이 이미 0.81) |
| 근접 formula miss 23%(1 heavy atom), 분자량↑ 악화 | 중 | reverse count head + diff embedding(T1 ablation) | 원자가 마스크(위반 0%), 어휘 제한(miss 97.5% in-vocab), hard-EM/RC(singleton 81%) |
| weight(presence×intensity) 오류: oracle weight +0.108, presence 강제 +0.025 | 대 | 자유 예측 렌더 spectrum 전체에 대한 cosine/entropy 목적(SCARF-Weave; ICEBERG 2.0; GLACIER), one-to-many 보조 라우트(T1×5), ∅ 가중 완화 | 임계값 조정(sweep 단조 손해), Huber 가중 상향(값이 이미 작음) |
| ambiguous-bag hit 0.10 | 소(5% peak) | RC/hard-EM — 상한 ~0.07 | — |

---

## 5. Discussion

### 5.1 해석

수정 전 프로젝트는 "identity가 병목이고 구조를 바꿔야 한다"는 결론에 이르렀으나, 그것은 학습이 불가능한 decoder의 증상이었다. 수정 후 identity는 6 epoch 만에 train에서 상한의 83%까지 배웠고, valid에서 남는 것은 **보지 못한 formula로의 일반화**다. 이는 (a) 인코더가 SMILES에서 구조를 얼마나 옮기는가, (b) adapter·decoder가 train formula를 얼마나 암기하는가의 문제이며, 문헌의 첫 지침은 아직 한 번도 하지 않은 lr·스케줄 스윕이다. identity 잔차의 형태(근접 치환, 질량 의존)는 SCARF의 reverse head ablation과 정확히 맞물린다. presence는 학습 목적의 문제다.

### 5.2 반론 점검 (Checkpoint 2·3)

- *"일반화가 아니라 아직 underfit이다."* train 0.81은 underfit 가설을 기각한다. 다만 valid 정체(ep 3–6)가 lr 급감 구간과 겹치므로, 정규화 실험은 반드시 스케줄 대조군(R1)과 함께 읽어야 한다 — 채택.
- *"seen/unseen 격차는 valid fold 구성 편향이다."* 두 fold의 분자량 분포가 같고 full valid의 formula 중복률(42.6%)이 subset(11.4%)보다 높다. 편향이라면 반대 방향이어야 한다 — 기각.
- *"presence를 제거한 GLACIER식이 답이다."* 근거는 ablation이 아닌 모델 간 비교이고 그래프 인코더·MAGMa와 분리되지 않는다 — 3순위 이하로.
- *"열거-후-점수 재설계가 필요하다."* 수정 전 진단에서 나온 결론이며, 수정 후 miss는 in-vocab·근접 formula라 출력 공간 문제가 아니고, vocab vs 생성은 데이터셋마다 뒤집힌다 — 보류.
- *"체리피킹."* RQ1 문헌은 정규화 효과를 분리한 출처가 없음을 명시했고, RQ3는 PLL 이득이 우리 bag 분포에서 상한이 작음을 스스로 제한했다.
- *"so what."* 상한의 53%에서 남은 47% 중 train 기준 30%p는 일반화 격차로 귀속되므로, 이 축을 닫는 것이 다른 어떤 축보다 크다.

### 5.3 편집·윤리 점검

근거 없는 주장은 없도록 모든 수치를 진단 JSON 또는 검증된 출처에 연결했다. 이해충돌(Coley 연구실 자기비교, PLL 동일 연구진)을 명시했다. AI 보조 연구 도구를 사용했다(§8). 인간 대상 연구가 아니며 이중용도 우려는 없다.

### 5.4 한계

- 진단은 단일 seed·단일 run이다. seen/unseen 격차의 신뢰구간은 미산정이나 n=46,740 / 63,077이라 표본오차는 작다.
- 문헌 이득은 다른 도메인(COCO, QA, NIST20 그래프 모델)의 값이며 우리 설정으로의 전이는 가설이다.
- 수정 후 run은 6 epoch에서 중단됐고 lr이 감쇠 중이었다 — 정체와 스케줄이 교락한다.
- presence 0.5 아래 sweep은 저장 범위 때문에 불가능했다.
- test fold는 봉인되어 있어 모든 수치는 valid 기준이다.

---

## 6. Recommendations — 실험 계획

성능 우선 원칙에 따라 여러 하이퍼파라미터를 한 run에서 함께 바꾸되, 각 run의 성공 기준은 valid의 **checkpoint 독립 평가**로 판정한다. 기준선: canonical cos@100 0.515, unseen-formula 0.453, bag_hit 0.388, matched recall 0.70, 정밀도 0.466.

| 순위 | 개입 | 근거 | 성공 기준 | 비용 |
|---|---|---|---|---|
| **R1** | **lr·스케줄 스윕**: lr {0.75, 1.5, 3}×1e-4; cosine 2배 지평 / constant+cooldown / 선형 0-감쇠. best에서 warm-start, 각 3 epoch | Lee 2026; He 2026; Biderman 2024; Hägele 2024; Bergsma 2025 | 어느 arm이든 valid 0.515↑ 또는 unseen 0.453↑; train bag_hit 불변인데 valid bag_hit >0.02 오르면 "스케줄 제한" 확정 | arm당 ~2.6 h (4 GPU) |
| **R2** | **일반화 묶음**: LoRA dropout 0.1–0.3, weight decay 1e-5(adapter+decoder), 무작위 SMILES 증강, 분자당 1/(#CE) 가중; rank {32, 128} 대조 | Lin 2024; Liu 2024; Nowatzky 2025; SimSon 2025 | train−valid bag_hit 격차(0.61−0.39) 축소, **unseen 층 상승**, cos@100 ≥ 0.515 | full 6 epoch ~5 h |
| **R3** | **reverse(neutral-loss) count head + diff embedding**: 원소마다 forward count와 precursor−count 인덱스의 loss count를 학습된 gate로 결합; prefix까지의 차분 count를 입력 | Goldman 2023 Table 1; Murphy 2023 §4.2; ms-pred `scarf_model.py:466-476` | 분자량 상위 사분위 bag_hit↑, "1 heavy atom" miss 22.9%↓, cos@100↑ | 구현 1일 + 5 h |
| **R4** | **weight 목적 + presence recall**: (i) spectrum loss를 매칭 슬롯만이 아니라 **모든 active 슬롯의 자유 예측 렌더링**에 걸고(binned cosine, entropy 변형 A/B) false positive와 세기 보정을 직접 최적화; (ii) one-to-many 보조 라우트(DAC식 self-attention 없는 보조 decoder 또는 Group식 분리 SA); (iii) ∅ 가중 완화(balanced → DETR식 비대칭); (iv) 저장 범위를 모든 슬롯으로 확대 | counterfactual §4.1(f); Goldman 2023 (Weave); Wang 2025 (entropy); Wang 2026; Hu 2023; Jia 2023; Chen 2023; Carion 2020 | `predicted_identity_oracle_weights` 격차(0.108) 축소, matched recall 0.70↑, 정밀도 ≥ 0.466, 중복 ≤ 5.7%, cos@100↑; 0.5 아래 임계값 sweep 가능 | 구현 1–2일 + 5 h |
| **R5** | **화학 사전학습 양방향 인코더 사이드채널**: MoLFormer-XL(47M, 양방향, SMILES 1.1B) memory를 Qwen memory와 함께 슬롯 decoder에 제공 | Ross 2022; Kristiadi 2024; BehnamGhader 2024 | **unseen 층**에서 집중된 상승 | 구현 1–2일 + 5 h |

**후순위(근거 부족 또는 상한 작음)**: hard-EM/RC/PRODEN objective(singleton 81%); 어휘 제한·열거-후-점수 재설계(miss in-vocab, 데이터셋 의존); presence 임계값; decoder 용량 확대; 4B 백본(일반화 문제에 크기는 답이 아님, GIF 0.35).

**순서와 병합.** R1은 다른 모든 실험의 전제이므로 먼저(~8 h, 3 arm 순차 또는 2 GPU×2 병렬). R2와 R3는 서로 다른 잔차를 겨냥하므로 한 run에 묶어도 해석이 흐려지지 않는다(성능 우선). R4(i)의 자유 예측 렌더 목적은 counterfactual상 단일 성분 이득이 가장 큰 축(+0.108)을 겨냥하므로 R2+R3 run에 함께 넣을 후보이며, R4(ii)–(iv)는 그 best에서 warm-start. R5는 인코더 교체라 마지막에 독립 A/B. 각 run 후 §4.1의 진단(층화·오류 유형·counterfactual)을 동일 스크립트로 재실행해 잔차 표를 갱신한다.

---

## 7. Conclusion

수정된 파이프라인에서 모델은 train 데이터를 상한의 83%까지 설명하고 valid에서 53%에 머문다. 남은 격차의 이름은 **일반화**이며, 그 아래에 근접 formula 판별(질량 의존)과 presence recall이 있다. 문헌은 이 세 잔차에 각각 lr·스케줄과 adapter 정규화·증강(RQ1), neutral-loss head(RQ3), one-to-many 감독(RQ2)을 대응시키며, 모호성 objective와 출력 공간 재설계는 우리 데이터의 bag 구조와 오류 형태상 후순위다. 다음 GPU 시간은 R1 → R2+R3 → R4 → R5 순으로 쓰는 것이 기대값이 가장 높다.

---

## 8. AI 사용 고지

이 보고서는 Claude Code(Anthropic)가 사용자의 지시 아래 작성했다. 1차 진단은 저장소의 스크립트로 계산했고, 2차 자료는 3개 subagent가 웹·arXiv·저널 페이지에서 검색·검증했으며 미확인 출처는 제외했다. 사용자는 연구 질문·모드·중단 시점을 결정했다.

---

## References (APA 7, 본문 인용분)

- BehnamGhader, P., Adlakha, V., Mosbach, M., Bahdanau, D., Chapados, N., & Reddy, S. (2024). LLM2Vec: Large language models are secretly powerful text encoders. *COLM 2024*. https://arxiv.org/abs/2404.05961
- Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D., & Hestness, J. (2025). Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs. *ICLR 2025*. https://arxiv.org/abs/2502.15938
- Biderman, D., et al. (2024). LoRA learns less and forgets less. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2405.09673
- Brinkmann, H., et al. (2025). *Digital Discovery, 4*(10), 2752.
- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. *ECCV 2020*. https://arxiv.org/abs/2005.12872
- Chen, Q., et al. (2023). Group DETR: Fast DETR training with group-wise one-to-many assignment. *ICCV 2023*. https://arxiv.org/abs/2207.13085
- Feng, L., Lv, J., Han, B., Xu, M., Niu, G., Geng, X., An, B., & Sugiyama, M. (2020). Provably consistent partial-label learning. *NeurIPS 33*.
- Goldman, S., Bradshaw, J., Xin, J., & Coley, C. W. (2023). Prefix-tree decoding for predicting mass spectra from molecules. *NeurIPS 36*. https://arxiv.org/abs/2303.06470
- Goldman, S., Li, J., & Coley, C. W. (2024). Generating molecular fragmentation graphs with autoregressive neural networks. *Analytical Chemistry, 96*(8), 3419–3428.
- Goldman, S., Xin, J., Provenzano, J., & Coley, C. W. (2024). MIST-CF: Chemical formula inference from tandem mass spectra. *JCIM, 64*(7), 2421–2431.
- Hägele, A., et al. (2024). Scaling laws and compute-optimal training beyond fixed training durations. *NeurIPS 2024*. https://arxiv.org/abs/2405.18392
- He, et al. (2026). A unified study of LoRA variants. arXiv:2601.22708.
- Hu, Z., Sun, Y., Wang, J., & Yang, Y. (2023). DAC-DETR: Divide the attention layers and conquer. *NeurIPS 36*.
- Jia, D., et al. (2023). DETRs with hybrid matching. *CVPR 2023*. https://arxiv.org/abs/2207.13080
- Kristiadi, A., Strieth-Kalthoff, F., Skreta, M., Poupart, P., Aspuru-Guzik, A., & Pleiss, G. (2024). A sober look at LLMs for material discovery. *ICML 2024, PMLR 235*. https://arxiv.org/abs/2402.05015
- Lee, C. E., et al. (2025). SimSon. *Bioinformatics, 41*(5).
- Lee, Y.-A., Ko, C.-Y., Chen, P.-Y., & Yeh, M.-Y. (2026). Learning rate matters: Vanilla LoRA may suffice for LLM fine-tuning. arXiv:2602.04998.
- Lin, Y., et al. (2024). LoRA Dropout as a sparsity regularizer for overfitting control. arXiv:2404.09610.
- Liu, S.-Y., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML 2024, PMLR 235*.
- Ludwig, M., et al. (2020). Database-independent molecular formula annotation using Gibbs sampling through ZODIAC. *Nature Machine Intelligence, 2*, 629–641.
- Martin, M. R., & Hassoun, S. (2025). General Intelligence-based Fragmentation. arXiv:2511.09571.
- Min, S., Chen, D., Hajishirzi, H., & Zettlemoyer, L. (2019). A discrete hard EM approach for weakly supervised question answering. *EMNLP-IJCNLP 2019*, 2851–2864.
- Mukhoti, J., et al. (2020). Calibrating deep neural networks using focal loss. *NeurIPS 33*.
- Murphy, M., Jegelka, S., Fraenkel, E., Kind, T., Healey, D., & Butler, T. (2023). Efficiently predicting high resolution mass spectra with graph neural networks. *ICML 2023, PMLR 202*.
- Nowatzky, Y., et al. (2025). FIORA. *Nature Communications*.
- Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., & Das, P. (2022). Large-scale chemical language representations capture molecular structure and properties. *Nature Machine Intelligence*.
- Wang, R., Manjrekar, M., et al. (2025). Neural spectral prediction for structure elucidation with tandem mass spectrometry (ICEBERG 2.0). *bioRxiv*.
- Wang, R.-X., Wang, R., & Coley, C. W. (2026). GLACIER: Rethinking mass spectrum prediction as an object detection problem. arXiv:2606.29161.
- Wang, W., et al. (2025). Realistic evaluation of deep partial-label learning algorithms. *ICLR 2025*.
- Young, A., Wang, F., Wishart, D. S., Wang, B., Greiner, R., & Röst, H. (2025). FraGNNet. *TMLR*.
- Zhang, C.-B., Zhong, Y., & Han, K. (2025). Mr. DETR. *CVPR 2025*.

전체 서지(52편, 등급·URL·수치)는 `docs/research/phase2_rq1_generalization_bibliography.md`, `..._rq2_presence_bibliography.md`, `..._rq3_identity_bibliography.md`에 있다.

## Appendix — 진단 산출물

- `bmscaffold_1/qwen_formula_slots_v1_full_fixed_seed0/diagnostics/evaluation_contract_v1/audit.json` — counterfactual·bags
- `$CLAUDE_JOB_DIR/tmp/identity_errors_fixed_ep6.json` — identity 오류 유형·원자가·어휘
- `predictions/valid_epoch00..06.parquet` — epoch별 예측(presence 포함)
- `HANDOFF.md` §0–§1 — 결함 기록과 정정 수치
