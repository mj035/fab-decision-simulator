# 사전등록 — LightGBM OOF SHAP 안정성·센서군 분석 v2

> 🟣 **Revision 2 (2026-07-21, CODEX 재감사 차단 항목 보완 — 분석 실행 0건 상태에서 개정)**
> - Revision 1 SHA-256: `b550f9958dcba41fcf0bee21e2f00bab748813487a358d97105339196cac5512`
> - **결과 확인 전 변경** — probe·fit·SHAP·CV·집계 실행 이력 0건. CODEX FAIL(저장·검증 연결부) 보완 목적.
> - 변경 규칙: seed metadata 의미 검증기(승격/cache/run 공통) · NPZ dtype 8종 고정 검증 ·
>   클래스별 6종 벡터 전부의 rank/Top-k 저장(signed 이중 정렬 분리) · commit marker 승격 구조 ·
>   probe summary/run gate 보강(logit QC·metadata 검증·marker 정합) · manifest 계보 검증(시작/종료 시각 분리·stale 차단) ·
>   비차단 권고(비반올림 사후검증·결정적 정렬·용량 추정 한계). 상세 = **부록 R2**(R1과 충돌 시 R2 우선).
>
> 🔵 **Revision 1 (2026-07-21, CODEX 실행 전 감사 반영 — 분석 실행 0건 상태에서 개정)**
> - Revision 0 SHA-256: `d47170b6f3de4335993adc049ba501fd3ea0e851bd909a51cf1cdccd758463b8`
> - **결과를 보고 변경한 것이 아니다** — probe·run·aggregate·groups·fit·SHAP·상관 계산 실행 이력 0건에서의 명세 정밀화다.
> - 개정 사유: probe 승계 검증 강화 · 전체 군집 저장(상위 30 계산 제한 제거) · fold Jaccard 구현 명세 ·
>   클래스별 집계 실저장 · seed별 군집 A/B/C 분포 · 실제 모델 get_params 저장 · 환경 fingerprint ·
>   similarity clip·수치 안정성. 상세 = **부록 R1**(기존 본문은 보존, R1이 우선한다).
>
> **실행 전 확정 문서.** 실행 후 수정 금지 — 수정 필요 시 중단하고 개정 승인을 받는다(개정 이력 필수).
> 작성 시점: 2026-07-21. **본 문서 작성 시점에 LightGBM fit·SHAP·CV·프로브·상관 계산은 일절 실행되지 않았다.**
> 격리: 신규 경로만 사용 — `src/oof_shap_v2.py`, `src/render_oof_shap_v2.py`(골격), `results/oof_shap_v2/`(실행 시 생성).

## 1. 목적과 비목적

**목적**: 운영 모델 LightGBM(champion, 474 피처)의 예측 설명이 **미관측 validation 데이터(OOF)**에서
얼마나 반복적으로 유지되는지 평가한다 — 센서별 OOF SHAP 순위 분포 · Top-k 진입 빈도·Jaccard 분포 ·
고상관 센서군 단위 기여도 · 임계값/군집방식 민감도 · S59의 OOF 설명 안정성과 해석 한계 병기 ·
우선 점검·DOE 후보군 제시.

**비목적(금지)**: 성능 개선 · 모델 교체 재평가 · 인과/공정 신호 검증 · PM 확정 · 피처 제거/차원축소 ·
새 튜닝 · early stopping. **결과가 기존 주장(기존 그림·S59 서사)에 불리해도 그대로 보고한다.**

## 2. 데이터·피처 계보

- 입력: `../디스플레이 AI챌린지 데이터셋/fab_process_yield.csv`
  (SHA-256 `e8b8187ce276db148707b5103463aa0f35cb8e521da2bc697c0fd45a6b06ad0f` — 실행 시 재계산·불일치 시 중단)
- `preprocess_adapter.adapt(df.drop(columns=["Time"]), "Pass/Fail", 1)` — N=1,567, 양성 104.
- **마스터 피처 474개**: 전체 데이터 기준 중앙값 대치 후 분산 0 제거로 도출한 **피처명 목록**(기존 정본 계보와 동일,
  S59 컬럼명 '59' 실측 확인 완료). **정렬·저장 기준으로만 사용하며 모델 fit에는 사용하지 않는다.**
  실행 시 474 불일치면 즉시 중단.

## 3. CV·seed (고정)

- seeds = **0~19** (기존 프로젝트 고정 seed 재사용). `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.
- 총 **100 fold-모델**. 각 seed에서 전 1,567행이 validation에 **정확히 1회** 포함 — 중복·누락 assert 필수.
- **주 분석 단위 = seed별 완전 OOF SHAP 행렬(1567×474)**. fold별 결과는 소표본(양성 ~21) 잡음 때문에 **참고 분포로만**.
- **모델 random_state = seed 확정**(base_lgbm 정본 관례). fold별 임의 변경 금지.

## 4. fold 내부 전처리 (누수 차단)

각 fold의 **train에서만** fit: ① `SimpleImputer(strategy="median")` ② `VarianceThreshold(0.0)` ③ `StandardScaler`
(정본 경로 일치 목적 — 트리 모델에 스케일은 불변이나 경로 통일). validation에는 transform만.
**전체 데이터 사전 fit 금지.**

피처 정렬: fold train에서 상수가 된 피처는 그 fold 모델에서 제거하되, 474열 SHAP 행렬에서 해당 열에 **0**을 기입하고
`feature_active[seed, fold, 474]` 마스크로 구분 저장. fold별 active 수·제거 피처명·피처 순서·train/val 인덱스(해시) 저장.

## 5. 모델 파라미터 (고정 — 출처 `src/model_cmp.py:101-103` 정본)

```
n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
scale_pos_weight = (fold train의 N_neg/N_pos),  verbose=-1,  random_state=seed
```
새 튜닝·early stopping·validation 기반 변경·결과 후 변경 금지.
`get_params()` 전체와 lightgbm 버전을 metadata에 저장.

## 6. SHAP — validation 전용

각 fold: train으로만 학습 → **validation 행에서만** `shap.TreeExplainer` 계산 → train 행 SHAP 저장 금지 →
원래 행 인덱스 위치에 배치 → seed 종료 시 1567×474 완성.

이진 출력 분기(고정): 2D ndarray → 그대로(positive-class) / list → class 1 / 3D → 마지막 축 class 1 /
그 외 shape → **즉시 중단**. expected_value도 class 1 기준 정규화. **SHAP은 raw margin 기준**임을 명시
(확률과 혼용 금지).

## 7. Additivity QC (필수, 기준 고정)

각 fold의 validation 전체에서 `expected_value + Σ SHAP ≈ raw margin`(`m.predict(raw_score=True)`)을 외부 검증:
- 전 값 finite 확인
- **`np.allclose(atol=1e-4, rtol=1e-4)`** (사전 고정 — 변경 금지)
- fold별 최대 절대 오차 저장
- 실패 시 해당 seed 계속 진행 금지 — **전체 분석 중단** 후 보고.

## 8. 저장 스키마 (seed별, 원자적 저장)

NPZ(압축, **float32**): `shap_values[1567,474]` · `oof_probability[1567]` · `oof_raw_margin[1567]` ·
`base_value_per_row[1567]` · `fold_id[1567]` · `original_row_index[1567]` · `y_true[1567]` · `feature_active[5,474]`

metadata JSON: seed · fold train/val 인덱스 해시 · active 피처 목록·수·제거 피처명 · fold별 scale_pos_weight ·
모델 get_params 전체 · 전처리 파라미터 · SHAP 출력 원형 shape·정규화 방식 · additivity 최대 오차 ·
입력 데이터 해시 · 코드 해시 · 사전등록 해시 · Python/NumPy/pandas/sklearn/LightGBM/SHAP 버전 · 시작·종료시각 · NPZ 해시.

원자적 저장: tmp 기록 → 검증 → rename. 중간 실패 결과는 공식 결과로 승격하지 않는다.
캐시 재사용 = 코드·사전등록·데이터 해시 전부 일치 시에만.

## 9. 집계 (seed 단위)

seed별로: ① 전체 행 mean|SHAP| ② 양성 행 mean|SHAP| ③ 음성 행 mean|SHAP| ④ 전체 signed mean
⑤ 양성/음성 signed mean ⑥ 474 순위. **클래스 불균형 때문에 전체 평균만으로 불량 우선순위를 결론내리지 않는다**(양성 분리 병기).
**헤드라인 순위 = 20개 seed별 mean|SHAP| 벡터의 등가중 평균.** 행별 SHAP을 seed 간 먼저 평균하는 방식 금지(모델 혼합).

## 10. 순위·Top-k·Jaccard (고정)

- 지표: median rank · rank IQR · min/max · Top-1/3/5/10/20 진입률 · seed별 top-k 목록 · **seed 간 Jaccard 전체 분포**(쌍별 190개).
- 동점 규칙: 중요도 내림차순, 완전 동점은 **마스터 피처 순서**로 결정(안정 정렬) — 항상 정확히 k개.
- fold별 top-k는 참고(같은 seed 내부 fold 쌍만 비교).
- **임의의 안정/불안정 합격 기준을 만들지 않는다** — 연속 지표·분포 그대로 보고.

## 11. OOF 예측 성능 — QC 전용 원칙

OOF probability·AUC는 **파이프라인 QC로만** 저장(정본 0.731±0.014와의 sanity 대조).
금지: 새 성능 성과 홍보 · RF 교체 재평가 · champion 결정 변경 · 선택적 은폐. manifest/부록에 투명 기록하되
기존 성능 프로토콜과 다름을 명시하고 헤드라인·모델 선택에 사용하지 않는다.

## 12. 센서군 정의 (사후 집계 — 모델 피처 불변)

- 상관: 전체 1,567행 · 라벨 미사용 · 중앙값 대치 후 · Pearson + Spearman(rankdata 평균 동순위) — **전체 데이터 비지도 상관** 병기.
- similarity(i,j) = **max(|Pearson|, |Spearman|)**, distance = 1 − similarity.
- **주 분석 = complete-linkage**(precomputed condensed distance, `fcluster(t=1−thr, criterion='distance')`),
  thr ∈ {0.85, 0.90, 0.95}. 절단 후 **각 군집 내 모든 쌍 similarity ≥ thr을 사후 assert**.
- 민감도·부록: connected component(동일 thr 3종, **연쇄 연결 한계 명시**) · Pearson-only t=0.90 · Spearman-only t=0.90 ·
  기존 greedy(Pearson>0.95, 정본 290) 대응표. S59 직접 이웃 0(`s59_corr_sensitivity` 판정 A)과의 정합 확인.

## 13. 군집 기여도 (혼합 금지, 별도 명명)

- **A `group_additive_meanabs`**: 샘플별 군 내 signed SHAP 합 → |·| → 샘플 평균 (additivity 유지, 순 기여).
- **B `group_magnitude_sum`**: 군 내 피처 mean|SHAP| 합 (총 배분 기여량, 상쇄 불허).
- **C `cancellation_ratio` = A/B**: B=0→NaN, **0~1 범위 assert**(수치 허용오차 내), 작을수록 내부 상쇄 큼.
- 전체·양성·음성 각각 계산. seed별 산출 → seed 간 분포 보고.

## 14. 기존 bootstrap SHAP과의 구분

기존 `xai_stability.json` = **학습 데이터 내 bootstrap 순위 반복성**(in-bag, 전체 데이터 SHAP, OOF 아님).
본 실험 = **미관측 validation OOF 설명 안정성**. 두 결과는 **별도 표**로만 병기하고 통계적으로 병합하지 않는다.
불일치가 나오면 불일치 자체를 보고한다(기존 그림·서사 격하 가능성 수용).

## 15. S59 캐비앗 (결과와 무관하게 유지)

S59는 행 순서와 상관 ρ 약 −0.44, Q4 단독 AUC 약 0.514 — 실제 공정 신호와 시간·배치 대리 가능성을 분리할 수 없다.
OOF SHAP도 같은 89일 데이터 내부 보간이므로 **이 한계를 해소하지 못한다**. 원인·제어·PM 확정 표현 금지,
"우선 모니터링·DOE 후보"로만. 고상관 짝 미확인(Pearson·Spearman × 0.85~0.95) 결과는 상관 구조 진술로만 병기.

## 16. probe 승계 규칙 (사전 고정)

- seed 0을 구조·시간·용량 probe로 먼저 실행한다(**본 사전등록 승인 후 별도 실행 지시에서**).
- probe 결과의 순위·결론을 보고 규칙을 변경하지 않는다.
- 코드·사전등록·환경이 그대로이고 전 assert 통과 시 **seed 0 결과를 공식 결과로 승계**한다.
- 코드·규칙을 하나라도 변경하면 기존 probe를 격리(legacy 이동 아님 — 무효 표기)하고 seed 0부터 전체 재실행.
  변경 전후 해시와 사유를 보존한다.
- 예상 총 실행시간 **약 10~15분**. **seed 0이 30분 초과 시 중단.**

## 17. 중단 조건

사전등록/데이터 해시 불일치 · 마스터 474 불일치 · fold 단일 클래스 · validation 중복/누락 ·
SHAP shape 미지원 · SHAP/피처 정렬 불일치 · additivity 실패(§7) · NaN/Inf · seed 저장 실패 ·
예상 저장 용량 150MB 초과 · seed 0 실행 30분 초과 · 실행 중 규칙 변경 필요 발견.
중단 후 임의 수정·재개 금지 — 개정 승인 요청.

## 18. 보호 파일·격리

보호(무수정): `results/rf_lgbm_followup/`·`results/model_cmp/`·`results/xai_stability.json`·
`results/s59_corr_sensitivity/`·기존 그림·보고서·시뮬레이터·KEY_NUMBERS·PROJECT_STATE·원 데이터·전 cache.
신규 출력은 `results/oof_shap_v2/`에만.

---

## 부록 R1 — Revision 1 고정 규정 (기존 §와 충돌 시 R1 우선)

**R1-1. 고정 입력 해시**: 데이터 SHA-256을 문서·코드에 **상수로 고정**
(`e8b8187ce276db148707b5103463aa0f35cb8e521da2bc697c0fd45a6b06ad0f`). 코드가 실행 시 재계산해 이 고정값과
**직접 비교**, 불일치 시 모든 모드 즉시 중단(현재/기대 해시를 오류에 출력). 마스터 474도 동일 고정 검증.
사전등록 해시도 코드에 상수로 고정(**Revision 1 해시만 허용**) — 문서 변경 시 코드가 실행을 거부한다.

**R1-2. 환경 fingerprint**: {python, numpy, pandas, scipy, sklearn, lightgbm, shap, OS, CPU arch}를
정렬 JSON 직렬화 후 SHA-256. seed metadata·probe_summary·manifest에 저장. `cache_valid()`는
data/prereg/code/**env fingerprint** 4종 전부 일치 시에만 캐시 유효.

**R1-3. probe 승계·승격**: seed 0은 `results/oof_shap_v2/.probe_tmp/`에 먼저 저장하고,
시간(≤30분)·용량(예상 총합 ≤150MB)·shape·additivity·finite·coverage·전 배열 검증·metadata 검증을
**전부 통과한 뒤에만** 공식 seed 0 파일로 원자적 승격한다(실패 시 공식 파일 미생성).
`probe_summary.json`은 원자 저장(tmp→fsync→재열람 검증→rename)하며 최소 필드:
verdict(PASS/FAIL)·seed·data/prereg/code hash·environment_fingerprint·seed0_npz_hash·seed0_metadata_hash·
seed0_elapsed_seconds·seed0_file_size_bytes·projected_total_size_bytes·additivity_max_error·
validation_coverage_pass·shape_checks_pass·created_at.
`--run` 게이트는 파일 존재가 아니라 **10항 검증**(verdict PASS·해시 4종 일치·seed0 npz/metadata 해시 일치·
metadata↔summary 정합·QC 통과·용량·시간)을 전부 통과해야 하며, 하나라도 실패하면 새 --probe를 요구한다.
코드·사전등록·환경 중 하나라도 변경되면 기존 seed 0을 자동 승계하지 않는다.

**R1-4. 전 군집 계산·저장**: A/B/C는 **모든 군집**(singleton 포함, 전체 threshold·전체 방식)에 대해 계산·저장한다.
상위 30 제한은 **표시 단계에만** 적용하며 기준을 고정한다 — 1차: `group_magnitude_sum`(all 클래스, seed 평균) 내림차순;
동점: ① `group_additive_meanabs` 내림차순 ② 군집 크기 내림차순 ③ 가장 앞선 마스터 피처 순서.

**R1-5. 전체 cluster 정보 저장**: 각 설정(주: complete-linkage combined 0.85/0.90/0.95 ·
부록: CC combined 0.85/0.90/0.95 · Pearson-only CL 0.90 · Spearman-only CL 0.90 · legacy greedy 대응)마다
method_id·threshold·similarity_definition·`cluster_labels[474]`·전 군집의 {ID, 구성원 피처명·인덱스, 크기,
singleton/multi, 군집 내 최소/최대 pairwise similarity}·S59 군집 ID·구성원·총/singleton/multi 수·최대 크기를 저장한다.

**R1-6. similarity 수치 안정성**: 결합 similarity에 `np.clip(sim, 0, 1)` 적용, 대칭성·대각 1·distance 대각 0·
음수 거리 없음·NaN/Inf 없음 assert. complete-linkage 사후검증은 `min_pairwise_sim ≥ threshold − 1e-9`(**tolerance 1e-9 고정**).

**R1-7. fold Top-k/Jaccard(참고 분석)**: 각 seed×fold의 validation 전체 행 mean|SHAP| → 순위·Top-{1,3,5,10,20}.
같은 seed 내부 fold 쌍만 비교(10쌍/seed, **총 200쌍**), k별 Jaccard 전체 분포 저장.
저장 레코드: seed·fold_a·fold_b·k·intersection·union·jaccard·fold_a/b top-k 목록.
**양성 전용 fold 결과는 계산·저장하되 `exploratory_small_sample=true`로 명시**하고 주 결론·안정성 판정에 사용하지 않는다.
임의 합격선 없음.

**R1-8. 클래스별 seed 집계 실저장**: seed별 474 벡터 6종(mean_abs all/pos/neg, signed all/pos/neg)을
**aggregate_arrays.npz(float32)**에 원본 저장 + 각 벡터의 순위·top-k, 피처명, dtype, pos/neg 표본 수.
20 seed 통계(등가중 평균·sd·median·IQR·min/max)도 저장. "전체 평균만으로 불량 우선순위 결론 금지" 경고 유지.

**R1-9. 군집 A/B/C의 seed 분포**: 각 seed×cluster×클래스(all/pos/neg)별 A_seed·B_seed·C_seed(=A/B, B=0→null)를 저장
(`groups_contrib.npz` + JSON 통계). 최종 저장: seed별 A/B/C 목록 · 각 mean/sd/median/IQR/min/max · valid C seed 수.
ratio-of-means는 저장 시 `ratio_of_means`로 **별도 명명**(C_seed 분포와 혼동 금지).
**수치 오차 규칙(고정)**: C는 이론상 [0,1]; |이탈| ≤ 1e-8이면 [0,1]로 clip, 초과하면 중단.

**R1-10. 실제 모델 파라미터**: fold별로 **실제 fit한 객체의 `m.get_params(deep=True)` 전체**를 저장
(random_state=seed·fold별 spw·n_jobs 포함 확인). 템플릿 파라미터는 `template_params`로만 별도 저장.

**R1-11. SHAP 출력 QC 메타데이터**: fold metadata에 raw SHAP type·full shape, normalized shape,
raw expected_value type·shape, normalized ev, class-selection branch, raw margin shape, predict_proba shape 저장.
**logit QC(고정)**: `logit(clip(proba[:,1], ε, 1−ε)) ≈ predict(raw_score=True)`, **ε=1e-15, atol=rtol=1e-8**,
최대 오차 저장, 실패 시 중단. 이 QC는 성능 평가가 아니라 SHAP 클래스·raw-margin 의미 확인용이다.

**R1-12. NPZ·metadata 전수 검증**: 공식 rename 전에 — 필수 key 8종·정확한 shape·dtype·finite·
`original_row_index`=0~1566 순열·fold_id∈[0,4]·행당 fold 1개·y_true 원본 동일·proba∈[0,1]·margin/SHAP finite·
feature_active[5,474]·**inactive 열이 해당 fold val 행에서 0**·metadata↔NPZ seed 정합·coverage(seen==1)을 검증하고,
임시 NPZ를 재열람해 동일 검증 후 rename. metadata도 임시 저장→재열람 검증→rename.

**R1-13. manifest**: 전체 run(+aggregate+groups) 완료 후 원자 저장 — prereg/code/render/data hash·env fingerprint·
seed 목록·seed별 NPZ/metadata hash·aggregate/groups hash·시작·종료시각·전체 크기·QC 요약·보호 파일 hash snapshot.
manifest 자기 해시는 내부에 넣지 않는다(상위에서 관리).

## 부록 R2 — Revision 2 고정 규정 (R0·R1과 충돌 시 R2 우선)

**R2-1. seed metadata 의미 검증**: 전용 `validate_seed_metadata()`를 **공식 승격·cache_valid·run gate에서 공통 호출**.
검증(최소): 필수 key 전부 · seed==기대값(파일명과도 일치) · fold 수 5 · data/prereg/code/env 해시가 **현재 정본**과 일치 ·
NPZ 해시가 실제 파일과 일치 · coverage/shape QC PASS 플래그 · additivity·logit 오차 finite 및 허용범위 이내 ·
fold metadata 5개(각각 실제 model params·train/val 인덱스 해시 보유). JSON 재파싱만으로 통과 처리 금지 —
**NPZ와 seed·행 수(1567)·피처 수(474)·fold 구조(fold별 n_val ↔ fold_id 카운트)를 교차 검증**한다.
허용범위(임의 숫자 아님 — allclose 의미론적 상한에서 유도): additivity ≤ `ATOL + RTOL·max|raw_margin|`,
logit ≤ `LOGIT_TOL·(1 + max|raw_margin|)` (해당 NPZ에서 계산).

**R2-2. NPZ dtype 고정**: shap_values/oof_probability/oof_raw_margin/base_value_per_row = **float32** ·
fold_id/original_row_index/y_true = **정수형** · feature_active = **bool**. R1-12 검증에 dtype 8종을 포함하고,
임시 NPZ 재열람 후 동일 검사.

**R2-3. 6종 벡터 Top-k 저장**: mean_abs_{all,pos,neg} · signed_{all,pos,neg} **각각**에 rank와 Top-1/3/5/10/20을
실저장한다("mean_abs_all top-10만" 구조 폐지). 동점 = 값 내림차순 후 마스터 피처 순서.
**signed 의미(고정)**: ① `signed_desc_top{k}` = signed 값 자체 내림차순(불량 방향 밀기 상위) ·
② `signed_absmag_top{k}` = |signed mean| 내림차순(순방향 크기 상위) — **서로 다른 필드명으로 분리, 혼용 금지**.

**R2-4. commit marker 승격**: seed 산출물은 임시 경로에 NPZ+metadata 저장 → 두 파일 전수 검증 → 공식 경로 이동 →
**마지막에 `seed_XX.commit.json` 원자 저장**(seed·NPZ 해시·metadata 해시·data/prereg/code/env 해시·created_at·verdict PASS).
cache_valid·run gate는 **marker까지 정합해야만** 공식 인정. 부분 승격(한쪽 파일만 존재)·이전 실패 파일 재사용 금지.

**R2-5. probe summary·run gate 보강**: summary에 `logit_qc_pass`·`logit_max_error`·`metadata_validation_pass`·
`seed0_commit_hash` 추가. run gate 추가 확인: metadata seed==0 · metadata 4해시=현재값 · `cache_valid(0)` PASS ·
additivity/logit 오차 finite·허용범위(R2-1) · commit marker 해시 정합. malformed JSON은 명확한 메시지로 중단.

**R2-6. manifest 계보**: `run_started_at`·`run_completed_at`을 **별도 저장**(created_at으로 대체 금지 —
`--run`이 자기 실행 구간을 `run_times.json`에 원자 기록). aggregate·groups 결과에는
data/prereg/code/env 해시 + **사용한 20개 seed NPZ 해시·metadata 해시 + 자체 NPZ 해시**를 내장한다.
manifest 생성 시 존재 여부가 아니라 **내장 계보를 현재 source·실제 seed 해시 목록과 비교**하고,
aggregate JSON↔aggregate NPZ·groups JSON↔groups NPZ 해시를 확인해 **stale이면 생성 차단**.

**R2-7. 비차단 권고(반영)**: complete-linkage 사후검증은 **반올림 전 similarity**로 수행(반올림 값은 출력 전용) ·
fold Top-k 목록은 argsort(stable) 순서 기반 **결정적 정렬**(set 순회 금지) ·
150MB 한도는 **seed NPZ 중심 추정**이며 metadata·aggregate·groups 파일은 별도임을 명시(총량은 manifest에 실측 기록).

## 19. 개정 이력

| # | 일자 | 내용 |
|:--:|---|---|
| — | 2026-07-21 | 최초 작성(실행 전 확정. fit·SHAP·CV·프로브·상관 계산 미수행 상태에서 작성). SHA-256 `d47170b6…63b8` |
| 1 | 2026-07-21 | **Revision 1** — CODEX 실행 전 감사 반영(부록 R1 신설: 고정 해시 직비교·env fingerprint·probe 승격/승계 10항 게이트·전 군집 저장+표시 30 기준·similarity clip/1e-9·fold Jaccard 200쌍·클래스별 실저장·A/B/C seed 분포+1e-8 clip 규칙·실제 get_params·SHAP/logit QC(ε=1e-15, 1e-8)·전수 검증·manifest). **분석 실행 0건 상태의 개정 — 결과 기반 변경 아님.** SHA-256 `b550f995…5512` |
| 2 | 2026-07-21 | **Revision 2** — CODEX 재감사 FAIL(저장·검증 연결부) 보완(부록 R2 신설: metadata 의미 검증기 공통 연결·NPZ dtype 8종·6종 벡터 전부 Top-k(signed 이중 정렬 분리)·commit marker 승격·probe/run gate 보강·manifest 계보 stale 차단·비반올림 사후검증·결정적 정렬·용량 추정 한계). **분석 실행 0건 상태의 개정 — 결과 확인 전 변경.** |
