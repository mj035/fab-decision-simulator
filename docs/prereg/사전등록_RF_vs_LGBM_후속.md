# 사전등록 — 일반 RF vs LightGBM 후속 확인실험 (rf_lgbm_followup)

> **실행 전 확정 문서.** 실행 후 수정 금지. 수정 필요 시 개정 이력에 남기고 승인받는다.
> 작성 시점: 2026-07-21. **본 문서 작성 시점에 모델 학습·프로브·본 실행은 일절 수행되지 않았다.**
> 격리 원칙: 정본 코드·결과·그림·보고서·시뮬레이터·기존 model_cmp 산출물을 일절 수정하지 않는다.
> 신규 경로만 사용 — `src/rf_lgbm_followup.py`, `results/rf_lgbm_followup/`.

---

## 1. 실험 배경

- 보고서 4장(구 프로토콜, `src/seed_flip.py`, 2026-07-15)에서 **PCA 없는 일반 RandomForest**가
  LightGBM 대비 AUC **+0.023**(RF 평균 0.7541 vs LGBM, 20 seed 중 19승, 부호검정 p≈0.00004)으로
  유의 우세였다. 단 비용 판정은 88% 구간에서 동일(`결론_범위한정.md` §2②).
- 최신 모델 계열 비교 실험(`사전등록_모델계열비교.md`, 50 seed 완주)은 **일반 RF를 포함하지 않았다**
  (§2④ 미실행 명시). 포함된 `pca_rf`(PCA 95% + RF 500트리)는 **다른 모델**이며 그 열세는
  일반 RF에 대한 진술이 아니다.
- 구 프로토콜(seed_flip)은 outer train 전체 전처리 fit(누수 소지)·pooled 평가·절대임계 비용 등
  최신 규율 이전의 설계였다. → **선행 우위가 최신 leakage-free 프로토콜에서도 유지되는지 확인이 필요하다.**

## 2. 실험 질문

**"seed_flip에서 +0.023 우위를 낸 PCA 없는 일반 RF는, 최신 model_cmp와 동일한
누수 방지·paired·rank-q 프로토콜에서도 base_lgbm보다 우세한가?"**

## 3. 가설

- **주가설 H1**: rf_no_pca의 대응 ΔAUC(RF − LGBM)는 양수이며, thresholds.json의 AUC 기준
  (통계적: 대응 95%CI 0 배제, 실질적: ΔAUC ≥ +0.02)을 충족한다.
- **부가가설 H2**: AUC 기준을 충족하더라도 비용 기준(cl 10~20 대응 비용차 95%CI 0 배제 +
  베이스라인 대비 ≥3.5% 절대비용 감소)은 **충족하지 못한다** — 구 실험에서 "AUC 유의하나
  비용 88% 동일"이었고, 헤드라인 3("AUC 우위가 운영점 우위를 보장하지 않음")과 정합적 예상.
- **예상이 틀려도 기준은 바꾸지 않는다** (모델계열비교 §8과 동일 원칙).

## 4. 비교 arm — 정확히 2개, 추가 금지

| arm | 전처리 | 모델 | 귀속 |
|---|---|---|---|
| `base_lgbm` | 중앙값대치 → VarThr(0.0) → 표준화 (fold 내부 fit) | LightGBM 정본 설정 | `src/model_cmp.py:101-103` |
| `rf_no_pca` | 위와 동일 | RandomForest (seed_flip 설정) | `src/seed_flip.py:40-41` |

PCA 없음 · SMOTE 없음 · ADASYN 없음 · 기타 오버샘플링 없음. 다른 arm 추가 금지.
**본 실험은 기존 model_cmp의 arm 추가가 아니라 완전히 분리된 후속 확인실험이다**
(별도 experiment_id, 별도 캐시, 별도 결과 디렉터리).

## 5. RF 설정 귀속 근거 (`rf_no_pca`)

출처: `src/seed_flip.py:40-41` (SHA-256 `72ba26b6da73ba8f0f495a73d80b4fc0cf955336d5bb8d2ec144a7f86ff297cc`).
+0.023 우위를 실제로 낸 구성을 **그대로 복원**한다. 책임자 확정 사항 — 재질문·임의 변경 금지.

**명시 파라미터 (seed_flip 원문):**

```
n_estimators      = 400
class_weight      = "balanced_subsample"
min_samples_leaf  = 3
random_state      = 0          ← seed_flip 원문 그대로 (아래 ⚠️ 참조)
n_jobs            = (실행환경 설정으로 분리 — §16, 통계 파라미터 아님)
```

**명시되지 않아 sklearn 기본값을 쓰는 파라미터 (sklearn 1.9.0 기준, 사전등록으로 고정):**

```
criterion="gini", max_depth=None, max_features="sqrt", min_samples_split=2,
bootstrap=True, max_leaf_nodes=None, min_weight_fraction_leaf=0.0,
min_impurity_decrease=0.0, oob_score=False, ccp_alpha=0.0, max_samples=None
```

기본값 고정 방법: 위 값들을 **실행 코드에 전부 명시적 키워드 인자로 기입**한다.
sklearn 버전이 바뀌어도 실험 설정이 변하지 않도록 기본값 의존을 제거한다.
버전 기록: python 3.12.10 / sklearn 1.9.0 / lightgbm 4.6.0 / numpy 2.4.4 / pandas 3.0.3 / scipy 1.17.1.

**⚠️ 모델 random_state 의미 차이 (비교 조건 차이의 명시, §3-3 공정성 조건 부속):**
seed_flip은 RF·LGBM 모두 `random_state=0` 고정(seed는 CV 분할만 변동)이었다.
최신 model_cmp의 base_lgbm은 `random_state=seed`(모델 난수도 seed 연동)다.
책임자 결정("seed_flip의 실제 구현을 그대로 복원" / "최신 base_lgbm 설정을 정확히 복원")을
문자 그대로 집행하면 **rf_no_pca는 random_state=0 고정, base_lgbm은 random_state=seed**가 된다.
이 비대칭은 paired 설계(동일 fold 공유)를 깨지 않으며, RF의 seed 간 분산을 CV 분할 변동만으로
제한한다(정본 `cms_seeds` 캐시도 "seed는 CV 분할만, 모델 rs=0 고정" 선례 — KEY_NUMBERS 헤드라인 2번 주석).
**본 문서에 명시하고 결과 보고 시 병기한다. 결과를 본 후 이 선택을 바꾸지 않는다.**

**사용하지 않는 설정 (혼입 금지):** model_cmp `pca_rf`의 (n_estimators=500, min_samples_leaf=2,
class_weight="balanced", PCA 95%). 두 RF 설정을 별도 arm으로 동시 비교하지도 않는다.

## 6. LGBM 설정 귀속 근거 (`base_lgbm`)

출처: `src/model_cmp.py:101-103` (SHA-256 `3e1e915b760f685352be0468893161559067116382abb2154be2cd5923e12c43`).

```
boosting_type="gbdt"(기본), n_estimators=300, learning_rate=0.03, num_leaves=7,
max_depth=-1(기본, 제한 없음 — num_leaves=7이 실질 제약), min_child_samples=25,
subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
scale_pos_weight=(학습 부분집합의 N_neg/N_pos — fold 내부에서 계산),
random_state=seed(모델 난수 seed 연동 — model_cmp 원문 그대로), verbose=-1,
n_jobs=(실행환경 설정으로 분리 — §16)
```

기본값 의존 항목(boosting_type, max_depth 등)도 실행 코드에 명시 기입해 고정한다.
lightgbm 4.6.0 기준.

## 7. 데이터와 라벨 정의

- 데이터: `data/fab_process_yield.csv`
  (SHA-256 `e8b8187ce276db148707b5103463aa0f35cb8e521da2bc697c0fd45a6b06ad0f`, 2026-07-06 18:19).
- 로딩: `preprocess_adapter.adapt_from_csv(csv, "Pass/Fail", 1, drop_cols=["Time"])` —
  model_cmp와 동일 정본 경로. 라벨 방향: Pass/Fail == 1 → 양성(불량). Time 열은 사용하지 않는다
  (23차: DD/MM 파싱 손상 확정. 시간축 필요 시 행 인덱스만).
- N=1567, 양성 104(6.64%). 전역 열 필터 없음(상수열 제거는 fold 내부 VarianceThreshold).

## 8. 전처리 순서와 누수 방지 규칙

`중앙값 대치 → VarianceThreshold(0.0) 상수 제거 → StandardScaler 표준화 → 모델 학습`.
**세 전처리기 모두 해당 training subset 내부에서만 fit** (model_cmp `prep()`과 동일 구현).

금지(코드에서 구조적으로 차단): 전체 데이터 중앙값/상수열/표준화 · outer test 포함 전처리 적합 ·
outer test를 이용한 모델/q/threshold/비용정책 선택 · 전체 OOF를 보고 사후 정책 재선택.

## 9. CV·paired 비교 정의

- 총 **50 seed**. seed별 `StratifiedKFold(5, shuffle=True, random_state=seed)` outer.
- **두 arm이 동일 seed에서 동일 outer fold 배열을 공유**(같은 split 객체 재현) — 대응 설계.
- inner: 각 outer training fold 내부 `StratifiedKFold(3, shuffle=True, random_state=seed)` —
  model_cmp `inner_eval`과 동일. 두 arm이 동일 inner fold 공유.
- paired ΔAUC = (rf_no_pca − base_lgbm), seed 단위 대응.
- 통계적 유의성은 **대응 CI로만** 판단. 개별 모델 seed SD 사용 금지(thresholds.json 원문 주의사항).

## 10. 주지표와 참고지표

- **주지표**: outer fold별 ROC-AUC의 seed 내 평균(fold-mean) → 50 seed paired 비교.
- **참고지표**(판정에 사용 금지): pooled OOF AUC, 각 모델 seed별 분산, fold별 원자료, paired 차이 분포 전량.

## 11. 비용정책 — rank/q 승계

model_cmp §2.4를 **그대로 승계**한다:
- 정책 = "점수 상위 q% 검사". q 격자 = `linspace(0.005, 0.995, 300)` (inner·outer 동일).
- 순위는 **각 모델 fit 내부에서만** 산출. fit 간 점수 결합 금지(inner 포함 — model_cmp 수정 (8)).
- **cutoff 동점은 전부 포함**, 목표 q·실제 검사율·동점 초과폭을 모두 저장(수정 (9)).
- inner 집계: AUC = fold별 AUC의 **평균** / 비용 = fold별 q 혼동행렬의 **합**. 혼용 금지.
- 비용 커널 = 정본 `cost_model.CM` 그대로(`evaluate`, `best_threshold_idx`, `_accumulate`).
  cl 2~64, cs=0, β=0.95, α=0.02. cl 10~20 대응 비용차가 판정 축(thresholds.json 원문).
- **점수 표현 규칙**: 두 모델의 predict_proba 출력은 보정 검증이 없으므로
  "위험점수(ranking score)"로만 표현한다. "보정된 불량확률"·"실제 불량확률" 표현 금지.

## 12. 판정 기준 — thresholds.json 승계 선언

> **"본 실험의 판정 기준은 실행 전 동결된 thresholds.json을 그대로 승계하며,
> 결과 확인 후 재유도하거나 변경하지 않는다."**

- 정본: `results\model_cmp\thresholds.json`
- SHA-256: `16b3e481bf061f2311585ba341c6d3bfe4eff5501add4e5223cde87b293e65ac`
- 파일 수정 시각: 2026-07-21 01:25:46 (+0900) — **본 실험 설계 착수 전에 동결된 파일이다.**

**판정 기준 원문 (`thresholds.json` → `최종_판정기준`, 그대로 전재):**

```json
"AUC": {
  "통계적": "대응 ΔAUC 95%CI(동일 seed·fold)가 0을 배제",
  "실질적": "ΔAUC ≥ +0.02",
  "주의": "개별 모델 seed SD로 유의성 판단 금지(리뷰어 지시)"
},
"비용": {
  "통계적": "cl 10~20 대응 비용차 95%CI가 0을 배제",
  "실질적": "베이스라인 대비 3.5% 이상 절대비용 감소"
},
"전방": "walk-forward 스텝별 AUC가 −0.015를 넘어 악화되지 않을 것",
"교체검토_조건": "위 3개 **모두** 충족. AUC만 충족 시 비교 결과로만 보고하고 교체하지 않는다."
```

**코드에서 해석한 판정 로직:**

```
AUC PASS  = (대응 ΔAUC 95%CI 하한 > 0) AND (평균 ΔAUC ≥ +0.02)
비용 PASS = (cl 10~20 대응 비용개선률 95%CI 하한 > 0) AND (평균 개선률 ≥ +3.5%)
전방 PASS = (3스텝 모두에서 대응 Δ 95%CI 상한이 −0.015 미만으로 떨어지지 않음, §4-bis 규칙)
교체 후보 = AUC PASS AND 비용 PASS AND 전방 PASS
```

**원문↔구현 자체 점검표:**

| 원문 항목 | 구현 | 일치 |
|---|---|:--:|
| AUC 통계적: 대응 CI 0 배제 | `ci_lo > 0` (양방향 t 기반 95%CI, 개선 방향) | ✅ |
| AUC 실질적: ΔAUC ≥ +0.02 | `mean_dA >= 0.02` | ✅ |
| 비용 통계적: cl10~20 대응 CI 0 배제 | cl 10~20 각 cl의 (base−arm)/base% 평균의 대응 CI, `ci_lo > 0` — model_cmp `finalize()` 동일 축 | ✅ |
| 비용 실질적: ≥3.5% | `mean_dC >= 3.5` (`권고_실질개선_기준_%` 필드값을 코드가 파일에서 직접 읽음) | ✅ |
| 전방 −0.015 | 모델계열비교 §4-bis 원문 규칙("어느 한 스텝이라도 CI 상한 < −0.015면 악화") 그대로 | ✅ |
| 3개 모두 충족 시만 교체검토 | AND 결합 | ✅ |

**충돌 점검**: 책임자 요약("최소 고정 기준: ΔAUC≥0.020 / 3.5% / −0.015")과 원문 사이 충돌 없음 —
요약은 '최소' 기준이며 원문은 여기에 통계적 조건(대응 CI 0 배제)을 **이미 포함**하고 있다.
CI 조건은 **새로 추가된 관문이 아니라 원문 승계분**이다. 원문 외 신규 조건
(cl=14 단독 유의성, 새 p-value 기준, 새 비용비, 새 효과크기, 결과 후 wf 기준)은 **추가하지 않는다.**
paired 95%CI와 정확 검정 p-value(wilcoxon)는 **필수 산출물이되, 원문에 통과 조건으로 명시되지 않은
부분(p-value 자체)은 보조 통계로만 보고**한다. p-value는 원시값·과학적 표기·표시용 반올림을 구분 저장한다
(단순 `0.0` 기록 금지 — model_cmp에서 `round(p,4)=0.0` 반올림 이력, 실제 7.6e-10).

## 13. walk-forward 조건부 실행 규칙

**이번 준비 단계와 본 실행 단계에서는 wf를 실행하지 않는다.** 코드상 `--walk-forward` 모드는
finalize 결과 JSON에서 **AUC PASS AND 비용 PASS**가 확인될 때만 열린다(미충족 시 즉시 중단 + 사유 기록).

승계 정본: thresholds.json(−0.015, wf_v2 실측 seed 잡음 유래) + 모델계열비교 사전등록 §4-bis.
- **20 seed** 고정. 외부(시간) 분할 고정 — seed는 내부 임계값 선택 fold + 모델 난수만 변동(분산 과소추정 병기).
- 시간축 = **행 인덱스(생산 순서)**. `sort_values('Time')` 금지(손상 Time 열 사용 금지).
  스텝 정의 = `fwd_cost.py:42` 원문 그대로: `q = floor(arange(N)/N*4)` quartile,
  3스텝 Q1→Q2 / Q1~2→Q3 / Q1~3→Q4.
- 정책은 §11의 동일 q 격자를 학습·전방 블록에 적용. 절대 임계값 배열의 fit 간 이전 금지.
- 판정: 스텝별 전방 AUC의 대응 Δ(arm − base_lgbm). **어느 한 스텝이라도 대응 Δ 95%CI 상한이
  −0.015 미만이면 "악화"** → 교체 검토 제외. 집계·pass/fail 규칙 §4-bis 원문 그대로, 신규 기준 없음.
- base_lgbm은 비교를 위해 항상 함께 실행. Q2 구간은 커널 권고 전략(원시 argmin) 병기
  (전수통과 권고 구간 — AUC 단독 해석 오도 방지).

**§13-bis. 실측 경로 구현 완료 선언 (개정 1 — 실행 0건 시점, CODEX 보완 지시)**

결과 확인 후 wf 구현을 추가하면 사후 유연성이 생기므로, **어떤 모델·프로브도 실행하지 않은
현 시점에 `--walk-forward`의 실측 계산 경로를 완성**했다. 승계한 함수·코드 경로:

| 승계 항목 | 원본 | 본 코드 |
|---|---|---|
| 스텝 분할(행 인덱스 quartile, 3스텝) | `fwd_cost.py:42` (`q=floor(arange(N)/N*4)`) 원문 | `wf_steps()` |
| 스텝 이름 | `fwd_cost.py:91` (`Q1~{k+1}→Q{k+2}`) 원문 | `wf_steps()` |
| 비용 계산 | `fwd_cost.py:79-83` — `CM.evaluate`(cl=15, cs=0, β=0.95, α=0.02) + 배포권고=`CM.CM_raw` 원시 argmin(28차 (a)) | wf 루프 내 동일 호출 |
| 학습 블록 inner fold 수(5) | `fwd_cost.py:73` (`StratifiedKFold(5)`) | `WF_INNER_FOLD=5`, `wf_inner_cm()` |
| inner 집계 방식 | 모델계열비교 §4-bis(q 격자·fold별 혼동행렬 합 — 수정 (8) 원칙. fwd_cost의 절대임계 pooled OOF를 §4-bis가 q격자로 대체) | `wf_inner_cm()` |
| 정책·동점 처리 | model_cmp §2.4 (`cms_rank` 동일 구현) | `cms_rank()` 재사용 |
| 20 seed·seed 의미(분산 과소추정 병기) | §4-bis + `fwd_cost.py:48-53` | `WF_NSEED=20` + `seed_의미` 필드 |
| 악화 기준 −0.015 | thresholds.json `walk_forward.seed_잡음`(실행 시 직독, 0.015 불일치 시 중단) + §4-bis "CI 상한 < −0.015" | 게이트 첫 검사 + `worse` 판정 |

게이트 순서(코드 강제): thresholds 정합 → finalize JSON의 AUC·비용 **둘 다 PASS** → 해시 재대조 →
출력 파일 무존재 확인 → **이후에만** 데이터 로딩·모델 fit 시작. 미충족 시 fit 0건으로 중단.

**wf_result.json 스키마 (지금 고정 — 실행 후 변경 금지):**

```json
{
 "experiment_id", "프로토콜", "seed_의미", "허용기준": -0.015,
 "steps": [
   {"step", "test_n", "test_npos", "유병률_%",
    "base_lgbm": {"AUC_mean","AUC_sd","AUC_seed별[20]","cl15_절감률_mean_%","커널권고_raw"},
    "rf_no_pca": {같은 필드},
    "대응_ΔAUC(RF−LGBM)": {"평균","95%CI","seed별[20]"},
    "악화": bool}   × 3 스텝
 ],
 "판정": {"악화_스텝": [...], "전방": "PASS|FAIL", "규칙": "§4-bis 원문"},
 "code/data/prereg/thresholds_sha256"
}
```

판정 로직(고정): `전방 = FAIL iff 존재 스텝 s.t. 대응 Δ 95%CI 상한 < −0.015`. 신규 분할·집계·조건 없음.

## 14. 기존 20-seed 결과와의 비병합 원칙

> **"본 실험은 기존 일반 RF의 선행 우위를 최신 동일 프로토콜에서 확인하는
> 별도 사전등록 비교이며, 기존 20-seed 결과와 신규 50-seed 결과를
> 통계적으로 병합하지 않는다."**

금지: 70-seed 결합 평균 · p-value 통합 · CI 통합 · 구 절대임계 비용과 신규 rank/q 비용의 직접 통합 ·
기존 결과를 신규 프로토콜의 일부처럼 취급 · 기존 19/20을 신규 실험의 확증 결과로 재사용.

기존 결과의 허용 용도: ① 방향 일치 여부 확인 ② rf_no_pca 설정 귀속 근거 ③ 선행 탐색이라는
역사적 배경 제시. **최종 보고서에서 두 실험은 별도 표로 병기**하며 프로토콜 차이
(20 vs 50 seed · outer-train 전처리 vs fold 내부 fit · pooled vs fold-mean · 절대임계 vs rank/q)를 명기한다.
**판정 근거는 신규 50-seed 실험 단독이다.**

또한 `results/report.json`은 역사적 산출물로만 취급하며, 그 안의 철회된 주장
("RF-LGBM 차이 무의미"(ceiling_proof, 비대응 분산 오류·자기정정 #5로 철회) ·
"AUC 0.75 천장" · "LGBM이 모든 모델 중 최고")을 본 실험의 어떤 문서·코드에도 사용하지 않는다.

## 15. 캐시·manifest 정책

- 캐시: `results/rf_lgbm_followup/cache/{arm}_s{seed}.npz` + 사이드카 `{arm}_s{seed}.meta.json`.
- meta 필수 필드: experiment_id, arm, seed, outer/inner fold 정의(spec+해시), 완료 상태, 생성 시각,
  python/sklearn/lightgbm/numpy/pandas 버전, 코드 SHA-256, 데이터 SHA-256, 사전등록 SHA-256,
  thresholds.json SHA-256, 모델설정 해시, 전처리설정 해시, q grid 해시, 비용설정 해시,
  fold index 해시, 실행환경 thread 설정, 예외/실패 상태.
- **재사용 조건**: experiment_id·arm·seed·코드/데이터/사전등록/thresholds/모델설정/전처리/
  q grid/비용설정/fold index 해시·완료 상태 **전부 일치**할 때만. 하나라도 다르면 무효
  (파일명만으로 판단 금지 — model_cmp 캐시키의 코드해시 미포함 결함을 본 실험에서 수정).
- 저장: tmp 기록 → flush → 완료 상태 기입 → atomic rename(`os.replace`) → 재검증.
  부분 캐시는 meta의 `status != "complete"`로 차단.
- manifest: `run_manifest_template.json`(본 단계 산출)을 실행 시점에 채워
  `run_manifest.json`으로 확정(본 단계에서는 template만 생성).

## 16. 실행환경·자원 제한

- 스레드: RF `n_jobs`·LGBM `n_jobs`는 **통계 하이퍼파라미터가 아니라 실행환경 설정**으로 취급.
  기본 보수값 1(중첩 병렬 OOM 이력 — fwd_cost 28차 (b)와 동일 원인 예방), 환경변수 `RFL_NJOBS`로
  조정 가능하되 **manifest에 실제 사용값 기록**.
- 중단 조건(하나라도 발생 시 본 실행 미개시 또는 안전 중단): 예상 완료가 지정 컷 초과 ·
  컷 30분 전 안전 완료 불가 예상 · 프로세스 메모리 > 시스템 가용의 80% · swap 급증 · OOM ·
  비정상 종료 · seed당 시간 비정상 증가(프로브 대비 2배↑) · 불완전 캐시/해시 불일치.
- **실행 컷**: 본 준비 단계에는 적용하지 않는다(실행 없음). CODEX 사전감사 승인 후
  별도 실행 지시에서 정확한 KST 컷을 받아 `--run --cut <ISO시각>` 인자로 고정한다.

## 17. 프로브 통과 조건

- 프로브 = **3 seed**(seed 0,1,2) × 2 arm. 산출: seed당 소요시간, 50 seed 예상 총시간, 피크 메모리(가능 시).
- 통과: 예상 총시간이 예산(6h, model_cmp §7-1 승계) 이내 AND 메모리 조건 위반 없음.
- 프로브 완료·통과 전 본 실행 차단(코드 강제). 프로브 캐시는 본 실행에서 재사용(동일 해시 조건).

## 18. 부분완료 규칙 (모델계열비교 §7-3 승계)

두 arm 공통 연속(seed 0부터) **30개 이상**일 때만 판정. 사용 범위는 공통 연속 최소 개수(결과 기반 선별 금지).
30 미만이면 "시도·미완"으로 기록하고 판정하지 않는다. n<50 판정 시 "n=NN seed (사전등록 §18)" 병기.

## 19. 결과 해석 허용 범위·금지 표현

허용: "사전등록된 2-arm 비교에서 rf_no_pca가 기준을 [충족/미충족]했다" + 판정별 후속
(AUC·비용·전방 모두 PASS → 교체 후보 지정 / AUC만 PASS → LGBM 유지 + RF는 AUC 참조 모델 /
AUC FAIL → LGBM 유지, 본 확인실험에 따른 교체 검토 종료).

금지: "LightGBM이 모든 모델 중 최고" · "RF가 모든 데이터에서 최고" · "AUC 0.75 절대 천장" ·
전 모델공간으로의 확대 해석 · `표현규칙_금지표현.md`의 금지어("차이가 없다"→"검출되지 않았다" 등) ·
스텝 평균 인용(전방) · 정본 수치(31.0% 등, 절대임계 축)와 rank/q 축의 직접 비교.

## 20. 예상 출력 파일 (실행 단계별)

| 단계 | 파일 |
|---|---|
| 준비(본 단계) | 본 문서, `src/rf_lgbm_followup.py`, `results/rf_lgbm_followup/run_manifest_template.json`, `model_config_comparison.md`, `README.md` |
| 프로브 | `results/rf_lgbm_followup/probe_result.json`, cache 6개(2 arm × 3 seed) |
| 본 실행 | cache 100개(2×50), `run_manifest.json` |
| 집계 | `rf_lgbm_followup.json`, `per_seed_results.csv`, `per_fold_results.csv`, `cost_by_ratio.csv` |
| 전방(조건부) | `wf_result.json` (AUC·비용 PASS 시에만) |

## 21. 해시 기록 방식

- 본 문서·실행 코드·데이터·thresholds.json의 SHA-256을 `--prepare` 모드가 계산해
  manifest에 기록하고, 이후 모든 모드가 **실행 시점 재계산값과 대조**한다(불일치 시 즉시 중단).
- 설정 해시 = 모델 파라미터 dict·전처리 순서 문자열·q grid 배열·비용 상수(cl 격자, cs, β, α)의
  canonical JSON을 SHA-256. fold index 해시 = seed별 outer/inner 분할 인덱스 배열의 SHA-256.

## 22. 참조 정본 해시 (작성 시점 고정)

| 파일 | SHA-256 |
|---|---|
| `results/model_cmp/thresholds.json` | `16b3e481bf061f2311585ba341c6d3bfe4eff5501add4e5223cde87b293e65ac` |
| `src/seed_flip.py` (RF 귀속) | `72ba26b6da73ba8f0f495a73d80b4fc0cf955336d5bb8d2ec144a7f86ff297cc` |
| `src/model_cmp.py` (LGBM 귀속·프로토콜) | `3e1e915b760f685352be0468893161559067116382abb2154be2cd5923e12c43` |
| `src/cost_model.py` (비용 커널) | `148dadfd3ec319da603cca32fbddb0380ed2867f9e0ba8f66257c38ad63b1c96` |
| `src/preprocess_adapter.py` (데이터 로딩) | `de802d7db7207afb94204c40fcbe994ad2438b6a36038f811e03f0f3061baa24` |
| 데이터 CSV | `e8b8187ce276db148707b5103463aa0f35cb8e521da2bc697c0fd45a6b06ad0f` |

## 23. 개정 이력

| # | 일자 | 내용 |
|:--:|---|---|
| — | 2026-07-21 | 최초 작성 (실행 전 확정. 모델 학습·프로브 미수행 상태에서 작성됨) |
| 1 | 2026-07-21 | **CODEX 보완 지시(실행 0건 시점)**: `--walk-forward` 실측 경로를 선언 차단에서 **완전 구현**으로 교체(§13-bis 신설 — 승계 코드 경로 표·결과 스키마·판정 로직 고정). 게이트(AUC·비용 PASS 시만, fit 이전 중단)·20 seed·−0.015·3스텝·행 인덱스 분할 불변. 신규 분할·집계·조건 추가 없음. random_state(RF=0/LGBM=seed)·arm 2개 유지. **모델·프로브·CV·캐시 실행은 여전히 0건.** |
