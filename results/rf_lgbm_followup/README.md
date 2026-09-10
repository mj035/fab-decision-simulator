# rf_lgbm_followup — 일반 RF vs LightGBM 후속 확인실험 (격리 디렉터리)

> 사전등록: `../../사전등록_RF_vs_LGBM_후속.md` · 코드: `../../src/rf_lgbm_followup.py`
> 판정 정본: `../model_cmp/thresholds.json` (승계 — 수정·재유도 금지)
> **상태: 준비 단계 완료. 모델 학습·프로브·본 실행·walk-forward 일절 미수행.**

## 격리 원칙

이 디렉터리와 `src/rf_lgbm_followup.py`, `사전등록_RF_vs_LGBM_후속.md`만이 본 실험의 파일이다.
기존 model_cmp(`../model_cmp/`)·정본 문서·시뮬레이터·APS 파일은 읽기만 하고 수정하지 않는다.

## 실행 순서 (CODEX 사전감사 승인 후, 별도 실행 지시 필요)

```
python -X utf8 src/rf_lgbm_followup.py --prepare        # 해시·환경 검증 (실행 없음)
python -X utf8 src/rf_lgbm_followup.py --probe          # 3 seed × 2 arm 프로브
python -X utf8 src/rf_lgbm_followup.py --run --cut 2026-MM-DDTHH:MM   # 컷 필수
python -X utf8 src/rf_lgbm_followup.py --finalize       # 집계·판정 (thresholds 원문)
python -X utf8 src/rf_lgbm_followup.py --walk-forward   # AUC·비용 PASS 시에만 열림
python -X utf8 src/rf_lgbm_followup.py --verify-cache   # 캐시 무결성 검사 (읽기 전용)
```

- 스레드: `RFL_NJOBS`(기본 1 — OOM 이력 예방). manifest에 실제값 기록.
- 프로브 미통과·해시 불일치·컷 미지정 시 본 실행이 **코드 수준에서 차단**된다.
- import만으로는 데이터 로딩·학습이 일어나지 않는다 (`__main__` 가드).

## walk-forward 실측 경로 (개정 1 — 실행 0건 시점에 완성)

사후 유연성 차단을 위해 **어떤 모델도 실행하지 않은 시점**에 wf 계산 경로를 완성했다.
게이트(finalize의 AUC·비용 둘 다 PASS + 해시 재대조)를 전부 통과하기 전에는 모델 fit이 시작되지 않는다.

**승계한 함수·코드 경로** (상세 표는 사전등록 §13-bis):
- 스텝 분할·이름: `src/fwd_cost.py:42,91` 원문 (행 인덱스 quartile 3스텝, 손상 Time 열 미사용)
- 비용 계산: `src/fwd_cost.py:79-83` 동일 — `cost_model.CM.evaluate`(cl=15,cs=0,β=0.95,α=0.02) + 배포권고 `CM.CM_raw`(원시 argmin)
- inner fold 수(5): `src/fwd_cost.py:73` / 집계는 §4-bis(q격자·fold별 혼동행렬 합, `cms_rank` 재사용)
- 20 seed·악화기준 −0.015: `../model_cmp/thresholds.json`(실행 시 직독 검증) + 모델계열비교 §4-bis
- 결과 스키마·판정 로직: `run_manifest_template.json`의 `walk_forward` 섹션에 지금 고정

## 이 단계에서 존재하는 파일 (준비 산출물 3종)

| 파일 | 역할 |
|---|---|
| `run_manifest_template.json` | 실행 시점에 채워질 manifest의 template (판정기준 원문·해시 슬롯 포함) |
| `model_config_comparison.md` | seed_flip RF / pca_rf / rf_no_pca / base_lgbm 4-설정 대조와 귀속 근거 |
| `README.md` | 본 문서 |

## 실행 후 생성될 파일 (아직 없음 — 존재하면 안 됨)

`probe_result.json` · `cache/*.npz` + `cache/*.meta.json` · `run_manifest.json` ·
`rf_lgbm_followup.json` · `per_seed_results.csv` · `per_fold_results.csv` ·
`cost_by_ratio.csv` · `wf_result.json`

## 인용 규칙 (요약 — 정본은 사전등록 §14·§19)

- 기존 20-seed(seed_flip)와 신규 50-seed는 **통계 병합 금지**. 별도 표 병기만.
- 판정은 사전등록된 2 arm에만 적용. "모든 모델 중 최고" 류 확대 해석 금지.
- 점수는 "위험점수(ranking score)"로만 표현. 보정 검증 없는 확률 표현 금지.
