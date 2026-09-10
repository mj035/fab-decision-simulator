# fab-decision-simulator

## 반도체 공정 센서 데이터 기반 검사 정책 의사결정 시뮬레이터

---

## 프로젝트 개요

본 프로젝트는 반도체 공정 센서 데이터로 불량을 예측하는 데서 끝나지 않고, 그 예측 점수를 입력으로 받아 **실제 검사 의사결정**까지 도출하는 시뮬레이터를 개발했습니다.

현장의 검사 정책은 두 비용 사이의 절충입니다. 불량을 놓치면 유출 비용이 발생하고, 정상품을 재검사하면 과검사 비용이 발생합니다. 이 프로젝트는 사용자가 두 비용의 비율을 입력하면 **Cost Layer**가 전수통과, 전수검사, 선별검사 세 전략의 총비용을 비교해 최소 비용 전략과 검사 임계값을 제시하고, **OOF SHAP**으로 어떤 센서를 먼저 점검해야 하는지 우선순위를 함께 보여줍니다.

핵심 관찰은 두 가지입니다. 비용 구조를 올바르게 파악하는 것이 모델을 교체하는 것보다 총비용에 더 크게 작용했고, AUC가 높은 모델이 운영점에서의 비용 우위를 보장하지 않았습니다.

## 시스템 구조

```
공정 데이터 CSV
      │
      ▼
전처리 (중앙값 대치 → 상수 센서 제거 → 표준화, fold 내부 fit)
      │
      ▼
LightGBM  nested 5-fold  →  OOF 예측 점수 (50 seed)
      │                         │
      ▼                         ▼
OOF SHAP                  Cost Layer
센서 기여도 / 군집화        비용비 → 혼동행렬 → 최소 비용 임계값
      │                         │
      └──────────┬──────────────┘
                 ▼
        웹 시뮬레이터 (web/simulator.html)
        추천 전략 / 임계값 / 절감률 / 점검 우선순위
```

- **데이터**: UCI SECOM. 1,567 생산 단위, 590 센서, 불량 104건(6.64%), 결측 4.54%. 상수 센서 116개를 제거하고 474개를 사용
- **Cost Layer**는 예측 점수가 아니라 혼동행렬 배열만 입력으로 받는 모델 독립 계층입니다 (`src/cost_model.py`)
- **시뮬레이터**는 데이터와 그림을 인라인한 정적 HTML 한 파일이라 서버 없이 열립니다

## 검증 방식

성능이 과대평가되지 않도록 다음 원칙을 적용했습니다.

- 전처리를 fold 내부에서 fit하고 OOF 예측만 사용
- nested CV로 임계값 선택과 평가를 분리
- 50 seed 짝지은 검정, fold 평균을 주지표로 사용
- 생산 순서 기준 walk-forward로 미래 구간 열화 확인
- 모델 교체와 튜닝 판정은 기준을 [사전등록 문서](docs/prereg/)에 먼저 고정한 뒤 실행

## 결과 요약

| 항목 | 값 | 조건 |
|---|---|---|
| OOF ROC-AUC | 0.731 ± 0.014 | LightGBM, nested 5-fold, 50 seed |
| 비용 절감률 | 30.9% | cl=15, 전수검사 대비 선별검사, 50 seed 중앙값 |
| 검사율 / 재현율 | 39.4% / 70.7% | cl=15 |
| walk-forward 절감률 | Q3 +53.1%, Q4 +11.6% | 20 seed. Q2 구간은 선별이 성립하지 않아 전수통과 권고 |
| 모델 손익분기 | cl = 16 (CI 14 ~ 16) | 단일 센서 대비 |
| 일반화 검증 | 절감 92.9% | SCANIA APS 60,000행, 비용비 50:1 |

RandomForest는 랜덤 CV에서 LightGBM보다 AUC가 높았지만, 사전등록한 walk-forward 조건을 통과하지 못해 교체를 보류했습니다.

프로젝트 중 철회하거나 정정한 항목도 그대로 남겼습니다. 제공 데이터의 시간 열이 일/월 파싱 오류로 손상된 것을 확인해 시간 홀드아웃 결과를 철회하고 행 순서 기준 walk-forward로 대체한 것이 대표적입니다 (`src/fix_time.py`).

---

## 시작하기

### 필수 환경
- Python 3.12
- `pip install -r requirements.txt`

### 프로젝트 구조

```
fab-decision-simulator/
├── src/          # 전처리, 모델, Cost Layer, XAI, walk-forward, 시뮬레이터 생성
├── web/          # 시뮬레이터 (simulator.html, sim_final.json)
├── results/      # 확정 결과 JSON
├── figures/      # 정본 그림
├── docs/         # 발표자료, 사전등록 문서
└── data/         # 데이터 준비 안내와 변환 스크립트
```

### 빠른 시작

1. **데이터 준비**
   - UCI SECOM을 내려받아 변환합니다. [데이터 준비 가이드](data/README.md) 참조
   ```bash
   python data/make_secom_csv.py secom.data secom_labels.data
   ```

2. **파이프라인 실행**
   ```bash
   python src/cache_cms.py          # nested CV 혼동행렬 캐시
   python src/seed_tail_metric.py   # 50 seed 캐시
   python src/cost_model_final.py   # Cost Layer
   python src/wf_v2.py              # walk-forward
   ```

3. **시뮬레이터**
   - `web/simulator.html`을 브라우저에서 바로 열면 됩니다
   - 재생성: `python src/build_sim_data.py && python src/build_sim_html.py`

각 스크립트는 `python src/<파일>.py`로 단독 실행되며, 필요한 캐시는 `results/` 아래에 생성됩니다. `src/oof_shap_v2.py`는 사전등록 문서와 원본 CSV의 SHA-256을 검증하므로 UCI에서 재생성한 CSV로는 실행되지 않습니다.

## 참고

- 코드 작성에 AI 코딩 도구를 활용했으며, 문제 정의와 검증 프로토콜 설계, 결과 판정과 정정은 직접 수행했습니다.
- 라이선스: 코드 MIT. 데이터는 UCI Machine Learning Repository의 SECOM과 APS Failure at Scania Trucks (CC BY 4.0).

## 발표자료

- [발표자료 (PDF)](docs/발표자료.pdf)
- [발표자료 (PPTX)](docs/발표자료.pptx)
