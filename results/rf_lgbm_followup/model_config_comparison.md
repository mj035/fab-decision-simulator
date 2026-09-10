# 모델 설정 비교표 — RF 귀속 확정 근거 (준비 단계 산출물)

> 사전등록_RF_vs_LGBM_후속.md §5·§6의 부속 표. 실행 전 작성. 결과 없음.

## 1. 네 가지 설정의 대조

| 항목 | ① 기존 seed_flip 일반 RF | ② model_cmp pca_rf | ③ 신규 rf_no_pca | ④ 최신 base_lgbm |
|---|---|---|---|---|
| 출처 | `src/seed_flip.py:40-41` | `src/model_cmp.py:104-106` | 본 실험 (①을 복원) | `src/model_cmp.py:101-103` |
| 모델 | RandomForestClassifier | RandomForestClassifier | RandomForestClassifier | LGBMClassifier |
| n_estimators | **400** | 500 | **400** | 300 |
| min_samples_leaf | **3** | 2 | **3** | (min_child_samples=25) |
| class_weight | **balanced_subsample** | balanced | **balanced_subsample** | (scale_pos_weight=Nneg/Npos) |
| max_features | sqrt(기본) | sqrt(명시) | sqrt(**명시 고정**) | (colsample_bytree=0.3) |
| PCA | **없음** | **95% (fold 내 fit)** | **없음** | 없음 |
| SMOTE/ADASYN | 없음 | 없음 | 없음 | 없음 |
| 모델 random_state | **0 고정** (seed=CV 분할만) | seed 연동 | **0 고정** (① 원문 복원) | seed 연동 (④ 원문 복원) |
| 전처리 | 대치→VarThr→표준화 (Pipeline, **outer train 전체 fit — 구 프로토콜**) | 대치→VarThr→표준화 (**fold 내부 fit**) | ②와 동일 (**fold 내부 fit**) | ②와 동일 |
| 평가 | cross_val_score 평균, 20 seed | fold-mean AUC + rank/q 비용, 50 seed paired | ②와 동일 프로토콜 | ②와 동일 프로토콜 |

## 2. 사용/비사용 결정과 이유

| 설정 | 사용? | 이유 |
|---|:--:|---|
| ① seed_flip RF 하이퍼파라미터 | ✅ **rf_no_pca로 복원** | 확인 대상인 +0.023 우위를 실제로 낸 구성. 다른 구성을 쓰면 "선행 우위의 확인"이 아니게 됨 (책임자 확정) |
| ① seed_flip의 **평가 프로토콜** | ❌ | outer train 전체 전처리 fit(누수 소지)·pooled 평가·20 seed — 구 프로토콜. 프로토콜은 최신 model_cmp 것을 사용 |
| ② pca_rf 구성 (500/leaf2/balanced/PCA95) | ❌ **혼입 금지** | 다른 모델. 이 구성의 열세(ΔAUC −0.0821)는 일반 RF에 대한 진술이 아님. 별도 arm 동시 비교도 하지 않음 (책임자 확정) |
| ③ rf_no_pca | ✅ 신규 arm | ①의 하이퍼파라미터 + ④와 동일한 최신 프로토콜 |
| ④ base_lgbm | ✅ 기준선 arm | 최신 정본 LGBM. model_cmp에서 fold평균 0.7285로 정본 0.731±0.014 내 재현 확인됨 |

## 3. 핵심 구분 재확인

- **①과 ②는 다른 모델이다.** 기존 +0.023 우위(19/20승)는 ①의 결과이고, model_cmp의 열세는 ②(PCA 통과 RF)의 결과다. 서로를 부정하지 않는다 (`results/model_cmp/결론_범위한정.md` §2②).
- ③의 sklearn 1.9.0 기본값 의존 항목(criterion=gini, max_depth=None, max_features=sqrt, min_samples_split=2, bootstrap=True, max_leaf_nodes=None, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, oob_score=False, ccp_alpha=0.0, max_samples=None)은 실행 코드에 **전부 명시 기입**해 버전 독립으로 고정했다.
- 모델 random_state 비대칭(③=0 고정 vs ④=seed 연동)은 각 원문의 문자 그대로 복원한 결과이며 사전등록 §5에 명시·결과 보고 시 병기한다. paired 설계(동일 fold)는 영향받지 않는다.

## 4. 철회된 주장 미사용 확인

다음은 본 실험의 어떤 문서·코드에도 사용하지 않았다:
- ~~"RF와 LGBM 차이는 통계적으로 무의미"~~ (report.json `ceiling_proof` — 비대응 분산 오류, 자기정정 #5로 철회)
- ~~비대응 분산을 paired 노이즈로 사용~~ (본 실험은 대응 CI만 사용, thresholds 원문 주의사항)
- ~~"AUC 0.75 절대 천장"~~
- ~~"LightGBM이 모든 모델 중 최고"~~ (결론_범위한정 §2① 금지)
