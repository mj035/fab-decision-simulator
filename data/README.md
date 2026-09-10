# 데이터 준비

저장소에는 데이터를 포함하지 않는다. 두 데이터셋 모두 UCI Machine Learning Repository에서 내려받는다.

## 1. SECOM (주 데이터)

- 출처: https://archive.ics.uci.edu/dataset/179/secom (McCann, Johnston. CC BY 4.0)
- 파일: `secom.data`, `secom_labels.data`
- 변환:

```bash
python data/make_secom_csv.py secom.data secom_labels.data
```

`data/fab_process_yield.csv`가 생성된다. 열 구성은 `Time`, 센서 `0`~`589`, `Pass/Fail`(불량 +1, 정상 −1)이다.
대회에서 제공된 원본 CSV와 시간 열 표기가 다를 수 있으며, 분석은 시간 열 대신 행 순서(생산 순서)를 시간축으로 쓰므로 결과에 영향이 없다.
단, `src/oof_shap_v2.py`는 대회 제공 원본 CSV의 SHA-256을 검증하므로 재생성한 CSV로는 실행되지 않는다.

## 2. APS Failure at Scania Trucks (일반화 검증)

- 출처: https://archive.ics.uci.edu/dataset/421 (CC BY 4.0)
- `aps_failure_training_set.csv`를 `data_aps/`에 넣은 뒤 `python src/aps_fetch.py`를 실행하면 `aps_X.csv`, `aps_y.csv`가 생성된다.
