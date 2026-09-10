"""
웹 시뮬레이터가 사용할 실측 데이터를 생성한다.  →  web/sim_data.json

⚠️ 합성·가짜 데이터 없음. 전부 SECOM 실측 1,567 lot에서 Nested CV로 계산한다.

핵심 설계:
  비용비(2:1~60:1)마다 Nested CV로 임계값을 '안쪽 fold에서만' 고르고, 바깥 fold에서 평가하여
  혼동행렬(tp, fn, fp, tn)을 누적한다. 이 개수만 저장하면, 브라우저에서 사용자가 입력한
  실제 금액(과검 비용, 미검 비용)을 곱해 세 전략의 비용을 그 자리에서 계산할 수 있다.

  → 낙관 편향이 제거된 상태로 저장되므로, 시뮬레이터가 보여주는 절감률은
    보고서의 37.9% / −7.0%와 정확히 일치한다.

RF와 LightGBM 두 모델 모두 계산한다 ("AUC 1등 ≠ 비용 1등"을 UI에서 직접 보이기 위함).
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "fab_process_yield.csv")

df = pd.read_csv(CSV)
y = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
N_pos, N_neg = int(y.sum()), int((y == 0).sum())

RATIOS = np.arange(2, 61)
THS = np.linspace(0.005, 0.995, 199)

head = lambda m: [("imp", SimpleImputer(strategy="median")),
                  ("var", VarianceThreshold(0.0)),
                  ("sc", StandardScaler()), ("m", m)]

MODELS = {
    "rf": lambda: Pipeline(head(RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample",
        min_samples_leaf=3, random_state=0, n_jobs=-1))),
    "lgbm": lambda: Pipeline(head(LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
        scale_pos_weight=N_neg / N_pos, random_state=0, n_jobs=-1, verbose=-1))),
}

out = {
    "meta": {
        "lot": int(len(y)), "sensor": int(X.shape[1]),
        "불량": N_pos, "정상": N_neg,
        "불량률(%)": round(N_pos / len(y) * 100, 2),
        "설명": "Nested CV(임계값은 안쪽 fold에서만 선택) 기준. 낙관 편향 제거됨. 합성 데이터 없음.",
    },
    "ratios": [int(r) for r in RATIOS],
    "models": {},
}

for key, mk in MODELS.items():
    print(f"\n=== {key.upper()} Nested CV ===")
    outer = StratifiedKFold(5, shuffle=True, random_state=7)
    # 비용비별 누적 혼동행렬
    TP = np.zeros(len(RATIOS), int); FN = np.zeros(len(RATIOS), int)
    FP = np.zeros(len(RATIOS), int); TN = np.zeros(len(RATIOS), int)
    THSEL = np.zeros(len(RATIOS))          # fold별 선택 임계값의 평균
    oof = np.zeros(len(y))

    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]

        inner = cross_val_predict(mk(), Xtr, ytr,
                                  cv=StratifiedKFold(4, shuffle=True, random_state=k),
                                  method="predict_proba", n_jobs=-1)[:, 1]
        te_score = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        oof[te] = te_score

        in_cm = np.array([confusion_matrix(ytr, (inner >= t).astype(int), labels=[0, 1]).ravel()
                          for t in THS])                      # tn, fp, fn, tp
        te_cm = np.array([confusion_matrix(yte, (te_score >= t).astype(int), labels=[0, 1]).ravel()
                          for t in THS])

        for i, R in enumerate(RATIOS):
            in_cost = in_cm[:, 1] * 1.0 + in_cm[:, 2] * float(R)   # fp*1 + fn*R
            j = int(in_cost.argmin())                              # 안쪽에서만 임계값 선택
            tn, fp, fn, tp = te_cm[j]
            TN[i] += tn; FP[i] += fp; FN[i] += fn; TP[i] += tp
            THSEL[i] += THS[j] / 5.0
        print(f"  fold {k+1}/5 완료")

    auc = float(roc_auc_score(y, oof))
    print(f"  OOF ROC-AUC = {auc:.4f}")

    out["models"][key] = {
        "name": "RandomForest" if key == "rf" else "LightGBM",
        "auc": round(auc, 4),
        "th": [round(float(t), 3) for t in THSEL],
        "tp": TP.tolist(), "fn": FN.tolist(),
        "fp": FP.tolist(), "tn": TN.tolist(),
    }

os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
p = os.path.join(ROOT, "web", "sim_data.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\n→ {p}")

# 검증: 보고서 수치(14:1 → 37.9%, 50:1 → −7.0%)와 일치하는지 확인
C_FP = 10.0
for R in [5, 10, 14, 20, 30, 43, 50]:
    i = int(np.where(RATIOS == R)[0][0])
    m = out["models"]["rf"]
    sel = m["fp"][i] * C_FP + m["fn"][i] * R * C_FP
    passall = N_pos * R * C_FP
    inspect = N_neg * C_FP
    best = min(passall, inspect)
    sav = (best - sel) / best * 100
    win = ["전수 통과", "전수 검사", "선별 검사"][int(np.argmin([passall, inspect, sel]))]
    print(f"  {R:>2}:1 | 절감 {sav:>6.1f}% | 최선: {win}")
