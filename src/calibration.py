"""
확률 캘리브레이션 검증.

질문: 모델이 "불량 확률 8%"라고 말할 때, 실제로 그 그룹의 8%가 불량인가?
   - Brier Score = 예측확률과 정답(0/1)의 MSE. 낮을수록 좋음. (MSE의 분류판)
   - Reliability Diagram = 예측확률 구간별 실제 불량률. 대각선이면 정직한 확률.
   - RandomForest는 캘리브레이션이 나쁜 것으로 알려져 있으므로 반드시 확인해야 한다.

주의: 임계값 최적화 자체는 실제 FP/FN을 세서 구했으므로 캘리브레이션과 무관하게 유효하다.
      캘리브레이션은 (a) 임계값의 '해석', (b) 본선에서 타 데이터셋과의 비교에 필요하다.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")

df = pd.read_csv(CSV)
y  = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X  = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def base(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("m",   model)])

RF = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                            min_samples_leaf=3, random_state=0, n_jobs=-1)

# ① 원본 RF (현재 우리가 쓰는 것)
oof_raw = cross_val_predict(base(RF), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

# ② 확률 보정된 RF (Isotonic / Platt)
oof_iso = cross_val_predict(
    base(CalibratedClassifierCV(RF, method="isotonic", cv=3)),
    X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
oof_sig = cross_val_predict(
    base(CalibratedClassifierCV(RF, method="sigmoid", cv=3)),
    X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

# 기준선: 항상 유병률(6.64%)을 뱉는 모델
oof_prior = np.full_like(oof_raw, y.mean())

rows = {}
for name, p in [("항상 6.64% (기준선)", oof_prior), ("RandomForest (원본)", oof_raw),
                ("RF + Isotonic 보정", oof_iso), ("RF + Platt(Sigmoid) 보정", oof_sig)]:
    rows[name] = {
        "Brier": round(float(brier_score_loss(y, p)), 5),
        "ROC_AUC": round(float(roc_auc_score(y, p)), 4) if p.std() > 0 else 0.5,
        "PR_AUC": round(float(average_precision_score(y, p)), 4),
        "평균예측확률": round(float(p.mean()), 4),
    }

print(f"{'모델':<26}{'Brier↓':>10}{'ROC-AUC↑':>11}{'PR-AUC↑':>10}{'평균확률':>10}   (실제 불량률 0.0664)")
print("-" * 78)
for k, v in rows.items():
    print(f"{k:<26}{v['Brier']:>10.5f}{v['ROC_AUC']:>11.4f}{v['PR_AUC']:>10.4f}{v['평균예측확률']:>10.4f}")

# 신뢰도 곡선
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].plot([0, 0.5], [0, 0.5], "k--", lw=1, label="완벽한 캘리브레이션")
for name, p, c in [("RandomForest (원본)", oof_raw, "#d94f4f"),
                   ("RF + Isotonic 보정", oof_iso, "#4f86d9"),
                   ("RF + Platt 보정",    oof_sig, "#3fa06b")]:
    try:
        pt, pp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax[0].plot(pp, pt, marker="o", ms=4, lw=1.6, color=c,
                   label=f"{name} (Brier={brier_score_loss(y,p):.4f})")
    except Exception as e:
        print("skip", name, e)
ax[0].set_xlabel("모델이 말한 불량 확률"); ax[0].set_ylabel("실제 불량 비율")
ax[0].set_title("신뢰도 곡선(Reliability Diagram)\n대각선에 가까울수록 '정직한 확률'", fontsize=10)
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)

ax[1].hist(oof_raw, bins=40, alpha=.65, color="#d94f4f", label="RandomForest (원본)")
ax[1].hist(oof_iso, bins=40, alpha=.65, color="#4f86d9", label="RF + Isotonic 보정")
ax[1].axvline(y.mean(), color="k", ls="--", lw=1.2, label=f"실제 불량률 {y.mean():.4f}")
ax[1].set_xlabel("예측 불량 확률"); ax[1].set_ylabel("lot 수"); ax[1].set_yscale("log")
ax[1].set_title("예측 확률 분포", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig7_calibration.png")); plt.close()

rep = json.load(open(os.path.join(ROOT, "results", "report.json"), encoding="utf-8"))
rep["calibration"] = {
    "설명": "Brier Score = 예측확률과 정답의 MSE(분류판). 낮을수록 확률이 정직함.",
    "결과": rows,
    "주의": "임계값 최적화는 실제 FP/FN을 세어 구했으므로 캘리브레이션과 무관하게 유효. "
            "캘리브레이션은 임계값의 '해석'과 본선 타 데이터셋 비교에 필요함.",
}
json.dump(rep, open(os.path.join(ROOT, "results", "report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n→ figures/fig7_calibration.png, report.json 갱신")
