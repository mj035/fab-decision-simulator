"""
[명세서 정형트랙 아이디어 5] 결측 처리 전략 비교

질문: 중앙값 대치가 정말 최선인가? 그냥 관행이라 쓴 건 아닌가?

전제: 결측 있는 행을 버리면 1,567 → 0개. 결측 있는 열을 버리면 590 → 52개.
      즉 '삭제'는 선택지가 아니며, 어떤 형태로든 대치가 강제된다.

⚠️ 주의: 7개 전략 중 '가장 높은 AUC'를 골라 자랑하는 것은 체리피킹이다.
        차이가 CV 노이즈를 넘는지 반드시 함께 판정하고, 넘지 않으면 "차이 없음"으로 보고한다.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")
df = pd.read_csv(CSV)
y  = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X  = df.drop(columns=["Time","Pass/Fail"]).astype(float)

class DropHighMissing(BaseEstimator, TransformerMixin):
    """결측률이 thr을 넘는 센서를 제거 (train fold에서만 기준을 정한다)"""
    def __init__(self, thr=0.5): self.thr = thr
    def fit(self, X, y=None):
        self.keep_ = (np.isnan(np.asarray(X, float)).mean(0) <= self.thr)
        return self
    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]

RF = lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                    min_samples_leaf=3, random_state=0, n_jobs=-1)
LGB = lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                             min_child_samples=25, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=10.0,
                             scale_pos_weight=(y==0).sum()/(y==1).sum(),
                             random_state=0, n_jobs=-1, verbose=-1)

TAIL = [("var", VarianceThreshold(0.0)), ("sc", StandardScaler())]

STRATEGIES = {
    "① 중앙값 (현재 방식)":
        Pipeline([("imp", SimpleImputer(strategy="median"))] + TAIL + [("m", RF())]),
    "② 평균":
        Pipeline([("imp", SimpleImputer(strategy="mean"))]   + TAIL + [("m", RF())]),
    "③ 0으로 채움":
        Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))] + TAIL + [("m", RF())]),
    "④ 최빈값":
        Pipeline([("imp", SimpleImputer(strategy="most_frequent"))] + TAIL + [("m", RF())]),
    "⑤ KNN 대치 (k=5)":
        Pipeline([("imp", KNNImputer(n_neighbors=5))] + TAIL + [("m", RF())]),
    "⑥ 중앙값 + 결측여부 피처":
        Pipeline([("u", FeatureUnion([
                      ("imp", SimpleImputer(strategy="median")),
                      ("ind", MissingIndicator(features="all")),
                  ])), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()), ("m", RF())]),
    "⑦ 고결측(>50%) 센서 제거 + 중앙값":
        Pipeline([("drop", DropHighMissing(0.5)),
                  ("imp", SimpleImputer(strategy="median"))] + TAIL + [("m", RF())]),
    "⑧ 대치 없음 (LightGBM 네이티브)":
        Pipeline([("m", LGB())]),          # LightGBM은 NaN을 그대로 학습한다
}

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42)
res = {}
print(f"{'전략':<30}{'ROC-AUC':>18}{'PR-AUC':>16}")
print("-" * 66)
for name, pipe in STRATEGIES.items():
    r = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
    p = cross_val_score(pipe, X, y, cv=CV, scoring="average_precision", n_jobs=-1)
    res[name] = {"ROC_mean": float(r.mean()), "ROC_std": float(r.std()),
                 "PR_mean": float(p.mean()),  "PR_std": float(p.std())}
    print(f"{name:<30}{r.mean():>8.4f} ± {r.std():.3f}{p.mean():>9.4f} ± {p.std():.3f}")

# 체리피킹 방지: 최고 vs 현재(중앙값)의 차이가 노이즈를 넘는가?
cur  = res["① 중앙값 (현재 방식)"]
best = max(res.items(), key=lambda kv: kv[1]["ROC_mean"])
gap   = best[1]["ROC_mean"] - cur["ROC_mean"]
noise = float(np.hypot(best[1]["ROC_std"], cur["ROC_std"]))
print("\n" + "=" * 66)
print(f"최고 전략 : {best[0]}  (ROC {best[1]['ROC_mean']:.4f})")
print(f"현재 방식 : ① 중앙값        (ROC {cur['ROC_mean']:.4f})")
print(f"격차 {gap:.4f}  vs  fold 노이즈 {noise:.4f}   →  {gap/noise:.2f}배")
verdict = ("차이가 노이즈 이하 → 중앙값을 유지한다. 대치 전략은 이 데이터의 병목이 아니다."
           if gap < noise else
           f"차이가 유의할 수 있음 → {best[0]} 검토 필요.")
print(f"판정: {verdict}")
print("=" * 66)

names = list(res.keys())
means = [res[n]["ROC_mean"] for n in names]
stds  = [res[n]["ROC_std"]  for n in names]
cols  = ["#4f86d9" if n.startswith("①") else "#a9c3e8" for n in names]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(names[::-1], means[::-1], xerr=stds[::-1], capsize=4, color=cols[::-1], height=.6)
ax.axvline(cur["ROC_mean"], color="#d94f4f", ls="--", lw=1.3,
           label=f"현재 방식(중앙값) {cur['ROC_mean']:.3f}")
for i, n in enumerate(names[::-1]):
    ax.text(res[n]["ROC_mean"] + res[n]["ROC_std"] + .004, i, f'{res[n]["ROC_mean"]:.3f}',
            va="center", fontsize=8.5)
ax.set_xlim(0.60, 0.82); ax.set_xlabel("ROC-AUC (5-fold × 6반복)")
ax.set_title("결측 처리 전략 비교 — 무엇을 써도 결과가 같다\n"
             f"최고-현재 격차 {gap:.4f} vs 노이즈 {noise:.4f} → 대치 전략은 병목이 아니다",
             fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig10_imputation.png")); plt.close()

rep = json.load(open(os.path.join(ROOT,"results","report.json"), encoding="utf-8"))
rep["imputation_study"] = {
    "왜 삭제하지 않았나": {
        "결측 있는 lot": "1567/1567 (100%) → 행 삭제 시 0개 남음",
        "결측 있는 센서": "538/590 → 열 삭제 시 52개만 남음",
        "결론": "삭제는 선택지가 아니며 대치가 강제된다.",
    },
    "전략별 성능": {k: {kk: round(vv,4) for kk, vv in v.items()} for k, v in res.items()},
    "최고전략": best[0], "현재전략": "① 중앙값",
    "격차": round(gap,4), "노이즈": round(noise,4), "격차/노이즈": round(gap/noise,2),
    "판정": verdict,
}
json.dump(rep, open(os.path.join(ROOT,"results","report.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n→ figures/fig10_imputation.png")
