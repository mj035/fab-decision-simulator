"""
END-TO-END (LightGBM 판) — 최종 배포 모델 LGBM으로 원본→의사결정 관통.
RF판(end_to_end_verify.py)과 SHAP 우선순위·의사결정이 일치하는지 비교한다.
'모델 선택은 저비용 결정'이라는 명제의 실증.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")
sim  = json.load(open(os.path.join(ROOT, "web", "sim_data.json"), encoding="utf-8"))

df = pd.read_csv(CSV)
y  = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X  = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
X.columns = [f"S{c}" for c in X.columns]
N_pos, N_neg = int(y.sum()), int((y == 0).sum())
SPW = N_neg / N_pos

LGB_KW = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=SPW, random_state=0, n_jobs=-1, verbose=-1)

# 1. 전처리 + SHAP (LGBM 전체 적합)
imp = SimpleImputer(strategy="median").fit(X)
var = VarianceThreshold(0.0).fit(imp.transform(X))
kept = X.columns[var.get_support()]
Xt = StandardScaler().fit_transform(var.transform(imp.transform(X)))
lgb_full = LGBMClassifier(**LGB_KW).fit(Xt, y)
sv = shap.TreeExplainer(lgb_full).shap_values(Xt)
sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
mean_abs = np.abs(sv1).mean(0)
shap_top = [str(kept[j]) for j in np.argsort(-mean_abs)[:5]]
print("[LGBM] SHAP 상위 5:", ", ".join(shap_top))

# 2. Nested CV 비용/임계값
RATIOS = [2, 5, 10, 20, 50]
THS = np.linspace(0.005, 0.995, 199)
mk = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)),
                       ("sc", StandardScaler()), ("m", LGBMClassifier(**LGB_KW))])
TP = {R: 0 for R in RATIOS}; FN = {R: 0 for R in RATIOS}
FP = {R: 0 for R in RATIOS}; TN = {R: 0 for R in RATIOS}; THSEL = {R: 0.0 for R in RATIOS}
oof = np.zeros(len(y))
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y)):
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
    inner = cross_val_predict(mk(), Xtr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=k),
                              method="predict_proba", n_jobs=-1)[:, 1]
    te_score = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    oof[te] = te_score
    in_cm = np.array([confusion_matrix(ytr, (inner >= t), labels=[0, 1]).ravel() for t in THS])
    te_cm = np.array([confusion_matrix(yte, (te_score >= t), labels=[0, 1]).ravel() for t in THS])
    for R in RATIOS:
        j = int((in_cm[:, 1] + in_cm[:, 2] * R).argmin())
        tn, fp, fn, tp = te_cm[j]
        TN[R] += tn; FP[R] += fp; FN[R] += fn; TP[R] += tp; THSEL[R] += THS[j] / 5.0
print(f"[LGBM] OOF AUC {roc_auc_score(y, oof):.4f} (sim {sim['models']['lgbm']['auc']:.4f})")

# 3. 두 시나리오
def decide(over, under):
    R = max(RATIOS[0], min(RATIOS[-1], round(under / over)))
    sel = FP[R]*over + FN[R]*under
    costs = [N_pos*under, N_neg*over, sel]
    win = ["전수 통과", "전수 검사", "선별 검사"][int(np.argmin(costs))]
    sav = (min(costs[0], costs[1]) - sel) / min(costs[0], costs[1]) * 100
    return R, THSEL[R], TP[R]/(TP[R]+FN[R])*100, FP[R]/N_neg*100, win, sav

for tag, o, u in [("과검:미검 1:5", 10, 50), ("과검:미검 1:2", 10, 20)]:
    R, th, det, okr, win, sav = decide(o, u)
    print(f"  {tag} ({R}:1) → 임계값 {th:.3f}·검출 {det:.1f}%·과검 {okr:.1f}% → {win} ({sav:+.1f}%)")
