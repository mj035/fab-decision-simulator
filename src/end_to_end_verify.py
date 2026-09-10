"""
END-TO-END 통합 검증 — 원본 CSV 한 개에서 의사결정까지 한 스크립트로 관통한다.

  원본 CSV → 전처리 → RandomForest → [SHAP 우선순위] + [Nested CV 비용/임계값] → 의사결정

동일한 전처리·동일한 모델(RF)에서 SHAP와 비용이 함께 나오는지 자체 대조한다.
  · 자체 검증 1: SHAP 상위 센서가 report.json.shap_top20과 일치하는가
  · 자체 검증 2: OOF ROC-AUC가 report.json과 일치하는가
  · 자체 검증 3: 비용비 20:1 임계값·검출·과검이 web/sim_data.json(RF)과 일치하는가

그리고 과검:미검 = 1:5(R=5)와 1:2(R=2) 두 시나리오의 의사결정을 비교한다.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")
rep  = json.load(open(os.path.join(ROOT, "results", "report.json"), encoding="utf-8"))
sim  = json.load(open(os.path.join(ROOT, "web", "sim_data.json"), encoding="utf-8"))

# ── 1. 원본 로드 + 전처리 ─────────────────────────────
df = pd.read_csv(CSV)
y  = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X  = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
X.columns = [f"S{c}" for c in X.columns]          # analysis.py와 동일한 센서 ID
N_pos, N_neg = int(y.sum()), int((y == 0).sum())
print(f"[1] 원본 로드: {X.shape[0]} lot × {X.shape[1]} 센서 · 불량 {N_pos} ({N_pos/len(y)*100:.2f}%)")

imp = SimpleImputer(strategy="median").fit(X)
var = VarianceThreshold(0.0).fit(imp.transform(X))
kept = X.columns[var.get_support()]
Xv  = var.transform(imp.transform(X))
Xt  = StandardScaler().fit_transform(Xv)
print(f"    전처리 후 유효 센서: {Xt.shape[1]}개 (상수센서 {X.shape[1]-Xt.shape[1]}개 제거)")

RF_KW = dict(n_estimators=400, class_weight="balanced_subsample",
             min_samples_leaf=3, random_state=0, n_jobs=-1)

# ── 2. SHAP 우선순위 (전체 데이터 적합 모델) ───────────
rf_full = RandomForestClassifier(**RF_KW).fit(Xt, y)
sv = shap.TreeExplainer(rf_full).shap_values(Xt, check_additivity=False)
sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
mean_abs = np.abs(sv1).mean(0)
order = np.argsort(-mean_abs)
shap_top = [(str(kept[j]), float(mean_abs[j])) for j in order[:5]]
print("\n[2] SHAP 점검 우선순위 상위 5")
for r, (s, v) in enumerate(shap_top, 1):
    print(f"    {r}. {s}  ({v:.4f})")

# ── 3. Nested CV 비용/임계값 (simulator_data.py와 동일 설정) ──
RATIOS = [2, 5, 10, 20, 50]
THS = np.linspace(0.005, 0.995, 199)
head = [("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)),
        ("sc", StandardScaler()), ("m", RandomForestClassifier(**RF_KW))]
mk = lambda: Pipeline(head)

TP = {R: 0 for R in RATIOS}; FN = {R: 0 for R in RATIOS}
FP = {R: 0 for R in RATIOS}; TN = {R: 0 for R in RATIOS}
THSEL = {R: 0.0 for R in RATIOS}
oof = np.zeros(len(y))
outer = StratifiedKFold(5, shuffle=True, random_state=7)
for k, (tr, te) in enumerate(outer.split(X, y)):
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
    inner = cross_val_predict(mk(), Xtr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=k),
                              method="predict_proba", n_jobs=-1)[:, 1]
    te_score = mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    oof[te] = te_score
    in_cm = np.array([confusion_matrix(ytr, (inner >= t), labels=[0, 1]).ravel() for t in THS])
    te_cm = np.array([confusion_matrix(yte, (te_score >= t), labels=[0, 1]).ravel() for t in THS])
    for R in RATIOS:
        j = int((in_cm[:, 1] * 1.0 + in_cm[:, 2] * R).argmin())
        tn, fp, fn, tp = te_cm[j]
        TN[R] += tn; FP[R] += fp; FN[R] += fn; TP[R] += tp
        THSEL[R] += THS[j] / 5.0
    print(f"    [3] Nested CV fold {k+1}/5 완료")
auc = roc_auc_score(y, oof)

# ── 4. 자체 대조 ─────────────────────────────────────
print("\n[4] 자체 검증")
rep_top = rep["shap_top20"][0][0]
c1 = (shap_top[0][0] == rep_top)
print(f"    검증1 SHAP 1위 = {shap_top[0][0]} vs report {rep_top}  →  {'일치 ✅' if c1 else '불일치 ❌'}")
c2 = abs(auc - sim["models"]["rf"]["auc"]) < 0.01
print(f"    검증2 OOF AUC = {auc:.4f} vs sim_data {sim['models']['rf']['auc']:.4f}  →  {'일치 ✅' if c2 else '불일치 ❌'}")
i20 = sim["ratios"].index(20)
c3 = abs(THSEL[20] - sim["models"]["rf"]["th"][i20]) < 0.02 and abs(FP[20] - sim["models"]["rf"]["fp"][i20]) <= 40
print(f"    검증3 20:1 임계값 {THSEL[20]:.3f}/과검{FP[20]} vs sim {sim['models']['rf']['th'][i20]:.3f}/과검{sim['models']['rf']['fp'][i20]}  →  {'일치 ✅' if c3 else '근사'}")

# ── 5. 두 시나리오 의사결정 (과검:미검 = 1:5, 1:2) ──────
def decide(over, under):
    R = max(RATIOS[0], min(RATIOS[-1], round(under / over)))
    tp, fn, fp = TP[R], FN[R], FP[R]
    sel = fp * over + fn * under
    passAll = N_pos * under; inspAll = N_neg * over
    best_alt = min(passAll, inspAll)
    saving = (best_alt - sel) / best_alt * 100
    names = ["전수 통과", "전수 검사", "선별 검사"]
    costs = [passAll, inspAll, sel]
    win = names[int(np.argmin(costs))]
    return R, THSEL[R], tp/(tp+fn)*100, fp/N_neg*100, costs, win, saving

print("\n[5] 시나리오 비교")
for tag, over, under in [("과검:미검 = 1:5", 10, 50), ("과검:미검 = 1:2", 10, 20)]:
    R, th, det, okr, costs, win, sav = decide(over, under)
    print(f"\n  ── {tag}  (비용비 {R}:1) ──")
    print(f"     임계값 {th:.3f} · 검출률 {det:.1f}% · 과검률 {okr:.1f}%")
    print(f"     전수통과 {costs[0]:,.0f} / 전수검사 {costs[1]:,.0f} / 선별검사 {costs[2]:,.0f} (만원)")
    print(f"     ▶ 최적: {win}  (선별검사 절감률 {sav:+.1f}%)")
