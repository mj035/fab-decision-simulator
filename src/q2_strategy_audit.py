"""
q2_strategy_audit.py — 25차 자기검증: "Q2 선별최적 0/20은 퇴화 가드와 독립"이라는 주장이 참인가.

의심: fwd_cost.py는 `eff_strategy`(가드 적용 **후**)를 세고 있다. 그렇다면 0/20이 가드 때문일 수 있고,
     "가드와 독립"이라는 근거 문장은 거짓이 된다. → 원시 argmin(가드 미적용)으로 재판정한다.
동시에 확인: 가드가 걸릴 때의 폴백이 '전수검사'가 맞는가, 아니면 '전수통과'인가.
     (fwd_cost.py의 배포권고 라벨이 '전수검사'로 하드코딩되어 있음 — 검증 필요)
산출: results/q2_strategy_audit.json
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_adapter import adapt
import cost_model as CM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
raw = pd.read_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"))
X, y, _ = adapt(raw.drop(columns=["Time"]), "Pass/Fail", 1); y = np.asarray(y); N = len(y)
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]; Xv = X[valid]
q = np.floor(np.arange(N) / N * 4).astype(int); q[q == 4] = 3
CL, CS, B, A = 15, 0, 0.95, 0.02
THS = np.linspace(0.005, 0.995, 300)
NAMES = ["전수통과", "전수검사", "선별검사"]


def mkpipe(spw, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()),
        ("m", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, scale_pos_weight=spw, random_state=seed,
            n_jobs=-1, verbose=-1))])


def cms(sc, yy):
    return np.array([[int(((sc < t) & (yy == 0)).sum()), int(((sc >= t) & (yy == 0)).sum()),
                      int(((sc < t) & (yy == 1)).sum()), int(((sc >= t) & (yy == 1)).sum())] for t in THS])


out = {}
for k, name in ((0, "Q1~1→Q2"), (1, "Q1~2→Q3"), (2, "Q1~3→Q4")):
    tr, te = np.where(q <= k)[0], np.where(q == k + 1)[0]
    nte = len(te); npos = int(y[te].sum()); nneg = nte - npos
    raws, effs, rows = [], [], []
    for s in range(20):
        spw = (y[tr] == 0).sum() / max(y[tr].sum(), 1)
        in_oof = cross_val_predict(mkpipe(spw, s), Xv.iloc[tr], y[tr],
                                   cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                   method="predict_proba", n_jobs=-1)[:, 1]
        sc = mkpipe(spw, s).fit(Xv.iloc[tr], y[tr]).predict_proba(Xv.iloc[te])[:, 1]
        folds = [(cms(in_oof, y[tr]), cms(sc, y[te]))]
        r = CM.evaluate(folds, nte, npos, nneg, cl=CL, cs=CS, beta=B, alpha=A)
        rawb = CM.CM_raw(folds, nte, npos, nneg, cl=CL, cs=CS, beta=B, alpha=A)
        raws.append(rawb); effs.append(r["eff_strategy"])
        lam = B * CL
        sel, agg, _, _ = CM._accumulate(folds, lam, 0.0, 0.0, CL)
        rows.append({"pass": npos * CL, "inspect": nte + npos * CL - npos * lam, "sel": round(float(sel), 1),
                     "rate": round(r["inspect_rate"] * 100, 1)})
    cnt = lambda a: {NAMES[i]: int(sum(1 for v in a if v == i)) for i in range(3) if sum(1 for v in a if v == i)}
    guard_changed = int(sum(1 for a, b in zip(raws, effs) if a != b))
    m = rows[0]
    out[name] = {"원시_argmin(가드_미적용)": cnt(raws), "가드_적용후": cnt(effs),
                 "가드가_판정을_바꾼_seed": f"{guard_changed}/20",
                 "비용_예시(seed0)": m,
                 "선별이_원시비교에서_이긴_seed": int(sum(1 for v in raws if v == 2))}
    print(f"[{name}] 원시 {cnt(raws)} | 가드후 {cnt(effs)} | 가드가 바꾼 seed {guard_changed}/20")
    print(f"    비용(seed0): 전수통과 {m['pass']:.0f} · 전수검사 {m['inspect']:.0f} · 선별 {m['sel']:.0f} (검사율 {m['rate']}%)")

q2 = out["Q1~1→Q2"]
out["판정"] = (
    f"Q2에서 선별이 **원시 비교**(가드 미적용, 3전략 절대비용 argmin)에서 이긴 seed = "
    f"{q2['선별이_원시비교에서_이긴_seed']}/20. 가드가 판정을 바꾼 seed = {q2['가드가_판정을_바꾼_seed']}. "
    + ("→ ✅ '선별 미채택은 가드와 독립'이라는 근거 문장은 **참**."
       if q2["선별이_원시비교에서_이긴_seed"] == 0 else
       "→ 🔴 근거 문장 **거짓** — 가드가 판정을 만들고 있다. 문장 재작성 필요.")
    + f" 폴백 전략의 실제 정체: {list(q2['원시_argmin(가드_미적용)'].keys())}")
print("\n[판정]", out["판정"])
json.dump(out, open(os.path.join(ROOT, "results", "q2_strategy_audit.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/q2_strategy_audit.json")
