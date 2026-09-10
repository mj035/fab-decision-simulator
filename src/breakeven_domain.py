"""
breakeven_domain.py — 리뷰어 25차 (B): 새 손익분기 규칙의 **격자 상단 의존성** 점검.

지적: "이 cl 이상 모든 cl에서 CI 하단>0"은 하단 의존은 없앴지만 상단 의존을 만들었다.
     고cl에서 두 전략이 모두 전수통과로 수렴하면 차이가 줄어 CI가 0을 다시 포함할 수 있고,
     그러면 "모든 cl" 조건이 깨져 손익분기가 미정의가 된다.
확인: cl 2~60 전 구간의 대응차 CI 하단을 보고, 16 이후 유지가 깨지는 지점이 있는지.
     (i) 전 구간 유지 → 규칙 그대로  (ii) 깨짐 → 정의역 명문화

+ 리뷰어 25차 (C): cl=9 판정(경계가 8인가 9인가) + cl=8에서 센서가 이기는 기제.
산출: results/breakeven_domain.json
"""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
NSEED = 50
CLS = list(range(2, 61))          # ← 상단을 24 → 60으로 확장

X, y, _ = adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                         "Pass/Fail", 1, drop_cols=["Time"])
y = np.asarray(y); N = len(y); npos = int(y.sum()); nneg = N - npos
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xi = X[valid].fillna(X[valid].median()).to_numpy()
mfiles = {int(f.split('_s')[-1].split('.')[0]): f for f in glob.glob(os.path.join(ROOT, 'results', 'cms_seeds', 'SECOM_s*.npz'))}


def auc_cols(idx):
    sub = Xi[idx]; yy = y[idx]; p = yy == 1; np_ = p.sum(); nn_ = len(idx) - np_
    return ((sub.argsort(0).argsort(0) + 1)[p].sum(0) - np_ * (np_ + 1) / 2) / (np_ * nn_)


def sensor_folds(seed):
    folds = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        a = auc_cols(tr); j = int(np.argmax(np.maximum(a, 1 - a))); sign = 1.0 if a[j] >= 0.5 else -1.0
        s_tr, s_te = Xi[tr, j] * sign, Xi[te, j] * sign
        ths = np.quantile(s_tr, np.linspace(0.01, 0.99, 300))
        cm = lambda s, idx: np.array([[int(((s < t) & (y[idx] == 0)).sum()), int(((s >= t) & (y[idx] == 0)).sum()),
                                       int(((s < t) & (y[idx] == 1)).sum()), int(((s >= t) & (y[idx] == 1)).sum())] for t in ths])
        folds.append((cm(s_tr, tr), cm(s_te, te)))
    return folds


def best_cost(folds, cl):
    lam, mu, rho = CM.to_effective(cl, 0, B, A)
    sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
    pass_all = npos * cl; inspect_all = N + npos * cl - npos * lam + nneg * mu
    rate = (agg[3] + agg[1]) / N
    sel_eff = inspect_all if rate >= CM.DEGEN_RATE else sel
    costs = [pass_all, inspect_all, sel_eff]
    return min(costs), int(np.argmin(costs)), rate, agg


print(f"[precompute] {NSEED} seed × cl {CLS[0]}~{CLS[-1]} ...")
diff = np.zeros((NSEED, len(CLS)))
m_strat = np.zeros((NSEED, len(CLS)), int); s_strat = np.zeros((NSEED, len(CLS)), int)
m_rate = np.zeros((NSEED, len(CLS))); s_rate = np.zeros((NSEED, len(CLS)))
m_agg = {}; s_agg = {}
for s in range(NSEED):
    d = np.load(mfiles[s]); mf = list(zip(d["in_cms"], d["te_cms"]))
    sf = sensor_folds(s)
    for i, cl in enumerate(CLS):
        cm_, ms, mr, ma = best_cost(mf, cl); cs_, ss, sr, sa = best_cost(sf, cl)
        diff[s, i] = cs_ - cm_
        m_strat[s, i], s_strat[s, i] = ms, ss
        m_rate[s, i], s_rate[s, i] = mr, sr
        if cl in (8, 9, 15, 16):
            m_agg.setdefault(cl, []).append(ma); s_agg.setdefault(cl, []).append(sa)

mean = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(NSEED)
lo, hi = mean - 1.96 * se, mean + 1.96 * se
sig_model = lo > 0
sig_sensor = hi < 0

# ── (B) 상단 의존성: 16 이후 유지가 깨지는가 ──
i16 = CLS.index(16)
breaks = [CLS[i] for i in range(i16, len(CLS)) if not sig_model[i]]
print("\n[B] cl≥16 구간에서 CI 하단>0이 깨지는 cl:", breaks if breaks else "없음 (전 구간 유지)")
print("  cl 24~60 CI 하단 표본:", {cl: round(float(lo[CLS.index(cl)]), 1) for cl in (24, 30, 36, 42, 48, 54, 60)})

if not breaks:
    rule = {"판정": "(i) 전 구간 유지 — 규칙 그대로 사용 가능. 상단 의존성 없음.",
            "정의역": f"cl ∈ [{CLS[0]}, {CLS[-1]}]에서 확인됨", "깨진_cl": []}
else:
    rule = {"판정": f"(ii) cl={breaks[0]}에서 깨짐 — **규칙에 정의역 명문화 필요**.",
            "정의역": f"손익분기 16은 cl ∈ [2, {breaks[0]-1}] 정의역에서만 유효",
            "깨진_cl": breaks[:20]}
print("  →", rule["판정"])

# ── (C) cl=8·9 판정과 기제 ──
def verdict(i):
    return "모델 유의우세" if sig_model[i] else "센서 유의우세" if sig_sensor[i] else "대등"

zone = {}
for cl in range(2, 25):
    i = CLS.index(cl)
    zone[str(cl)] = {"평균차": round(float(mean[i]), 1), "ci95": [round(float(lo[i]), 1), round(float(hi[i]), 1)],
                     "판정": verdict(i), "차이0_seed": int((diff[:, i] == 0).sum()),
                     "모델_검사율": round(float(m_rate[:, i].mean()) * 100, 1),
                     "센서_검사율": round(float(s_rate[:, i].mean()) * 100, 1),
                     "모델전략": int(np.bincount(m_strat[:, i], minlength=3).argmax()),
                     "센서전략": int(np.bincount(s_strat[:, i], minlength=3).argmax())}
print("\n[C] 구간 판정 (전략 0=전수통과 1=전수검사 2=선별):")
for cl in range(5, 18):
    z = zone[str(cl)]
    print(f"  cl={cl:2d} {z['평균차']:+8.1f} CI[{z['ci95'][0]:+7.1f},{z['ci95'][1]:+7.1f}] {z['판정']:<10s}"
          f" 검사율 모델 {z['모델_검사율']:5.1f}% / 센서 {z['센서_검사율']:5.1f}%  전략 {z['모델전략']}/{z['센서전략']}")

# 기제: cl=8에서 모델이 과검사하는가
i8 = CLS.index(8)
ma8 = np.array(m_agg[8]); sa8 = np.array(s_agg[8])   # (seed,4) tn,fp,fn,tp
mech = {
    "모델": {"검사건수": int((ma8[:, 3] + ma8[:, 1]).mean()), "tp": int(ma8[:, 3].mean()), "fp": int(ma8[:, 1].mean()),
           "fn": int(ma8[:, 2].mean()), "검사율_%": round(float(m_rate[:, i8].mean()) * 100, 1)},
    "센서": {"검사건수": int((sa8[:, 3] + sa8[:, 1]).mean()), "tp": int(sa8[:, 3].mean()), "fp": int(sa8[:, 1].mean()),
           "fn": int(sa8[:, 2].mean()), "검사율_%": round(float(s_rate[:, i8].mean()) * 100, 1)},
}
mech["해석"] = (f"cl=8(저비용비)에서 모델은 검사율 {mech['모델']['검사율_%']}%, 센서는 {mech['센서']['검사율_%']}%. "
              + ("**모델이 더 많이 검사한다 = 과검사**. 유출비용이 낮은 구간에선 검사 건수 자체가 비용이므로, "
                 "적발력이 좋아도 검사를 많이 하면 손해다. 센서는 신호가 약해 일찍 전수통과로 후퇴하고 그게 정답이 된다."
                 if mech['모델']['검사율_%'] > mech['센서']['검사율_%'] else
                 "모델 검사율이 더 낮음 — 과검사 가설 기각. 다른 기제 필요."))
print("\n[C-기제] cl=8:", json.dumps(mech, ensure_ascii=False))

out = {"프로토콜": f"50 seed, cl {CLS[0]}~{CLS[-1]}, 최선전략 절대비용 대응차(센서−모델). 정규근사 95%CI.",
       "B_상단의존성": rule, "C_구간판정": zone, "C_기제_cl8": mech,
       "3구간_요약": {"모델우세_최소cl(단조유지)": 16,
                  "센서_유의우세_cl": [cl for cl in range(2, 25) if zone[str(cl)]["판정"] == "센서 유의우세"],
                  "대등_cl": [cl for cl in range(2, 25) if zone[str(cl)]["판정"] == "대등"]}}
json.dump(out, open(os.path.join(ROOT, "results", "breakeven_domain.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n→ results/breakeven_domain.json")
