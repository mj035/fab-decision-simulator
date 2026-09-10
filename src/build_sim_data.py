# -*- coding: utf-8 -*-
"""본선 시뮬레이터 데이터 생성 → web/sim_final.json
 ② 역산: 검사율(조밀 grid) → 정점스냅 → 지지비율≥50% R구간 + 지지곡선 (50 seed, inverse_validate와 동일 프로토콜)
 ① 정책: 50seed 평균 혼동행렬(300임계) → 브라우저서 cl·cs·β·α로 3전략+폴백 계산 / 선별밴드 IQR
합성 없음. 전부 SECOM 실측 50seed nested CV 캐시(results/cms_seeds)에서 계산."""
import os, sys, json, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THS = np.linspace(0.005, 0.995, 300)
B, A = 0.95, 0.02

fs = sorted(glob.glob(os.path.join(ROOT, "results", "cms_seeds", "SECOM_s*.npz")),
            key=lambda p: int(p.split("_s")[-1].split(".")[0]))
aggs = [np.load(f)["te_cms"].sum(axis=0) for f in fs]        # 각 (300,4) = seed별 outer 누적
folds_by_seed = [list(zip(np.load(f)["in_cms"], np.load(f)["te_cms"])) for f in fs]
N = int(aggs[0][0].sum()); npos = int(aggs[0][0][2] + aggs[0][0][3]); nneg = N - npos
print(f"SECOM N={N} 양성={npos} · seed {len(aggs)}")

def hull_vertices(cm):
    fpr = cm[:,1]/(cm[:,0]+cm[:,1]); tpr = cm[:,3]/(cm[:,2]+cm[:,3])
    idx = sorted(range(len(cm)), key=lambda j:(fpr[j], tpr[j])); hull=[]
    for j in idx:
        while len(hull) >= 2:
            x1,y1=fpr[hull[-2]],tpr[hull[-2]]; x2,y2=fpr[hull[-1]],tpr[hull[-1]]; x3,y3=fpr[j],tpr[j]
            if (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1) >= 0: hull.pop()
            else: break
        hull.append(j)
    return hull

def snap_interval(agg, r_obs):
    hv = hull_vertices(agg); rates = [(agg[j,3]+agg[j,1])/N for j in hv]
    k = int(np.argmin([abs(rv-r_obs) for rv in rates]))
    lo, hi = CM.implied_R_interval(agg, hv[k])
    return rates[k], lo, hi

# ══ ② 역산: 조밀 검사율 grid ══
RATE_GRID = [round(r, 3) for r in np.arange(0.03, 0.66, 0.01)]
Rg = np.round(np.logspace(np.log10(1), np.log10(120), 200), 2)     # 지지곡선 R축(정본 해상도)
inv = []
for r_obs in RATE_GRID:
    snaps, ivs = [], []
    for agg in aggs:
        sr, lo, hi = snap_interval(agg, r_obs)
        snaps.append(sr)
        if lo is not None: ivs.append((lo, hi))
    sup = np.array([np.mean([lo <= R <= (hi if np.isfinite(hi) else np.inf) for lo, hi in ivs]) if ivs else 0.0
                    for R in Rg])
    peak = float(sup.max())
    h50 = Rg[sup >= 0.5]                          # 다수지지(≥50%) 밴드 — 정본, peak≥0.5일 때만 존재
    fwhm = Rg[sup >= peak/2] if peak > 0 else np.array([])   # FWHM: 항상 존재(크래시 방지, 분광학 표준)
    inv.append({
        "rate": r_obs,
        "snap_rate": round(float(np.median(snaps)), 4),
        "snap_dist": round(float(np.median([abs(s-r_obs) for s in snaps])), 4),
        "R50_lo": round(float(h50.min()), 2) if len(h50) else None,   # ≥50% 하한(정본)
        "R50_hi": round(float(h50.max()), 2) if len(h50) else None,
        "Rf_lo": round(float(fwhm.min()), 2) if len(fwhm) else None,  # FWHM 하한(폴백)
        "Rf_hi": round(float(fwhm.max()), 2) if len(fwhm) else None,
        "peak": round(peak, 2),
        "majority": bool(peak >= 0.5),
        "sup": [round(float(s), 3) for s in sup],
    })
print(f"② 역산 grid {len(inv)}점 (검사율 {RATE_GRID[0]*100:.0f}~{RATE_GRID[-1]*100:.0f}%)")

# ══ ① 정책: (λ,μ) 격자마다 nested 내부선택 미리계산 (낙관편향 제거 = 정본과 일치) ══
# 각 fold: 내부 in_cm으로 임계 선택 → 외부 te_cm 카운트 누적(50seed 평균 = 단일N 스케일).
MUS = [0.0, 0.25, 0.5, 1.0, 2.0]
LAMS = list(range(2, 101))
grid = {}
for mu in MUS:
    for lam in LAMS:
        aggsum = np.zeros(4); jj = []
        for folds in folds_by_seed:
            for in_cm, te_cm in folds:
                j = CM.best_threshold_idx(in_cm, lam, mu, 0.0)
                aggsum += te_cm[j]; jj.append(j)
        cnt = aggsum / len(folds_by_seed)                       # (tn,fp,fn,tp) 단일N 스케일
        grid[f"{lam}_{mu:g}"] = {"c": [round(float(x), 1) for x in cnt],   # :g → JS ${mu}와 동일 키
                                 "th": round(float(THS[int(np.median(jj))]), 3)}
# 선별상한 = 두 경계(가드·경제교차) — 정본 seed_boundaries.json. ⚠️ rec=min(가드,교차)는 철회(순열대조 기각).
sb = json.load(open(os.path.join(ROOT, "results", "seed_boundaries.json"), encoding="utf-8"))
ab = json.load(open(os.path.join(ROOT, "results", "analyze_boundaries.json"), encoding="utf-8"))
policy = {"MUS": MUS, "LAMS": LAMS, "grid": grid,
          "guard_median": sb["guard"]["median"], "guard_iqr": sb["guard"]["iqr"],
          "econ_median": sb["econ"]["median"], "econ_iqr": sb["econ"]["iqr"],
          "corr_r": ab["corr_guard_econ"]["pearson"], "order_flip": sb["order_flip"], "n_seed": sb["n_seed"]}
# 검산: cl=15,μ=0 nested sel_vs_inspect (정본 +28.7% 대조)
c15 = grid["14_0"]["c"]; tn,fp,fn,tp = c15; lam=B*15
pa=npos*15; ia=N+npos*15-npos*lam; sv=(tp+fp)+npos*15-tp*lam
print(f"① 정책 격자 {len(grid)}개. 검산 cl=15(λ=14.25,μ=0): sel_vs_inspect {(ia-sv)/ia*100:+.1f}%")
print(f"   두 경계: 가드 {sb['guard']['median']:.0f}{sb['guard']['iqr']} · 경제교차 {sb['econ']['median']:.0f}{sb['econ']['iqr']} · r={ab['corr_guard_econ']['pearson']} 순서뒤집힘 {sb['order_flip']}/{sb['n_seed']} (rec 철회)")

out = {"meta": {"N": N, "npos": npos, "nneg": nneg, "prevalence": round(npos/N*100, 2),
                "beta": B, "alpha": A, "seeds": len(aggs),
                "hull_median": 15, "note": "SECOM 실측 50seed nested CV. 합성 없음."},
       "Rg": [round(float(r), 2) for r in Rg], "inverse": inv, "policy": policy}
os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "web", "sim_final.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
sz = os.path.getsize(os.path.join(ROOT, "web", "sim_final.json"))/1024
print(f"→ web/sim_final.json ({sz:.0f} KB)")
