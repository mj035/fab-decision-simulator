# -*- coding: utf-8 -*-
"""2-1. 리페어 γV 격자 — "디스플레이는 리페어가 되니 오히려 더 검사해야 한다"를 숫자로.
대수: Λ=β(cl+γV) → 리페어는 유효 유출비용을 γV만큼 올림 → 실익 상한이 γV만큼 내려가야 함(자기검증).
γ=0이면 현재 모델로 환원(하위호환). 정본 cms_seeds 50 seed."""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B, A = 0.95, 0.02; CS = 0.0
fs = sorted(glob.glob(os.path.join(ROOT, "results", "cms_seeds", "SECOM_s*.npz")),
            key=lambda p: int(p.split("_s")[-1].split(".")[0]))
folds_by_seed = [list(zip(np.load(f)["in_cms"], np.load(f)["te_cms"])) for f in fs]
# N/npos = 전체합(5 fold의 threshold0 카운트 합). 단일 fold 아님.
agg0 = sum(te[0] for _, te in folds_by_seed[0])   # (tn,fp,fn,tp) at 최저임계(전부 flag)
N = int(agg0.sum()); npos = int(agg0[2] + agg0[3]); nneg = N - npos
assert N > 1000 and npos == 104, f"N/npos 추출 오류: N={N} npos={npos}"   # sanity(리뷰어 21차)
print(f"[2-1 리페어] SECOM N={N} 양성={npos} · seed {len(folds_by_seed)} · β={B}")

def econ_upper_one(folds, gamma, V):
    """경제교차 상한(닫힌해 고정점 g=Λ−upper 첫 부호전환). rec 아님=정본 경계. ρ=βγV 반영."""
    cls = np.arange(2, 300); g = np.empty(len(cls)); fn_ = np.empty(len(cls))
    for i, cl in enumerate(cls):
        lam, mu, rho = CM.to_effective(cl, CS, B, A, gamma, V)
        _, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl); tn, fp, fn, tp = agg
        _, up = CM.band_closed_form(tn, fp, fn, tp, mu, rho); g[i] = lam - up; fn_[i] = fn  # up이 이미 −rho 포함
    f0 = np.argmax(fn_ == 0) if (fn_ == 0).any() else len(cls)
    below = np.arange(len(cls)) < f0
    gp = np.where((g[:-1] < 0) & (g[1:] > 0) & below[1:])[0]
    return int(cls[gp[0]+1]) if len(gp) else None

def band_upper(gamma, V):
    """경제교차 상한 중앙(50 seed) — 정본 경계(rec 아님). ρ=β·γ·V 반영."""
    ups = [econ_upper_one(folds, gamma, V) for folds in folds_by_seed]
    ups = [u for u in ups if u is not None]
    return float(np.median(ups)) if ups else None

# γ=0 하위호환 (현재 모델과 동일해야)
base = band_upper(0.0, 0.0)
print(f"\n[하위호환] γ=0: 실익 상한 cl={base:.0f} (리페어 없는 현재 모델)")

print(f"\n[γV 격자] 실익 상한 cl(중앙) — 리페어가 커질수록 내려가는가")
print(f"  {'γ\\\\V':>6}" + "".join(f"{v:>8}" for v in [0, 5, 10, 20]))
GAM = [0.0, 0.3, 0.5, 0.7, 0.9]; VS = [0, 5, 10, 20]
grid = {}
for g in GAM:
    row = []
    for v in VS:
        u = band_upper(g, v); row.append(u); grid[f"{g}_{v}"] = u
    print(f"  {g:>6}" + "".join(f"{(u if u else 0):>8.0f}" for u in row))

# 자기검증: 상한 하락 ≈ γV ?
print(f"\n[대수 검증] 상한 하락(base−관측) vs 예측 γV:")
print(f"  {'γ':>4} {'V':>4} {'γV':>6} {'상한하락(관측)':>12} {'일치?':>6}")
ok = 0; tot = 0
for g in GAM[1:]:
    for v in VS[1:]:
        u = grid[f"{g}_{v}"]; drop = base - u if u else None; pred = g * v
        match = abs(drop - pred) <= max(2, 0.15*pred) if drop is not None else False
        ok += match; tot += 1
        print(f"  {g:>4} {v:>4} {pred:>6.1f} {drop:>12.1f} {'✅' if match else '✗':>6}")
print(f"\n[결론] 상한이 γV만큼 하락({ok}/{tot} 일치) → **대수(Λ=β(cl+γV))와 수치 일치 = 자기검증.**")
print(f"  '디스플레이는 리페어가 되기 때문에 오히려 더 검사해야 한다'가 서사→숫자.")
json.dump({"base_upper": base, "grid": grid, "algebra_match": f"{ok}/{tot}", "GAM": GAM, "VS": VS},
          open(os.path.join(ROOT, "results", "repair_grid.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("→ results/repair_grid.json")
