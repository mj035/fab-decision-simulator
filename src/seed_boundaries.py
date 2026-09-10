"""
🔴 리뷰어 재요청 — 세 경계의 seed 분산 (서사 성패 결정).
seed별로 독립 측정:
  guard_cl  = 검사율(비용최적 임계값 기준) ≥ 0.90 도달 최소 cl        (운영 경계)
  econ_cl   = 경제 교차점 = best==2 '최대 연속구간(꼬리 제외)'의 상단   (보정 정의; blip/꼬리 강건)
  fn0_cl    = 누적 fn=0 도달 최소 cl (없으면 None)                    (표본 꼬리)
  rec_cl    = min(guard_cl, econ_cl)  = 그 seed의 보수적 권고 상한
동시 검증:
  · econ_cl 을 닫힌해 고정점 g(cl)=β·cl−upper(counts_at(cl)) 부호전환으로도 구해 Δ→0 확인 (리뷰어 #3)
  · 경계 순서(econ<guard = 순서 뒤집힘) seed 카운트
보고: min–max 가 아니라 중앙값+IQR+백분위 (리뷰어 5c). seed=CV분할만 (모델 rs 고정) → 분산 과소.
그림: seed별 세 경계 가로 스트립.
"""
import os, sys, time, json
import numpy as np, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["figure.dpi"] = 140
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from cache_cms import nested_cms, load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B, A, CS = 0.95, 0.02, 0
N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CL_LO, CL_HI = 2, 1500

def sequences(folds, N, npos, nneg):
    """cl별 best(0/1/2), rate, fn, 닫힌해상한(λ) 배열."""
    cls = np.arange(CL_LO, CL_HI)
    best = np.empty(len(cls), int); rate = np.empty(len(cls)); fn_ = np.empty(len(cls)); up_ = np.empty(len(cls))
    for i, cl in enumerate(cls):
        lam, mu, rho = CM.to_effective(cl, CS, B, A)
        sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
        tn, fp, fn, tp = agg
        ins = N + npos*cl - npos*lam + nneg*mu - npos*rho
        pa = npos*cl
        best[i] = int(np.argmin([pa, ins, sel])); rate[i] = (tp+fp)/N; fn_[i] = fn
        _, up = CM.band_closed_form(tn, fp, fn, tp, mu, rho); up_[i] = up
    return cls, best, rate, fn_, up_

def contiguous_runs(mask):
    runs = []; s = None
    for i, m in enumerate(mask):
        if m and s is None: s = i
        if not m and s is not None: runs.append((s, i-1)); s = None
    if s is not None: runs.append((s, len(mask)-1))
    return runs

X, y, info = load("SECOM"); N = len(y); npos = int(y.sum()); nneg = N - npos
print(f"SECOM N={N} 양성={npos}  seed {N_SEED}개 (seed=CV분할만, 모델 rs=0 고정 → 분산 과소평가)")
rows = []; t0 = time.time()
for s in range(N_SEED):
    ins, tes, auc = nested_cms(X, y, seed=s); folds = list(zip(ins, tes))
    cls, best, rate, fn_, up_ = sequences(folds, N, npos, nneg)
    # guard: rate>=0.9 첫 cl
    gidx = np.argmax(rate >= 0.90) if (rate >= 0.90).any() else None
    guard_cl = int(cls[gidx]) if gidx is not None else None
    # fn0 도달
    f0 = np.argmax(fn_ == 0) if (fn_ == 0).any() else None
    fn0_cl = int(cls[f0]) if f0 is not None else None
    # econ: best==2 연속구간 중 '꼬리(fn0 이상) 제외'한 최장 구간 상단
    sel_mask = best == 2
    if fn0_cl is not None:
        sel_mask = sel_mask & (cls < fn0_cl)     # fn=0 꼬리 제외
    runs = contiguous_runs(sel_mask)
    econ_cl = None
    if runs:
        lo, hi = max(runs, key=lambda r: r[1]-r[0])   # 최장 연속구간
        econ_cl = int(cls[hi])
    # 닫힌해 고정점 g(cl)=β·cl - up_ 부호전환(neg->pos) = 경제교차 (Δ→0 검증용)
    g = B*cls - up_          # up_=inf(fn=0)에선 g=-inf
    econ_closed = None
    below = (cls < fn0_cl) if fn0_cl is not None else np.ones(len(cls), bool)
    gp = np.where((g[:-1] < 0) & (g[1:] > 0) & below[1:])[0]
    if len(gp): econ_closed = int(cls[gp[0]+1])
    rec_cl = min([v for v in (guard_cl, econ_cl) if v is not None], default=None)
    order = ("econ<guard(뒤집힘)" if (econ_cl and guard_cl and econ_cl < guard_cl)
             else "guard<=econ" if (econ_cl and guard_cl) else "교차없음" if econ_cl is None else "-")
    rows.append({"seed": s, "auc": round(float(auc), 4), "guard_cl": guard_cl, "econ_cl": econ_cl,
                 "econ_closed": econ_closed, "fn0_cl": fn0_cl, "rec_cl": rec_cl, "order": order})
    d = None if (econ_cl is None or econ_closed is None) else econ_cl - econ_closed
    print(f"  s{s:>2} AUC {auc:.3f} | guard {str(guard_cl):>4} | econ {str(econ_cl):>5} (닫힌해 {str(econ_closed):>5}, Δ{str(d):>4}) "
          f"| fn0 {str(fn0_cl):>5} | rec {str(rec_cl):>4} | {order}  [{time.time()-t0:.0f}s]")

def stats(key):
    v = np.array([r[key] for r in rows if r[key] is not None], float)
    if not len(v): return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 1),
            "iqr": [round(float(np.percentile(v, 25)), 1), round(float(np.percentile(v, 75)), 1)],
            "p10": round(float(np.percentile(v, 10)), 1), "min": float(v.min()), "max": float(v.max())}

summary = {"n_seed": N_SEED, "seed_varies": "CV split only (model random_state fixed=0) → variance underestimated",
           "guard": stats("guard_cl"), "econ": stats("econ_cl"), "rec": stats("rec_cl"),
           "fn0_reach": sum(r["fn0_cl"] is not None for r in rows),
           "no_crossing": sum(r["econ_cl"] is None for r in rows),
           "order_flip": sum(r["order"].startswith("econ<") for r in rows),
           "econ_vs_closed_maxΔ": max([abs(r["econ_cl"]-r["econ_closed"]) for r in rows
                                       if r["econ_cl"] and r["econ_closed"]], default=None),
           "rows": rows}
json.dump(summary, open(os.path.join(ROOT, "results", "seed_boundaries.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ── 스트립 그림 ──
order_idx = sorted(range(len(rows)), key=lambda i: (rows[i]["rec_cl"] is None, rows[i]["rec_cl"] or 9e9))
fig, ax = plt.subplots(figsize=(8.5, max(4, N_SEED*0.22)))
for yy, i in enumerate(order_idx):
    r = rows[i]
    if r["guard_cl"]: ax.plot(r["guard_cl"], yy, "o", color="#1f4e8c", ms=6)
    if r["econ_cl"]:  ax.plot(r["econ_cl"],  yy, "s", color="#d94f4f", ms=6)
    if r["fn0_cl"]:   ax.plot(r["fn0_cl"],   yy, "^", color="#b8860b", ms=6)
    if r["guard_cl"] and r["econ_cl"]:
        ax.plot([min(r["guard_cl"],r["econ_cl"]),max(r["guard_cl"],r["econ_cl"])],[yy,yy],"-",color="#ccc",lw=1,zorder=0)
ax.set_xscale("log"); ax.set_xlabel("cl = 유출:검사 비용비"); ax.set_ylabel(f"seed (rec_cl 오름차순, n={N_SEED})")
ax.set_yticks([]); ax.set_title("세 경계의 seed 분산 — 단일 상한으로 말할 수 없다\n"
    "● 가드(검사율90%)  ■ 경제교차  ▲ fn=0꼬리   (seed=CV분할만, 모델rs고정→분산 과소)", fontsize=9)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="o",color="#1f4e8c",ls="",label="가드(검사율90%)"),
                   Line2D([],[],marker="s",color="#d94f4f",ls="",label="경제교차"),
                   Line2D([],[],marker="^",color="#b8860b",ls="",label="fn=0 꼬리")], fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "fig_seed_boundaries.png")); plt.close()

print(f"\n{'='*74}")
print(f"가드 경계:   {summary['guard']}")
print(f"경제 교차점: {summary['econ']}")
print(f"권고 상한 rec=min(guard,econ): {summary['rec']}")
print(f"fn=0 도달 {summary['fn0_reach']}/{N_SEED} | 교차없음 {summary['no_crossing']}/{N_SEED} | 순서뒤집힘(econ<guard) {summary['order_flip']}/{N_SEED}")
print(f"econ grid↔닫힌해 최대Δ: {summary['econ_vs_closed_maxΔ']}")
print("→ results/seed_boundaries.json, figures/fig_seed_boundaries.png")
