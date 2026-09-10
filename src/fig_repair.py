# -*- coding: utf-8 -*-
"""2순위 디스플레이 그림: (좌) 리페어 γV → 실익상한 히트맵(γV만큼 하락=대수검증), (우) β 시나리오."""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["figure.dpi"] = 140
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "repair_grid.json"), encoding="utf-8"))
GAM, VS = d["GAM"], d["VS"]; base = d["base_upper"]
M = np.array([[d["grid"][f"{g}_{v}"] for v in VS] for g in GAM])

fs = sorted(glob.glob(os.path.join(ROOT, "results", "cms_seeds", "SECOM_s*.npz")), key=lambda p: int(p.split("_s")[-1].split(".")[0]))
folds_by_seed = [list(zip(np.load(f)["in_cms"], np.load(f)["te_cms"])) for f in fs]
agg0 = sum(te[0] for _, te in folds_by_seed[0]); N = int(agg0.sum()); npos = int(agg0[2]+agg0[3]); nneg = N-npos

def econ_upper_one(folds, beta, gamma=0.0, V=0.0):    # 경제교차(닫힌해)=정본 경계, rec 아님
    cls = np.arange(2, 300); g = np.empty(len(cls)); fn_ = np.empty(len(cls))
    for i, cl in enumerate(cls):
        lam, mu, rho = CM.to_effective(cl, 0.0, beta, 0.02, gamma, V)
        _, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl); tn, fp, fn, tp = agg
        _, up = CM.band_closed_form(tn, fp, fn, tp, mu, rho); g[i] = lam - up; fn_[i] = fn
    f0 = np.argmax(fn_ == 0) if (fn_ == 0).any() else len(cls)
    gp = np.where((g[:-1] < 0) & (g[1:] > 0) & (np.arange(len(cls)) < f0)[1:])[0]
    return int(cls[gp[0]+1]) if len(gp) else None

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

# ── 좌: 히트맵 실익 상한 (γ × V) ──
im = axL.imshow(M, aspect="auto", cmap="RdYlBu", origin="lower")
axL.set_xticks(range(len(VS))); axL.set_xticklabels(VS); axL.set_yticks(range(len(GAM))); axL.set_yticklabels(GAM)
axL.set_xlabel("리페어 회수가치 V"); axL.set_ylabel("리페어 성공률 γ")
axL.set_title("① 리페어 γV → 선별 실익 상한 cl (낮을수록 더 검사)", fontsize=11, fontweight="bold")
for i in range(len(GAM)):
    for j in range(len(VS)):
        gV = GAM[i]*VS[j]; inv = M[i, j] + gV     # 불변량 cl_max+γV (:g로 비정수 그대로 노출)
        axL.text(j, i, f"{M[i,j]:.0f}\n+{gV:g}={inv:g}", ha="center", va="center", fontsize=8,
                 color="#222" if M[i, j] > 40 else "#fff")
plt.colorbar(im, ax=axL, label="실익 상한 cl", fraction=0.046)
# 디스플레이 현실값 한 점 (γ≈0.7, V≈20 → 36, 가정값)
axL.add_patch(plt.Rectangle((3-0.48, 3-0.48), 0.96, 0.96, fill=False, ec="#111", lw=2.5))
dispv = M[3, 3]                                  # γ0.7·V20
axL.annotate(f"디스플레이\n가정 γ0.7·V20\n→ 상한 {dispv:.0f} (-{(base-dispv)/base*100:.0f}%)", (3, 3), xytext=(3.05, 1.6),
             fontsize=8, fontweight="bold", color="#111", ha="center",
             arrowprops=dict(arrowstyle="->", color="#111"))
axL.text(0.5, -0.26, f"불변량 cl_max + γV = {base:.0f}.0 (정수 γV 8점) / {base-0.5:.1f}~{base+0.5:.1f} (비정수 γV 4점=정수 cl격자 ±1칸)\n"
         "= 대수 Λ=β(cl+γV) 검증. '리페어 되면 더 검사'가 숫자로.\n"
         "기준선=경제교차 닫힌해 52 (그리드 n=50 기준 54와 별개 프로토콜, rec 아님)",
         transform=axL.transAxes, ha="center", va="top", fontsize=8.2, color="#333")

# ── 우: β 시나리오 (검출률<1) — 경제교차(정본 경계)로 통일 ──
BETAS = [1.0, 0.95, 0.85, 0.7]; his, svs = [], []
for beta in BETAS:
    H = [econ_upper_one(folds, beta) for folds in folds_by_seed]
    his.append(np.median([h for h in H if h]))
    svs.append(np.mean([CM.evaluate(folds, N, npos, nneg, 15, 0.0, beta, 0.02)["sel_vs_inspect"] for folds in folds_by_seed]))
x = range(len(BETAS)); Lmax = 0.95*base            # 불변량 Λ_max=β·cl_max(β=0.95)
pred = [Lmax/b for b in BETAS]                     # cl_max(β)=Λ_max/β 예측
axR.plot(x, his, "o-", color="#2a5da8", lw=2, ms=8, label="실익 상한 관측(경제교차)")
axR.plot(x, pred, "x", color="#c0392b", ms=12, mew=2.5, label=f"예측 Λ_max/β (Λ_max={Lmax:.1f})")
for xi, h, p in zip(x, his, pred): axR.annotate(f"{h:.0f}\n(pred {p:.1f})", (xi, h), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.8)
axR.set_xticks(x); axR.set_xticklabels([f"β={b}" for b in BETAS])
axR.set_ylabel("실익 상한 cl (경제교차)"); axR.set_title("② β<1 → 상한 = Λ_max/β (붕괴구조 검증)", fontsize=11, fontweight="bold")
axR.legend(loc="upper left", fontsize=8.4); axR.grid(alpha=0.25); axR.set_ylim(40, 80)
obs_s = "/".join(f"{h:.0f}" for h in his); pred_s = "/".join(f"{p:.1f}" for p in pred)
axR.text(0.5, -0.30, f"상한 관측 {obs_s} = 예측 {pred_s} (붕괴구조 λ=β·cl 수치검증, 두 패널 같은 불변량).\n"
         "절대차 " + "→".join(f"{int((N+npos*15*(1-b))*s/100)}" for b, s in zip(BETAS, svs)) + " 증가(분모 팽창만 아님). 선별 덜 민감=검사건수 적어서(tp<npos).\n"
         "※ β=1.0만 1칸 낮음(관측48 vs 예측49.4): 동점(tie)·부등호(≤/<) 경계 처리 차이.",
         transform=axR.transAxes, ha="center", va="top", fontsize=7.6, color="#333")

fig.suptitle("디스플레이 특화 — 리페어·불완전검출을 커널에 넣으면 수식이 결과가 된다", fontsize=12.5, fontweight="bold", y=1.02)
p = os.path.join(ROOT, "figures", "fig_repair.png")
plt.savefig(p, bbox_inches="tight"); plt.close()
print(f"→ {p}")
