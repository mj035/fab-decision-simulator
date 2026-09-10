"""
근거 그림 — 'raw 선별 vs 전수검사'에는 유한 교차점이 없다(퇴화).
좌: sel-inspect 비용차(음수=선별승)를 cl에 대해. 비단조 + fn=0 이후 상수 -tn_skip 로 영구 미세승.
우: 검사율(rate). cl↑ 하면 rate→1 로 몰려 '선별'이 사실상 전수검사로 퇴화.
→ 그래서 유한 상한은 raw 경제교차가 아니라 '가드(검사율 컷)'가 정의한다.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETA, ALPHA, CS = 0.95, 0.02, 0

def load(name):
    d = np.load(os.path.join(ROOT, "results", "cms", f"{name}.npz"))
    return list(zip(d["in_cms"], d["te_cms"])), int(d["N"]), int(d["npos"]), int(d["nneg"]), float(d["auc"])

fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), gridspec_kw={"height_ratios": [2, 1]})
for col, name in enumerate(["SECOM", "APS"]):
    folds, N, npos, nneg, auc = load(name)
    cls = np.arange(2, 1500)
    delta = []; rate = []; fnz = []
    for cl in cls:
        lam, mu, rho = CM.to_effective(cl, CS, BETA, ALPHA)
        sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
        tn, fp, fn, tp = agg
        ins = N + npos * cl - npos * lam + nneg * mu - npos * rho
        delta.append(sel - ins); rate.append((tp + fp) / N); fnz.append(fn)
    delta = np.array(delta); rate = np.array(rate); fnz = np.array(fnz)
    genuine = CM.genuine_selective_band(folds, N, npos, nneg, CS, BETA, ALPHA)

    ax = axes[0, col]
    ax.axhline(0, color="#888", lw=.8)
    ax.plot(cls, delta, color="#2e7d32", lw=1.3)
    ax.fill_between(cls, delta, 0, where=delta < 0, color="#2e7d32", alpha=.15)
    ax.fill_between(cls, delta, 0, where=delta > 0, color="#d94f4f", alpha=.15)
    if genuine:
        ax.axvline(genuine[1], color="#1f4e8c", ls="--", lw=1.2)
        ax.text(genuine[1], ax.get_ylim()[1]*.7, f" 가드경계 cl={genuine[1]}\n (검사율 90%컷)",
                color="#1f4e8c", fontsize=8, va="top")
    fz = cls[np.argmax(fnz == 0)] if (fnz == 0).any() else None
    if fz is not None:
        ax.axvline(fz, color="#b8860b", ls=":", lw=1.2)
        ax.text(fz, ax.get_ylim()[0]*.7, f" fn=0 도달 cl={fz}\n → 이후 상수 미세승(교차없음)",
                color="#b8860b", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_title(f"{name} (AUC {auc:.3f}) · cs=0\nsel-inspect 비용차 (음수=선별승)", fontsize=9)
    ax.set_ylabel("비용차 (선별 - 전수검사)")

    ax2 = axes[1, col]
    ax2.plot(cls, rate * 100, color="#555", lw=1.2)
    ax2.axhline(90, color="#1f4e8c", ls="--", lw=1)
    ax2.text(cls[0], 91, "검사율 90% (가드)", color="#1f4e8c", fontsize=7)
    ax2.set_xscale("log"); ax2.set_ylim(0, 105)
    ax2.set_xlabel("cl = 유출:검사 비용비"); ax2.set_ylabel("검사율 %")

fig.suptitle("raw 선별 vs 전수검사: 유한 교차점이 없다 — 상한은 '경제교차'가 아니라 '가드(검사율컷)'가 정의",
             fontsize=10.5, y=.99)
plt.tight_layout(rect=[0, 0, 1, .96])
out = os.path.join(ROOT, "figures", "fig_raw_no_crossing.png")
plt.savefig(out); plt.close()
print("->", out)
