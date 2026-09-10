"""
fig_cost_zones.py — 그림 5-4 재생성 (리뷰어 25차 (C) + 17→16 정정 병합).

교체 이유 3가지:
 1. 🔴 `fig_sensor_vs_model_v2.png`가 **철회된 승률컷 규칙**(wins≥75% & p<0.05)로 be=17을 그리고 있었다.
    (`process_opt5.py:80`) — 철회 목록 5번의 그림 상 부활. 정정 규칙으로 교체.
 2. cl 격자가 [10,12,14,15,16,...]로 성겨서 cl=7~9 구조가 보이지 않았다 → cl 2~30 촘촘히.
 3. 25차 확인 결과를 반영: 손익분기 16(단조유지 규칙, 정의역 cl∈[2,64]), cl=8 센서우세는
    **고립점**이므로 '구간 음영'이 아니라 **단일점 표식**으로 그린다(3구간 승격 반대 판정).

산출: figures/fig_sensor_vs_model_diff.png (정본 교체), figures/fig_cost_zones.png (동일 내용 신규명)
"""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["figure.dpi"] = 140
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
NSEED = 50
CLS = list(range(2, 31))

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
    return min(pass_all, inspect_all, inspect_all if rate >= CM.DEGEN_RATE else sel)


print(f"[precompute] {NSEED} seed × cl 2~30 ...")
diff = np.zeros((NSEED, len(CLS)))
for s in range(NSEED):
    d = np.load(mfiles[s]); mf = list(zip(d["in_cms"], d["te_cms"])); sf = sensor_folds(s)
    for i, cl in enumerate(CLS):
        diff[s, i] = best_cost(sf, cl) - best_cost(mf, cl)

mean = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(NSEED)
lo, hi = mean - 1.96 * se, mean + 1.96 * se
sig_m = lo > 0

# 손익분기 = 이후 단조유지 최소 cl (정의역 cl∈[2,64]는 별도 검증)
be = None
for i in range(len(CLS) - 1, -1, -1):
    if sig_m[i]: be = CLS[i]
    else: break
print(f"  손익분기(단조유지) = {be}")

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.axhline(0, color="#888", lw=.8)
ax.fill_between(CLS, lo, hi, color="#1f4e8c", alpha=.15, label="95% CI (50 seed 대응)")
ax.plot(CLS, mean, "o-", color="#1f4e8c", lw=1.6, ms=4, label="절대비용 차이 (센서 - 모델)")

# 모델 우위 구간(단조유지)만 음영 — cl=8은 고립점이라 음영 금지
ax.axvspan(be, CLS[-1], color="#1f4e8c", alpha=.07)
ax.axvline(be, color="#333", ls="--", lw=1.1)
ax.text(be + .3, max(mean) * .78, f"손익분기 cl={be}\nboot CI [14,16]\n(이후 단조유지 규칙,\n 정의역 cl∈[2,64])", fontsize=7.5, va="top")

# cl=8 고립점 표식 (구간 아님)
i8 = CLS.index(8)
ax.plot([8], [mean[i8]], "v", color="#d94f4f", ms=9, zorder=5)
ax.annotate("cl=8: 센서 유의우세 (고립점)\n동일 검사량서 tp +2.1건 = 국소 순위 역전\n[주의] cl 7·9는 대등 - 구간 아님, 다중비교 미보정",
            xy=(8, mean[i8]), xytext=(10.5, min(lo) - 50), fontsize=7.5, color="#a03030",
            arrowprops=dict(arrowstyle="->", color="#a03030", lw=1, connectionstyle="arc3,rad=-0.2"))

ax.set_xlabel("cl = 유출 : 검사 비용비")
ax.set_ylabel("절대비용 차이  (>0 = 모델 우위)"); ax.set_ylim(min(lo)-58, max(hi)*1.06)
ax.set_title("단일센서 S59 vs 474센서 모델 — 대응설계 정본 (50 seed)\n"
             f"모델 투자가 정당화되는 구간은 cl ≥ {be}  ·  cl 9~15는 대등  ·  전역 AUC 격차 0.052",
             fontsize=9.5)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)
fig.text(0.01, 0.005, "[주의] 승률컷 규칙(철회) 미사용. 25차 정정: cl<=15 대등 -> cl 9~15 대등.", fontsize=7, color="#666")
plt.tight_layout(rect=[0, 0.02, 1, 1])
for name in ("fig_sensor_vs_model_diff.png", "fig_cost_zones.png"):
    plt.savefig(os.path.join(ROOT, "figures", name))
plt.close()
print("→ figures/fig_sensor_vs_model_diff.png (정본 교체), figures/fig_cost_zones.png")

json.dump({"프로토콜": f"50 seed 대응, cl 2~30, 최선전략 절대비용차(센서−모델). 손익분기=이후 단조유지 최소 cl.",
           "손익분기": be, "정의역": "cl ∈ [2, 64] (cl8_promote_check.py 확인)",
           "cl별": {str(cl): {"평균차": round(float(mean[i]), 1), "ci": [round(float(lo[i]), 1), round(float(hi[i]), 1)]}
                  for i, cl in enumerate(CLS)},
           "⚠️철회규칙_미사용": "승률컷(wins≥75%&p<0.05 → cl=17)은 사용하지 않음. process_opt5.py의 fig_sensor_vs_model_v2.png는 이 규칙을 아직 쓰고 있어 [폐기] 처리 필요."},
          open(os.path.join(ROOT, "results", "fig_cost_zones.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
