"""
drift_v2.py — 드리프트 재실행 (리뷰어 23차 §2). 시간축=행 인덱스(생산 순서). 손상 타임스탬프 불사용.

기존 drift.py/drift2.py는 손상시간 argsort로 행을 섞은 뒤 quartile을 만들었음 → 값 전부 폐기.
 (a) 유병률 quartile + chi2
 (b) 입력 드리프트: PSI(전후반) + domain classifier OOF AUC (s59_timeproxy_v2 결과 인용)
 (c) 구간별 모델 OOF AUC / pAUC[재현율 .9~1] + S59 단독 AUC (→ S59 판정의 대조축)
 (d) 고정 cl(15,20,30) 구간별 최적 검사율·절감률 — 임계값은 나머지 3개 구간에서 선택(leave-quartile-out, 정직)
 (e) 구간별 역산 R (정점스냅, 검사율 12%/30%)
 (f) 유병률 통제 반사실: 최고유병률 구간을 전체평균으로 다운샘플 → R 이동 잔존 여부
산출: results/drift_v2.json, figures/fig_drift_v2.png
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
raw = pd.read_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"))
# ⚠️ 정렬 금지 — 원본 행 순서 = 생산 순서 (fix_time.py 검증: 복원시간 spearman 1.0, 역전 0)
X, y, info = adapt(raw.drop(columns=["Time"]), "Pass/Fail", 1); y = np.asarray(y); N = len(y); npos = int(y.sum())
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xi = X[valid].fillna(X[valid].median()).to_numpy(); j59 = valid.index("59")
q = np.floor(np.arange(N) / N * 4).astype(int); q[q == 4] = 3
half = (np.arange(N) >= N / 2).astype(int)

def mkpipe(spw): return Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()),
    ("m", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.3, reg_lambda=10.0, scale_pos_weight=spw, random_state=0, n_jobs=-1, verbose=-1))])

# ── (a) 유병률 quartile + chi2 ──
prev = [float(y[q == b].mean()) for b in range(4)]
tab = np.array([[int(y[q == b].sum()), int((q == b).sum() - y[q == b].sum())] for b in range(4)])
chi2, pv, _, _ = chi2_contingency(tab)
print(f"[(a)] 유병률 Q1~Q4 (행 인덱스): {[round(p*100,1) for p in prev]}% · 양성 {[int(r[0]) for r in tab]} · chi2 p={pv:.2e}")

# ── (b) 입력 드리프트 (라벨 무관) ──
def psi(a, b, bins=10):
    cut = np.quantile(a, np.linspace(0, 1, bins + 1)); cut[0] = -np.inf; cut[-1] = np.inf
    pa = np.histogram(a, cut)[0] / len(a) + 1e-6; pb = np.histogram(b, cut)[0] / len(b) + 1e-6
    return np.sum((pb - pa) * np.log(pb / pa))
early, late = Xi[half == 0], Xi[half == 1]
psis = np.array([psi(early[:, j], late[:, j]) for j in range(len(valid))])
try:
    s59j = json.load(open(os.path.join(ROOT, "results", "s59_timeproxy_v2.json"), encoding="utf-8"))
    domain_auc = s59j["a_domain"]["half_binary"]["oof_auc"]
except Exception:
    domain_auc = None
print(f"[(b)] PSI>0.2 센서 {int((psis>0.2).sum())}/{len(valid)} · 최대 {psis.max():.2f} · domain OOF AUC {domain_auc} (s59_v2 인용)")

# ── OOF 점수 (전체 모델, 랜덤 CV seed7 — 시간 무관 프로토콜) ──
oof = cross_val_predict(mkpipe((y == 0).sum() / npos), X[valid], y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba", n_jobs=-1)[:, 1]
print(f"\n전체 OOF AUC {roc_auc_score(y, oof):.3f}")

# ── (c) 구간별 모델 AUC / pAUC + S59 단독 ──
def pauc_tail(yy, ss, lo=0.9):
    """재현율(TPR) [lo,1] 구간의 TNR 적분 (seed_tail_metric 방식) — 고재현율 꼬리 성능."""
    fpr, tpr, _ = roc_curve(yy, ss)
    band = tpr >= lo
    if band.sum() < 2: return None
    o = np.argsort(tpr[band]); t_, s_ = tpr[band][o], 1 - fpr[band][o]
    return float(np.trapezoid(s_, t_) / max(t_[-1] - t_[0], 1e-9))
model_q, pauc_q, s59_q = [], [], []
for b in range(4):
    m = q == b
    model_q.append(round(float(roc_auc_score(y[m], oof[m])), 3))
    pauc_q.append(round(pauc_tail(y[m], oof[m]), 3))
    a59 = roc_auc_score(y[m], Xi[m, j59]); s59_q.append(round(max(a59, 1 - a59), 3))
print(f"[(c)] 구간별 모델 OOF AUC : {model_q}")
print(f"      구간별 pAUC(재현율.9~1): {pauc_q}")
print(f"      구간별 S59 단독 AUC  : {s59_q}  ← 모델과 대조(S59 판정 축)")

# ── (d) 고정 cl 구간별 최적 검사율·절감률 (임계값 = 나머지 3구간에서 선택) ──
THS = np.linspace(0.005, 0.995, 300)
def cm_of(idx):
    s = oof[idx]; yy = y[idx]
    return np.array([[int(((s < tt) & (yy == 0)).sum()), int(((s >= tt) & (yy == 0)).sum()),
                      int(((s < tt) & (yy == 1)).sum()), int(((s >= tt) & (yy == 1)).sum())] for tt in THS])
d_res = {}
print(f"\n[(d)] 고정 cl 구간별 (임계값 leave-quartile-out):")
for cl in [15, 20, 30]:
    lam, mu, rho = CM.to_effective(cl, 0, B, A)
    row = []
    for b in range(4):
        idx, oth = np.where(q == b)[0], np.where(q != b)[0]
        j = CM.best_threshold_idx(cm_of(oth), lam, mu, rho)          # 정직: 임계값은 밖에서
        tn, fp, fn, tp = cm_of(idx)[j]
        n_, np_, nn_ = len(idx), int(y[idx].sum()), int((y[idx] == 0).sum())
        _, inspect_all, selective = CM.strat_costs(n_, np_, nn_, tp, fp, fn, cl, lam, mu, rho)
        row.append({"rate": round((tp + fp) / n_, 3), "sel_vs_inspect_%": round((inspect_all - selective) / inspect_all * 100, 1),
                    "npos": np_})
    d_res[cl] = row
    print(f"  cl={cl}: " + " · ".join(f"Q{b+1} 검사율{r['rate']:.0%}/절감{r['sel_vs_inspect_%']}%(양성{r['npos']})" for b, r in enumerate(row)))

# ── (e) 구간별 역산 R (정점스냅) ──
def hull(cm):
    fpr = cm[:, 1] / np.maximum(cm[:, 0] + cm[:, 1], 1); tpr = cm[:, 3] / np.maximum(cm[:, 2] + cm[:, 3], 1)
    idx = sorted(range(len(cm)), key=lambda j: (fpr[j], tpr[j])); h = []
    for j in idx:
        while len(h) >= 2:
            x1, y1 = fpr[h[-2]], tpr[h[-2]]; x2, y2 = fpr[h[-1]], tpr[h[-1]]; x3, y3 = fpr[j], tpr[j]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) >= 0: h.pop()
            else: break
        h.append(j)
    return h
def R_at_rate(cm, n_, rate):
    hv = hull(cm); rates = [(cm[j, 3] + cm[j, 1]) / n_ for j in hv]
    k = hv[int(np.argmin([abs(r - rate) for r in rates]))]
    return CM.implied_R_interval(cm, k)
Rres = {}
print(f"\n[(e)] 구간별 역산 R:")
for rate in [0.12, 0.30]:
    row = []
    for b in range(4):
        idx = np.where(q == b)[0]; lo, hi = R_at_rate(cm_of(idx), len(idx), rate)
        row.append(None if lo is None else [round(lo, 1), round(hi, 1) if np.isfinite(hi) else None])
    Rres[f"{rate:.2f}"] = row
    print(f"  검사율 {rate*100:.0f}%: Q1~Q4 R = {row}")

# ── (f) 유병률 통제 반사실 — 최고유병률 구간을 전체평균 6.64%로 다운샘플 ──
rng = np.random.default_rng(0)
hi_b = int(np.argmax(prev)); target_prev = float(y.mean())
idx_hi = np.where(q == hi_b)[0]; p_ = idx_hi[y[idx_hi] == 1]; n_ = idx_hi[y[idx_hi] == 0]
keep_pos = max(2, int(round(len(n_) * target_prev / (1 - target_prev))))   # 음성 고정, 양성 다운샘플
ctrl_lo = []
for _ in range(50):
    sub = np.concatenate([rng.choice(p_, min(keep_pos, len(p_)), replace=False), n_])
    lo, hi = R_at_rate(cm_of(sub), len(sub), 0.30)
    if lo is not None: ctrl_lo.append(lo)
ctrl_med = float(np.median(ctrl_lo)) if ctrl_lo else None
raw_hi = Rres["0.30"][hi_b]; others = [r for b, r in enumerate(Rres["0.30"]) if b != hi_b and r]
oth_med = float(np.median([r[0] for r in others])) if others else None
print(f"\n[(f)] 반사실: Q{hi_b+1}(유병률 {prev[hi_b]*100:.1f}%)을 {target_prev*100:.2f}%로 다운샘플(양성 {len(p_)}→{keep_pos})")
print(f"      Q{hi_b+1} 원 R⁻={raw_hi[0] if raw_hi else None} → 통제후 중앙 {ctrl_med:.1f} · 타구간 중앙 {oth_med:.1f}"
      f" → {'통제후 타구간에 근접 = R 이동은 유병률 산술' if ctrl_med and oth_med and abs(ctrl_med-oth_med) < abs(raw_hi[0]-oth_med)*0.5 else '통제후에도 차이 잔존 = 관계 드리프트 성분'}")

# ── 구버전(손상시간) 값 로드 — 신구 대조용 ──
old = {}
for f_, k_ in [("drift.json", "drift_v1"), ("drift2.json", "drift2_v1")]:
    try: old[k_] = json.load(open(os.path.join(ROOT, "results", f_), encoding="utf-8"))
    except Exception: pass

out = {
    "프로토콜": "시간축=행 인덱스 quartile(생산 순서, fix_time.py). OOF=StratifiedKFold(5,shuffle,seed7) 랜덤CV. "
             "(d) 임계값 leave-quartile-out. 구간당 양성 10~45 → 구간 통계 노이즈 큼(MDD 유의).",
    "a_유병률": {"quartile_%": [round(p * 100, 2) for p in prev], "양성수": [int(r[0]) for r in tab], "chi2_p": float(f"{pv:.3e}")},
    "b_입력드리프트": {"psi_gt02": int((psis > 0.2).sum()), "max_psi": round(float(psis.max()), 2), "domain_auc_binary": domain_auc},
    "c_구간별": {"model_oof_auc": model_q, "pauc_tail_.9": pauc_q, "s59_solo_auc": s59_q},
    "d_고정cl": d_res,
    "e_역산R": Rres,
    "f_반사실": {"통제구간": f"Q{hi_b+1}", "원유병률_%": round(prev[hi_b] * 100, 1), "목표_%": round(target_prev * 100, 2),
               "원R하한": raw_hi[0] if raw_hi else None, "통제후_중앙": round(ctrl_med, 1) if ctrl_med else None,
               "타구간_중앙": round(oth_med, 1) if oth_med else None, "정의됨": len(ctrl_lo)},
    "구버전_손상시간_폐기값": {"유병률_%": [round(p * 100, 1) for p in old.get("drift_v1", {}).get("quartile_prev", [])],
                      "domain_auc": old.get("drift_v1", {}).get("input_drift", {}).get("domain_auc"),
                      "역산R": old.get("drift2_v1", {}).get("inverse_R_quartile")},
}
json.dump(out, open(os.path.join(ROOT, "results", "drift_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n→ results/drift_v2.json")

# ══════════════ fig_drift_v2.png ══════════════
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["figure.dpi"] = 140
BLUE, RED, GRAY, GOLD = "#4f86d9", "#d94f4f", "#8c8c8c", "#d9a44f"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.suptitle("드리프트 재실행 (시간축=행 인덱스·생산 순서) — 구버전(손상 타임스탬프) 값은 전부 폐기", fontsize=12.5, fontweight="bold")

xs = np.arange(4)
old_prev = old.get("drift_v1", {}).get("quartile_prev", [None] * 4)
if all(v is not None for v in old_prev):
    ax1.bar(xs - 0.2, [v * 100 for v in old_prev], 0.4, color=GRAY, alpha=0.7, label="구버전(손상시간) — 폐기")
ax1.bar(xs + 0.2, [p * 100 for p in prev], 0.4, color=BLUE, label="복원(행 인덱스)")
ax1.axhline(y.mean() * 100, color=RED, lw=1, ls="--", label=f"전체 {y.mean()*100:.1f}%")
ax1.set_xticks(xs, [f"Q{b+1}" for b in xs]); ax1.set_ylabel("유병률 %")
ax1.set_title(f"(a) 유병률 드리프트 (chi2 p={pv:.1e})", fontsize=10.5); ax1.legend(fontsize=8)

ax2.plot(xs, model_q, "o-", color=BLUE, label="474센서 모델 (OOF)")
ax2.plot(xs, s59_q, "s-", color=RED, label="S59 단독")
ax2.axhline(0.5, color=GRAY, lw=1, ls=":")
ax2.set_xticks(xs, [f"Q{b+1}" for b in xs]); ax2.set_ylabel("AUC"); ax2.set_ylim(0.45, 0.95)
ax2.set_title("(c) 구간별 AUC — 모델 vs S59", fontsize=10.5); ax2.legend(fontsize=8.5)
for b in range(4): ax2.annotate(f"{s59_q[b]:.2f}", (b, s59_q[b]), textcoords="offset points", xytext=(0, -14), fontsize=8, color=RED, ha="center")

r30 = Rres["0.30"]
for b in range(4):
    if r30[b]:
        lo, hi = r30[b][0], r30[b][1] if r30[b][1] else r30[b][0] * 1.6
        ax3.plot([b, b], [lo, hi], color=BLUE, lw=5, alpha=0.75)
if ctrl_med: ax3.scatter([hi_b + 0.18], [ctrl_med], marker="D", color=GOLD, zorder=5, label=f"Q{hi_b+1} 유병률통제 후")
ax3.set_xticks(xs, [f"Q{b+1}" for b in xs]); ax3.set_ylabel("함의 R (검사율 30%)")
ax3.set_title("(e)(f) 구간별 역산 R + 반사실", fontsize=10.5); ax3.legend(fontsize=8.5)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(ROOT, "figures", "fig_drift_v2.png"), bbox_inches="tight")
print("→ figures/fig_drift_v2.png")
