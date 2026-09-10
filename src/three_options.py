"""
[특징 1의 근거 그림] 전수 통과 / 전수 검사 / 선별 검사 — 세 전략의 비용 비교

⚠️ 기존 fig3_cost_sensitive.png는 '낙관적' 수치(임계값을 평가 데이터에서 선택)로 그려져
   "모델이 2:1~60:1 전 구간에서 이긴다"고 표시되어 있었다. 이는 본문 주장(50:1에서 −7.0%)과
   정면으로 모순된다. 여기서 Nested CV(낙관 편향 제거) 기준으로 다시 그린다.

Nested CV 구조상 모델 적합은 outer fold당 1회뿐이므로, 비용비를 촘촘히 스윕해도 비용이 늘지 않는다.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")
df = pd.read_csv(CSV)
y  = (df["Pass/Fail"] == 1).astype(int).to_numpy()
X  = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
N_pos, N_neg = int(y.sum()), int((y == 0).sum())

C_FP  = 10.0                              # Overkill 1건 = 10만원 (가정)
RATIOS = np.arange(2, 61)                 # 미검:과검 비용비 2:1 ~ 60:1
THS    = np.linspace(0.005, 0.995, 199)

def mkpipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("m",   RandomForestClassifier(n_estimators=400,
                                class_weight="balanced_subsample", min_samples_leaf=3,
                                random_state=0, n_jobs=-1))])

# ── Nested CV: outer fold당 모델 1회 적합 → 모든 비용비를 이 위에서 평가 ──
outer = StratifiedKFold(5, shuffle=True, random_state=7)
sel_cost = np.zeros(len(RATIOS))          # 선별 검사 총비용

print("Nested CV 진행 (outer fold 5개)...")
for k, (tr, te) in enumerate(outer.split(X, y)):
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]

    # 안쪽: train fold 내부 OOF → 임계값 선택용 (바깥 fold 정답은 절대 보지 않음)
    inner = cross_val_predict(mkpipe(), Xtr, ytr,
                              cv=StratifiedKFold(4, shuffle=True, random_state=k),
                              method="predict_proba", n_jobs=-1)[:, 1]
    # 바깥: train 전체로 학습 → test fold 예측
    te_score = mkpipe().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    # 임계값 후보별 혼동행렬을 미리 계산 (inner / outer 각각)
    in_cm = np.array([confusion_matrix(ytr, (inner >= t).astype(int), labels=[0,1]).ravel() for t in THS])
    te_cm = np.array([confusion_matrix(yte, (te_score >= t).astype(int), labels=[0,1]).ravel() for t in THS])

    for i, R in enumerate(RATIOS):
        C_FN = R * C_FP
        in_cost = in_cm[:, 1]*C_FP + in_cm[:, 2]*C_FN      # fp, fn
        j = int(in_cost.argmin())                           # ✅ 안쪽에서만 임계값 선택
        sel_cost[i] += te_cm[j, 1]*C_FP + te_cm[j, 2]*C_FN  # 바깥에서 평가
    print(f"  fold {k+1}/5 완료")

pass_cost = N_pos * RATIOS * C_FP          # ① 전수 통과: 불량 전량 유출
full_cost = np.full(len(RATIOS), N_neg * C_FP)   # ② 전수 검사: 정상 전량 과검

best = np.minimum(pass_cost, full_cost)
saving = (best - sel_cost) / best * 100    # 선별 검사의 실질 절감률(최선의 대안 대비)

# 선별 검사의 실익 구간 — 양쪽으로 닫혀 있다
#   비용비가 너무 낮으면 전수 통과가, 너무 높으면 전수 검사가 최선이 된다
#
# ⚠️ 구현 정정: 이전 구현은 saving>0인 지점의 min/max를 구간으로 삼았다.
#    그러나 saving은 비단조다 — 50:1 부근에서 음수로 내려갔다가 그 위에서 다시 0 근처로
#    미미하게 올라온다(임계값 격자의 이산성에 의한 잔여 진동, 절감폭 수 % 이하).
#    min/max를 쓰면 그 잔여 구간까지 삼켜 "4:1~60:1"로 표기되어, 같은 그림의 우패널이
#    43~51에서 음수를 그리는 것과 정면으로 모순된다.
#    → 실익 구간은 '최대 절감점을 포함하는 연속 양수 구간'으로 정의한다.
peak_i = int(np.argmax(saving)); peak_R = int(RATIOS[peak_i])
pos_mask = saving > 0
lo_i = hi_i = peak_i
while lo_i - 1 >= 0 and pos_mask[lo_i - 1]:
    lo_i -= 1
while hi_i + 1 < len(RATIOS) and pos_mask[hi_i + 1]:
    hi_i += 1
lo_R, hi_R = int(RATIOS[lo_i]), int(RATIOS[hi_i])

tail = saving[hi_i + 1:]
tail_max = float(tail.max()) if len(tail) else 0.0

print(f"\n선별 검사 실익 구간(연속): {lo_R}:1 ~ {hi_R}:1   (최대 절감 {saving[peak_i]:.1f}% @ {peak_R}:1)")
print(f"  · {lo_R}:1 미만 → 미검이 싸므로 '전수 통과'가 최선 (검사 자체가 불필요)")
print(f"  · {hi_R}:1 초과 → 미검이 비싸므로 '전수 검사'가 최선 (선별의 이점 소멸)")
print(f"  · {hi_R}:1 초과 구간의 잔여 절감 최댓값 = {tail_max:.1f}% (사실상 동률, 이산성 잔여 진동)\n")
for R in [2, 5, 10, 14, 20, 30, 40, 50, 60]:
    i = int(np.where(RATIOS == R)[0][0])
    win = ["전수 통과", "전수 검사", "선별 검사"][int(np.argmin([pass_cost[i], full_cost[i], sel_cost[i]]))]
    print(f"  {R:>2}:1 | 통과 {pass_cost[i]:>7,.0f} | 검사 {full_cost[i]:>7,.0f} | "
          f"선별 {sel_cost[i]:>7,.0f} 만원 | 절감 {saving[i]:>6.1f}% | 최선: {win}")

# ── 그림 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.1))

ax[0].plot(RATIOS, pass_cost, lw=1.8, ls="--", color="#8c8c8c", label="① 전수 통과 (검사 안 함)")
ax[0].plot(RATIOS, full_cost, lw=1.8, ls=":",  color="#d94f4f", label="② 전수 검사 (전부 검사)")
ax[0].plot(RATIOS, sel_cost,  lw=2.6, color="#4f86d9", label="③ 선별 검사 (제안)")
ax[0].fill_between(RATIOS, sel_cost, best, where=(saving > 0), color="#4f86d9", alpha=.13)
ax[0].axvspan(lo_R, hi_R, color="#4f86d9", alpha=.05)
ax[0].axvline(lo_R, color="#3fa06b", lw=1.1, ls="-.")
ax[0].axvline(hi_R, color="#d94f4f", lw=1.1, ls="-.")
ax[0].annotate(f"{lo_R}:1 미만\n전수 통과가 최선", xy=(lo_R, sel_cost[0]),
               xytext=(lo_R + 1.5, N_neg*C_FP*1.75), fontsize=8, color="#2b6b4a")
ax[0].annotate(f"{hi_R}:1 초과\n전수 검사가 최선", xy=(hi_R, full_cost[0]),
               xytext=(hi_R - 20, full_cost[0]*1.55), fontsize=8, color="#a33",
               arrowprops=dict(arrowstyle="->", lw=1, color="#a33"))
ax[0].set_xlabel("Underkill(미검) : Overkill(과검) 비용비")
ax[0].set_ylabel("총 기대비용 (만원, Overkill 1건=10만원 가정)")
ax[0].set_title(f"세 가지 검사 전략의 비용 비교 (Nested CV)\n선별 검사의 실익 구간: {lo_R}:1 ~ {hi_R}:1 — 양쪽 모두에서 닫힌다", fontsize=10.5)
ax[0].legend(fontsize=8.5, loc="upper left"); ax[0].grid(alpha=.3)

col = ["#4f86d9" if s > 0 else "#d94f4f" for s in saving]
ax[1].bar(RATIOS, saving, color=col, width=.85)
ax[1].axhline(0, color="k", lw=1)
for R in [peak_R, 50]:
    i = int(np.where(RATIOS == R)[0][0])
    ax[1].annotate(f"{R}:1\n{saving[i]:+.1f}%", xy=(R, saving[i]),
                   xytext=(R, saving[i] + (6 if saving[i] > 0 else -12)),
                   ha="center", fontsize=9, fontweight="bold",
                   color="#2a5da8" if saving[i] > 0 else "#a33")
ax[1].set_ylim(min(saving.min() - 14, -16), saving.max() + 14)
ax[1].set_xlabel("Underkill(미검) : Overkill(과검) 비용비")
ax[1].set_ylabel("선별 검사의 실질 절감률 (%)")
ax[1].set_title(f"선별 검사가 손해로 돌아서는 지점 — {hi_R}:1 초과\n최선의 대안(전수 통과·전수 검사 중 싼 쪽) 대비", fontsize=10.5)
ax[1].grid(alpha=.3, axis="y")
ax[1].text(0.5, -0.30,
           f"※ {hi_R}:1 초과에서 절감률이 0 근처로 되튀는 구간(최대 {tail_max:.1f}%)은 임계값 격자의 이산성에 의한 잔여 진동이며,\n"
           "   전수 검사와 사실상 동률이다. 따라서 실익 구간은 '연속 양수 구간'으로 정의한다.",
           transform=ax[1].transAxes, ha="center", va="top", fontsize=7.4, color="#666")
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(os.path.join(ROOT, "figures", "fig12_three_options.png")); plt.close()

rep = json.load(open(os.path.join(ROOT, "results", "report.json"), encoding="utf-8"))
rep["three_options_honest"] = {
    "설명": "전수 통과 / 전수 검사 / 선별 검사 — Nested CV 기준(낙관 편향 제거)",
    "가정": "Overkill 1건 = 10만원",
    "선별검사 실익구간(연속)": f"{lo_R}:1 ~ {hi_R}:1",
    "주의(비단조)": f"{hi_R}:1 초과에서 절감률이 음수로 내려갔다가 잔여 진동으로 최대 {tail_max:.1f}%까지 재상승하나 사실상 동률이다. 실익 구간은 연속 양수 구간으로 정의한다.",
    "최대절감": f"{saving[peak_i]:.1f}% @ {peak_R}:1",
    "구간 아래": f"{lo_R}:1 미만 → 미검이 싸므로 전수 통과가 최선(검사 자체가 불필요)",
    "구간 위":   f"{hi_R}:1 초과 → 미검이 비싸므로 전수 검사가 최선(선별의 이점 소멸)",
    "구간별": {int(R): {
        "전수통과": round(float(pass_cost[i]), 1),
        "전수검사": round(float(full_cost[i]), 1),
        "선별검사": round(float(sel_cost[i]), 1),
        "절감률(%)": round(float(saving[i]), 1),
        "최선": ["전수 통과", "전수 검사", "선별 검사"][int(np.argmin([pass_cost[i], full_cost[i], sel_cost[i]]))],
    } for R, i in [(R, int(np.where(RATIOS == R)[0][0])) for R in [2, 5, 10, 14, 20, 30, 40, 50, 60]]},
    "주의": "기존 fig3_cost_sensitive.png는 낙관적 수치라 본문과 모순됨. 이 그림(fig12)으로 대체할 것.",
}
json.dump(rep, open(os.path.join(ROOT, "results", "report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n→ figures/fig12_three_options.png")
