# -*- coding: utf-8 -*-
"""render_oof_shap_v2.py — OOF SHAP v2 **render-only** 시각화.

정본만 읽는다: run_manifest.json(계보 게이트) → aggregate.json/npz · groups.json/npz.
모델 fit·SHAP·CV·상관 재계산 없음(lightgbm/shap/sklearn 미import). 새로운 통계 판정 생성 없음 —
저장된 값의 표시만. 출력: figures/fig_oof_shap_summary.png · figures/fig_oof_group_summary.png
+ results/oof_shap_v2/render_manifest.json (사용 source·산출 hash 기록).

표현 고정: "반복 관측"·"고상관 짝이 확인되지 않았다"·"우선 모니터링·DOE 후보"만.
금지: 불량 원인·인과·공정 신호 검증·통계적 독립·시간대리 아님·PM/제어 확정.
"""
import os, sys, json, hashlib, datetime
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "oof_shap_v2")
FIG = os.path.join(ROOT, "figures")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

# ── 계보 게이트: manifest ↔ aggregate/groups 불일치 시 즉시 중단 ──
M = json.load(open(os.path.join(OUT, "run_manifest.json"), encoding="utf-8"))
SRC = {f: os.path.join(OUT, f) for f in
       ("aggregate.json", "aggregate_arrays.npz", "groups.json", "groups_contrib.npz")}
for key, f in (("aggregate_sha256", "aggregate.json"), ("aggregate_arrays_sha256", "aggregate_arrays.npz"),
               ("groups_sha256", "groups.json"), ("groups_contrib_sha256", "groups_contrib.npz")):
    if M[key] != sha(SRC[f]):
        print(f"🔴 중단: {f} 해시가 manifest와 불일치 — render 금지"); sys.exit(1)
print("✅ 계보 게이트 통과 (manifest ↔ aggregate/groups)")

A = json.load(open(SRC["aggregate.json"], encoding="utf-8"))
G = json.load(open(SRC["groups.json"], encoding="utf-8"))
Z = np.load(SRC["aggregate_arrays.npz"])
names = Z["feature_names"].tolist()

CAVEAT = ("예측 기여도 결과이며 인과관계를 의미하지 않습니다. "
          "시간·배치 대리 가능성에 대한 추가 공정 검증이 필요합니다.")

# ══════════ 그림 1 — OOF SHAP 상위 10 + S59 안정성 + Jaccard ══════════
mean_all = Z["mean_abs_all_mean"]
order = np.argsort(-mean_all, kind="stable")[:10][::-1]
top_names = [f"S{names[i]}" for i in order]; top_vals = mean_all[order]
s59_stats = A["sensor_stats"]["59"]
jac = A["jaccard_seed_분포"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
cols = ["#d94f4f" if n == "S59" else "#c7ced6" for n in top_names]
axL.barh(top_names, top_vals, color=cols)
axL.set_xlabel("OOF mean |SHAP| (validation 전용, 20 seed 등가중 평균)")
axL.set_title("LightGBM OOF 예측 기여도 — 상위 10", fontsize=11.5, fontweight="bold")
axL.text(0.97, 0.06, f"S59: {int(s59_stats['top1_rate']*20)}/20 seed 1위 · 전체·양성·음성 집계 모두 1위",
         transform=axL.transAxes, ha="right", color="#d94f4f", fontsize=9, fontweight="bold")
ks = ["1", "3", "5", "10", "20"]
axR.boxplot([jac[k] for k in ks], tick_labels=[f"top-{k}" for k in ks], widths=0.5)
axR.set_ylim(0, 1.05); axR.set_ylabel("seed 간 Jaccard (190쌍 분포)")
axR.set_title("Top-k 집합 안정성", fontsize=11.5, fontweight="bold")
axR.text(0.03, 0.06, "top-1은 전 쌍 1.0(=S59 고정)\ntop-10 중앙값 부근 0.7(집합은 흔들림)",
         transform=axR.transAxes, fontsize=8.4, color="#555")
fig.suptitle("OOF SHAP 예측 기여도 순위 안정성 (미관측 validation 데이터, 20 seed × 5-fold)",
             fontsize=12.5, fontweight="bold", y=1.045)
fig.text(0.5, 0.985, CAVEAT, ha="center", fontsize=9.3, color="#8a4a4a")
fig.text(0.5, 0.005, "render-only: results/oof_shap_v2 정본에서만 렌더링 · OOF AUC는 QC 전용 · 우선 모니터링·DOE 후보 제시 목적",
         ha="center", fontsize=8, color="#666")
p1 = os.path.join(FIG, "fig_oof_shap_summary.png")
plt.savefig(p1, bbox_inches="tight"); plt.close()
print(f"→ {p1}")

# ══════════ 그림 2 — 센서군 기여도 (CL 0.90 상위 5) ══════════
cfg = G["configs"]["complete_linkage_combined_0.9"]
st = cfg["contrib"]; ids = st["all"]["cluster_ids"]
byid = {c["cluster_id"]: c for c in cfg["clusters"]}
top5 = st["display_top30_cluster_ids"][:5]
CZ = np.load(SRC["groups_contrib.npz"])
A_m = CZ["CL|0.9|all|A"].mean(1); B_m = CZ["CL|0.9|all|B"].mean(1)
C_m = np.nanmean(np.where(CZ["CL|0.9|all|B"] > 0, CZ["CL|0.9|all|C"], np.nan), 1)

labels, Av, Bv, Cv = [], [], [], []
for cid in top5:
    c = byid[cid]; i = ids.index(cid)
    nm = ("S" + c["member_names"][0] + " (단독)") if c["size"] == 1 else \
         ("·".join("S" + m for m in c["member_names"][:3]) + (f" 외 {c['size']-3}개" if c["size"] > 3 else "") + f" ({c['size']}개 군)")
    labels.append(nm); Av.append(A_m[i]); Bv.append(B_m[i]); Cv.append(C_m[i])

fig, ax = plt.subplots(figsize=(10.5, 5.0))
ypos = np.arange(len(top5))[::-1]
ax.barh(ypos + 0.2, Bv, height=0.38, color="#c7ced6")
ax.barh(ypos - 0.2, Av, height=0.38,
        color=["#d94f4f" if l.startswith("S59") else "#4f86d9" for l in labels])
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9.5)
for y, a, b, cv in zip(ypos, Av, Bv, Cv):
    ax.text(max(a, b) + 0.004, y, f"내부 상쇄 지표 C={cv:.2f}", va="center", fontsize=8.2, color="#666")
ax.set_xlabel("OOF SHAP 기여도 (20 seed 평균, 전체 행)")
ax.set_title("고상관 센서군 기여도 — combined complete-linkage 0.90 · 상위 5\n"
             "(S59는 단독 피처로 1위 — 시험한 8개 군집 설정 전부에서 고상관 짝이 확인되지 않음)",
             fontsize=11.5, fontweight="bold")
from matplotlib.patches import Patch                       # 범례 = 실제 색상과 일치 (UI fix)
ax.legend(handles=[Patch(fc="#c7ced6", label="B 총 배분 기여량 (군 내 |SHAP| 합)"),
                   Patch(fc="#4f86d9", label="A 순 기여 크기 (additive)"),
                   Patch(fc="#d94f4f", label="S59 강조 (A 막대)")],
          fontsize=8.5, loc="lower right")
fig.text(0.5, 0.005, CAVEAT + "  |  A·B는 서로 다른 정의(혼합 금지) · C=A/B(1에 가까울수록 내부 상쇄 작음)",
         ha="center", fontsize=8, color="#666")
p2 = os.path.join(FIG, "fig_oof_group_summary.png")
plt.savefig(p2, bbox_inches="tight"); plt.close()
print(f"→ {p2}")

# ══════════ render manifest ══════════
rm = {"rendered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
      "render_code_sha256": sha(os.path.abspath(__file__)),
      "run_manifest_sha256": sha(os.path.join(OUT, "run_manifest.json")),
      "sources": {f: sha(p) for f, p in SRC.items()},
      "outputs": {"fig_oof_shap_summary.png": sha(p1), "fig_oof_group_summary.png": sha(p2)},
      "성격": "render-only — 계산·판정 없음, 저장값 표시만"}
rp = os.path.join(OUT, "render_manifest.json")
tmp = rp + ".tmp"
json.dump(rm, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(tmp, rp)
print(f"→ {rp}")
