# -*- coding: utf-8 -*-
"""이상탐지 결과 그림 (리뷰어 17차 재작업).
 (A) 손익분기=1차모델 한계정밀도. A(IForest)·B(임계내리기) 달성 + Wilson CI → '판정 불가'가 그림으로.
 (B) cl축 절감곡선 3행(지도/IForest/무작위) → 무작위 기준선 위 IForest 순정보."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "anomaly.json"), encoding="utf-8"))
B, A = d["beta"], d["alpha"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2))

# ── (A) 손익분기 정밀도 = 판정 기준 + Wilson CI ──
cls = np.arange(10, 105)
be = 1.0 / (B*cls)                         # cs=0 → (1+μ)/(λ+μ)=1/λ
axL.plot(cls, be*100, color="#c0392b", lw=2.4, zorder=5,
         label="손익분기 = 1차모델 한계정밀도  (1+μ)/(λ+μ)")
p1 = d["part1"]; xs = sorted(int(k) for k in p1)
precA = np.array([p1[str(c)]["precA"]*100 for c in xs])
precB = np.array([p1[str(c)]["precB"]*100 for c in xs])
loA = np.array([p1[str(c)]["wilsonA"][0]*100 for c in xs])
hiA = np.array([p1[str(c)]["wilsonA"][1]*100 for c in xs])
YTOP = 15.0
yerr = np.clip([precA-loA, np.minimum(hiA, YTOP)-precA], 0, None)   # 상한 초과분은 화살표로
axL.errorbar(xs, precA, yerr=yerr, fmt="s", color="#2c7fb8", ms=8, capsize=5,
             lw=1.8, label="A: IForest 상위5% 정밀도 [Wilson 95%]", zorder=6)
for c, hi in zip(xs, hiA):                                          # CI 상한 초과 표시
    if hi > YTOP: axL.annotate("", (c, YTOP-0.3), (c, YTOP-1.6), arrowprops=dict(arrowstyle="-|>", color="#2c7fb8"))
axL.plot(xs, precB, "D--", color="#7a5195", ms=7, lw=1.6,
         label="B: 지도점수 상위5% (=1차 임계값 더 내리기)", zorder=6)
for c in xs:
    m = round(p1[str(c)]["m5"]); dd = round(p1[str(c)]["dA"])
    axL.annotate(f"m={m}·d={dd}", (c, precA[xs.index(c)]), textcoords="offset points",
                 xytext=(6, 7), ha="left", fontsize=8, color="#666")
axL.set_xlabel("유출:검사 비용비  cl  (λ=β·cl)"); axL.set_ylabel("정밀도 (%)")
axL.set_title("① 2차 필터 손익분기 = 1차모델 한계정밀도 (1+μ)/(Λ+μ)", fontsize=11, fontweight="bold")
axL.set_xlim(8, 112); axL.set_ylim(0, YTOP); axL.legend(fontsize=8.3, loc="upper right"); axL.grid(alpha=0.25)
axL.text(0.97, 0.60, "⭐ B(보라)가 손익분기선 위에 앉음\n= 항등식 데이터 확증(6.4 vs 7.0, 4.3 vs 3.5).\n"
         "A(IForest)는 CI가 손익분기선 감쌈\n→ 전 cl 판정불가. 유일점 cl=15서 B가\n6:2 우세 → 직교정보 증거 없음",
         transform=axL.transAxes, fontsize=8.0, color="#333", va="top", ha="right",
         bbox=dict(boxstyle="round", fc="#fbf6ec", ec="#e0d3b8"))

# ── (B) cl축 절감곡선 3행: 지도 / IForest / 무작위 ──
p2 = d["part2"]; g = np.array(p2["cl_grid"])
axR.axhline(0, color="#888", lw=1.0)
axR.fill_between(g, p2["curve_floor"], 0, color="#cfcfcf", alpha=0.5, label="무료 기준선 = max(전수통과, 전수검사)")
axR.plot(g, p2["curve_floor"], "-", color="#666666", lw=1.6)
axR.plot(g, p2["curve_sup"], "-", color="#2c7fb8", lw=2.4, label=f"지도 LGBM (AUC {d['sup_auc']:.3f})")
axR.plot(g, p2["curve_ano"], "-", color="#e08214", lw=2.2, label=f"IForest 비지도 (AUC {d['ano_auc']:.3f})")
axR.plot(g, p2["curve_rnd"], "--", color="#999999", lw=1.8, label="무작위 점수")
axR.set_xlabel("유출:검사 비용비  cl"); axR.set_ylabel("sel_vs_inspect 절감률 (%)")
axR.set_title("② 라벨의 가치 — 전수통과 기준선 대비 순정보", fontsize=11, fontweight="bold")
axR.legend(fontsize=8.6, loc="upper right"); axR.grid(alpha=0.25)
axR.text(0.02, 0.03, "저cl 80% 수렴·무작위 절감 = 무료기준선(회색)\n"
         "효과일 뿐. 무작위는 회색보다도 낮음.\n"
         "순정보(폴백)=max(0,곡선-회색). 지도만 큰 우위=라벨 가치",
         transform=axR.transAxes, fontsize=8.2, color="#333", va="bottom",
         bbox=dict(boxstyle="round", fc="#eef4fb", ec="#bcd4ea"))

p3 = d["part3"]
fig.suptitle("아이디어 2 — 이상탐지: 손익분기=1차모델 한계정밀도, 대조는 '임계 더 내리기' | "
             f"③Q4 novelty 판별불가(n={p3['n_q4_pos']})",
             fontsize=12, fontweight="bold", y=1.02)
p = os.path.join(ROOT, "figures", "fig_anomaly.png")
plt.savefig(p, bbox_inches="tight"); plt.close()
print(f"→ {p}")
