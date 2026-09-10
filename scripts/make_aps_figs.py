# -*- coding: utf-8 -*-
"""APS 일반화 발표 그림 — 기존 동결 결과만 사용(재학습·재설계 없음).
입력: results/aps_result.json, results/aps_multiseed.json, results/nested_multiseed.json, web/sim_final.json
출력: figures/figI_secom_vs_aps.{png,svg} · figJ_winrate.{png,svg} · figK_costlayer.{png,svg}
      + data_secom_vs_aps.csv · data_winrate.csv · APS_일반화_수치정의.md"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 프로젝트 기준 상대경로: 이 스크립트는 submission/scripts/ 에 위치.
SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # …/submission
OUT = os.path.join(SUB, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, BLUE_D, ORANGE, RED, GRAY, INK, INK2, GRID = \
    "#2a78d6", "#1c5cab", "#eb6834", "#d03b3b", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 300,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
})

aps = json.load(open(os.path.join(SUB, "results", "aps_result.json"), encoding="utf-8"))
apsm = json.load(open(os.path.join(SUB, "results", "aps_multiseed.json"), encoding="utf-8"))
nm = json.load(open(os.path.join(SUB, "results", "nested_multiseed.json"), encoding="utf-8"))
sim = json.load(open(os.path.join(SUB, "web", "sim_final.json"), encoding="utf-8"))

# ── APS 50:1 유도값 (동결 파일에서) ──
npos, nrow, C = aps["n_pos"], aps["n_row"], aps["cost_C_FP"]
nneg = nrow - npos
sel50 = aps["nested_cv"]["50"]["cost"]
insp = nneg * C
aps_sel_vs_inspect_50 = (insp - sel50) / insp * 100        # 전수검사 대비 = 92.9
aps_sel_vs_floor_50 = aps["nested_cv"]["50"]["save_pct"]    # 무료기준선 대비 = 91.6 (파일 저장값)

# ── SECOM 동결값 (sim_final = cl=15 sel_vs_inspect 31.0) ──
g = sim["policy"]["grid"]["14_0"]["c"]; tn, fp, fn, tp = g
Ns, nps, nns = sim["meta"]["N"], sim["meta"]["npos"], sim["meta"]["nneg"]
cl, beta = 15, sim["meta"]["beta"]; lam = beta * cl
inspAll = Ns + nps * cl - nps * lam
selS = (tp + fp) + nps * cl - tp * lam
secom_sel_vs_inspect = (inspAll - selS) / inspAll * 100     # = 31.0

# ═══ 그림 A — SECOM vs APS 스펙 비교 표 ═══
secom_win50 = None  # 아래 winrate 계산 후 채움 (도식 순서상 재배치)
rows = [
    ("표본 수 / 양성",      f"{Ns:,} / {nps} ({nps/Ns*100:.2f}%)",  f"{nrow:,} / {npos:,} ({npos/nrow*100:.2f}%)"),
    ("OOF ROC-AUC",       "0.731",                f"{aps['oof_auc']:.4f}"),
    ("대표 비용조건",       "15 : 1 (사례)",         "50 : 1 (데이터셋 공식)"),
    ("최적 전략",          "선별검사",              "선별검사"),
    ("절감률 (전수검사 대비)", f"{secom_sel_vs_inspect:.1f}%",  f"{aps_sel_vs_inspect_50:.1f}%"),
    ("50:1 선별 승률",      "70%",                  "100%"),
]
fig, ax = plt.subplots(figsize=(8.4, 4.5)); ax.axis("off")
ncol = [0.36, 0.63, 0.895]
ax.text(ncol[1], 0.96, "SECOM", ha="center", fontsize=13, fontweight="bold", color=BLUE_D)
ax.text(ncol[2], 0.96, "APS", ha="center", fontsize=13, fontweight="bold", color=ORANGE)
ax.text(0.03, 0.96, "항목", ha="left", fontsize=11, fontweight="bold", color=INK2)
ax.plot([0.02, 0.98], [0.905, 0.905], color=INK2, lw=1.1)
for i, (k, a, b) in enumerate(rows):
    y = 0.85 - i * 0.14
    hl = (k == "최적 전략" or k.startswith("절감률"))
    if hl:
        ax.add_patch(plt.Rectangle((0.02, y - 0.045), 0.96, 0.11, transform=ax.transAxes,
                     facecolor="#eef4fb", edgecolor="none", zorder=0))
    ax.text(0.03, y, k, ha="left", va="center", fontsize=10.5, color=INK2)
    ax.text(ncol[1], y, a, ha="center", va="center", fontsize=11,
            fontweight="bold" if hl else "normal", color=BLUE_D if hl else INK)
    ax.text(ncol[2], y, b, ha="center", va="center", fontsize=11,
            fontweight="bold" if hl else "normal", color=ORANGE if hl else INK)
    ax.plot([0.02, 0.98], [y - 0.055, y - 0.055], color=GRID, lw=0.6)
ax.set_title("SECOM vs APS — 데이터·예측·검사정책 비교", pad=16)
fig.text(0.5, 0.01, "절감률은 전수검사 대비(sel_vs_inspect)로 통일 · 각 데이터의 대표 비용조건 기준 · "
         "APS 파일 저장값 91.6%는 무료기준선 대비(sel_vs_floor, 별도 정의)",
         ha="center", fontsize=7.8, color=INK2)
fig.savefig(os.path.join(OUT, "figI_secom_vs_aps.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figI_secom_vs_aps.svg"), bbox_inches="tight"); plt.close(fig)

# ═══ 그림 B — 비용비별 선별 승률 (SECOM vs APS) ═══
# SECOM: nested_multiseed lgbm(챔피언) — 각 ratio에서 코드2(선별) 비율
sr = nm["ratios"]
lg = np.array([nm["lgbm"][str(s)] for s in range(len(nm["seeds"]))])   # (20, 59)
secom_win = (lg == 2).mean(axis=0) * 100
ar = apsm["ratios"]; aps_win = np.array(apsm["aps_select_winrate"], float)

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.grid(color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.plot(ar, aps_win, "-", color=ORANGE, lw=2.4, label="APS (양성 1,000 · AUC 0.987)")
ax.plot(ar, aps_win, "o", color=ORANGE, ms=3, mew=0)
ax.plot(sr, secom_win, "-", color=BLUE, lw=2.4, label="SECOM (양성 104 · AUC 0.731)")
ax.plot(sr, secom_win, "o", color=BLUE, ms=3, mew=0)
ax.axvline(50, ls="--", lw=1.1, color=GRAY)
ax.text(50, 12, "50:1\n(APS 공식)", ha="center", fontsize=8.5, color=INK2)
ax.set_xlim(2, 60); ax.set_ylim(-3, 106)
ax.set_xlabel("비용비  (미검출 비용 / 과검 비용)")
ax.set_ylabel("20 seed 중 선별검사가 최적인 비율 (%)")
ax.set_title("비용비에 따른 선별 전략 선택 안정성")
ax.legend(loc="center right", fontsize=9.5, frameon=False)
fig.text(0.5, -0.02, "양성 표본이 더 많고 예측 신호가 강한 APS에서는 고비용 구간에서도 선별 전략 선택이 안정적으로 유지됐습니다.",
         ha="center", fontsize=9, color=INK, fontweight="bold")
fig.text(0.5, -0.075, "AUC는 5-fold OOF, 비용 임계값은 nested 평가 — 동일 Cost Layer 구조의 적용 가능성 확인 · 실제 계산 지점만 표시",
         ha="center", fontsize=7.6, color=INK2)
fig.savefig(os.path.join(OUT, "figJ_winrate.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figJ_winrate.svg"), bbox_inches="tight"); plt.close(fig)

# ═══ 그림 C — Cost Layer 이식 구조도 ═══
fig, ax = plt.subplots(figsize=(9.0, 3.9)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, txt, fc, ec, fs=10, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=fc, edgecolor=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, c=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                 color=c, lw=1.6, shrinkA=2, shrinkB=2))
box(0.2, 3.3, 1.9, 1.0, "SECOM 데이터", "#eef4fb", BLUE, tc=BLUE_D)
box(0.2, 0.7, 1.9, 1.0, "APS 데이터", "#fdf0e9", ORANGE, tc="#c2521f")
box(2.7, 3.3, 2.2, 1.0, "SECOM 전용 모델\n전처리·임계값", "#fff", BLUE)
box(2.7, 0.7, 2.2, 1.0, "APS 전용 모델\n전처리·임계값", "#fff", ORANGE)
box(5.6, 1.85, 2.1, 1.3, "동일 Cost Layer\n(수식·구조)", "#f1f3f6", INK2, fs=10.5, bold=True)
box(8.1, 1.95, 1.75, 1.1, "최적\n검사정책", "#eef7ee", "#0ca30c", tc="#0a7a0a", bold=True)
arrow(2.1, 3.8, 2.7, 3.8, BLUE); arrow(2.1, 1.2, 2.7, 1.2, ORANGE)
arrow(4.9, 3.8, 5.6, 2.9, BLUE); arrow(4.9, 1.2, 5.6, 2.1, ORANGE)
arrow(7.7, 2.5, 8.1, 2.5)
ax.text(6.65, 1.55, "위험점수 → 임계값별 혼동행렬\n→ 3전략 비용 → 최소비용 선택",
        ha="center", va="top", fontsize=7.8, color=INK2)
ax.set_title("Cost Layer 이식 구조 — 데이터·모델은 별도, 의사결정 계층은 동일", pad=6)
fig.text(0.5, 0.0, "데이터와 예측 모델이 달라도, 위험점수를 비용조건별 검사정책으로 변환하는 Cost Layer는 동일하게 적용할 수 있었습니다.",
         ha="center", fontsize=9, color=INK, fontweight="bold")
fig.savefig(os.path.join(OUT, "figK_costlayer.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figK_costlayer.svg"), bbox_inches="tight"); plt.close(fig)

# ═══ 원본 CSV ═══
with open(os.path.join(OUT, "data_secom_vs_aps.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["metric", "SECOM", "APS", "source"])
    w.writerow(["n_row", Ns, nrow, "sim_final.meta / aps_result"])
    w.writerow(["n_pos", nps, npos, ""])
    w.writerow(["pos_rate_%", round(nps/Ns*100, 2), round(npos/nrow*100, 2), ""])
    w.writerow(["oof_auc", 0.731, aps["oof_auc"],
                "SECOM: repeated nested OOF / APS: 5-fold OOF, fold 내부 전처리 (비용 임계값만 nested)"])
    w.writerow(["representative_condition", "15:1(사례)", "50:1(데이터셋 공식)", "대표 비용조건"])
    w.writerow(["best_strategy", "선별검사", aps["nested_cv"]["50"]["strategy"], ""])
    w.writerow(["save_sel_vs_inspect_%", round(secom_sel_vs_inspect, 1), round(aps_sel_vs_inspect_50, 1), "전수검사 대비"])
    w.writerow(["save_sel_vs_floor_%(APS파일저장값)", 27.2, aps_sel_vs_floor_50, "무료기준선 대비"])
    w.writerow(["select_winrate_50to1_%", 70, 100, "nested_multiseed / aps_multiseed"])
    w.writerow(["detect_%@50to1", "-", aps["nested_cv"]["50"]["detect_pct"], "aps_result"])
    w.writerow(["overkill_%@50to1", "-", aps["nested_cv"]["50"]["overkill_pct"], "aps_result"])
    w.writerow(["th_mean@50to1", "-", aps["nested_cv"]["50"]["th_mean"], "aps_result"])
with open(os.path.join(OUT, "data_winrate.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["cost_ratio", "SECOM_select_winrate_%", "APS_select_winrate_%"])
    sd = {r: v for r, v in zip(sr, secom_win)}
    ad = {r: v for r, v in zip(ar, aps_win)}
    for r in sorted(set(sr) | set(ar)):
        w.writerow([r, round(sd.get(r, ""), 1) if r in sd else "", ad.get(r, "")])

print("figures:", [f for f in sorted(os.listdir(OUT)) if f.startswith(("figI", "figJ", "figK"))])
print(f"SECOM sel_vs_inspect(cl=15) = {secom_sel_vs_inspect:.1f}%  (헤드라인 31.0 대조)")
print(f"APS  sel_vs_inspect(50:1)   = {aps_sel_vs_inspect_50:.1f}%")
print(f"APS  sel_vs_floor(50:1)파일  = {aps_sel_vs_floor_50}%")
print(f"SECOM 선별승률 @2/@50/@60 = {secom_win[0]:.0f}/{secom_win[sr.index(50)]:.0f}/{secom_win[-1]:.0f}%  ·  APS = 100/100/100%")

# ═══ 수치정의 Markdown (동결 파일 값으로 생성) ═══
a50 = aps["nested_cv"]["50"]
noinsp = npos * 50 * C
md = f"""# APS 일반화 분석 — 최종 동결 (수치 정의·출처·발표 확정값)

> 현재 산출물 기준 **최종 동결**. 추가 재학습·분석 설계 변경 없음.
> 입력: `data_aps/`(원본) · `results/aps_result.json` · `results/aps_multiseed.json` · `results/nested_multiseed.json` · `web/sim_final.json`
> 생성: `scripts/make_aps_figs.py` (동결 파일만 읽음 · 재학습 없음)

## 1. 비교 지표 통일 — 전수검사 대비 절감률

발표 본문은 **"전수검사 대비 절감률"(sel_vs_inspect)로 통일**한다.

| 데이터 | 대표 비용조건 | 전수검사 대비 절감률 |
|---|---|---|
| SECOM | 15 : 1 (사례) | **{secom_sel_vs_inspect:.1f}%** |
| APS | 50 : 1 (데이터셋 공식) | **{aps_sel_vs_inspect_50:.1f}%** |

- **{aps_sel_vs_floor_50}%**(`aps_result.json` save_pct)는 선별검사와 더 저렴한 기준선(50:1에서는 무검사)을 비교한 **sel_vs_floor(최저비용 기준선 대비)**이므로 **헤드라인 사용 금지**. 필요 시 각주로만 표기.
- **74.1%** = 비용비 **5:1** 결과(50:1 아님).
- **74.7%** = 현재 동결 산출물에서 재현되지 않으므로 **폐기(deprecated)**.
- 검산(50:1): 무검사 {noinsp:,.0f} · 전수검사 {insp:,.0f} · 선별 {sel50:,.0f} → sel_vs_inspect = ({insp:,.0f}−{sel50:,.0f})/{insp:,.0f} = **{aps_sel_vs_inspect_50:.1f}%**.

## 2. 일반화 범위 (최종 결론 문장)

> SECOM과 APS의 예측 모델은 각각 별도로 학습했지만, 위험점수를 임계값별 혼동행렬과 비용으로 변환하여 무검사·전수검사·선별검사를 비교하는 동일한 Cost Layer 구조와 수식을 적용할 수 있었다.

정확한 구분:
- 동일 구조·수식 적용: **맞음**
- 동일 코드 파일 재사용: **아님** (`aps_pipeline.py`는 `cost_model.py`를 import하지 않고 동일 수식 재구현)
- SECOM 모델을 APS에 이식: **아님**
- APS 모델·전처리·임계값: **별도 산출** (`src/aps_pipeline.py`)

## 3. APS 발표 확정 수치

- 데이터: {nrow:,}행 × {aps['n_feat']}피처
- 양성: {npos:,}건 ({npos/nrow*100:.2f}%)
- OOF ROC-AUC: **{aps['oof_auc']:.4f}** (고정 모델 5-fold OOF, fold 내부 전처리 · nested는 비용 임계값에만)
- 공식 비용비: 미검출 500 / 불필요 점검 10 = **50 : 1**
- 최적 전략: **{a50['strategy']}**
- 전수검사 대비 절감률: **{aps_sel_vs_inspect_50:.1f}%**
- 검출률: {a50['detect_pct']}% · 과검률: {a50['overkill_pct']}% · 평균 임계값: {a50['th_mean']}
- 비용비 2~60, 20 seed 전 구간 선별검사 승률: **100%**

## 4. SECOM–APS 비교표 (figI 최종값)

| 항목 | SECOM | APS |
|---|---|---|
| 표본 / 양성 | {Ns:,} / {nps} ({nps/Ns*100:.2f}%) | {nrow:,} / {npos:,} ({npos/nrow*100:.2f}%) |
| OOF ROC-AUC | 0.731 | {aps['oof_auc']:.4f} |
| 대표 비용조건 | 15:1 (사례) | 50:1 (데이터셋 공식) |
| 최적 전략 | 선별검사 | 선별검사 |
| **절감률 (전수검사 대비)** | **{secom_sel_vs_inspect:.1f}%** | **{aps_sel_vs_inspect_50:.1f}%** |
| 50:1 선별 승률 | {secom_win[sr.index(50)]:.0f}% | 100% |
| (참고) 무료기준선 대비(sel_vs_floor) | 27.2% | {aps_sel_vs_floor_50}% |

## 5. 해석 문구 (인과 단정 금지)

> 양성 표본이 더 많고 예측 신호가 강한 APS에서는 고비용 구간에서도 선별 전략 선택이 안정적으로 유지됐다.

- ❌ "양성 표본 수만이 성능 차이의 유일한 원인" / "알고리즘 차이가 전적으로 데이터 수 때문임을 증명" — 사용 금지.
- SECOM과 APS는 피처 구성·목표 정의도 다르므로, 현재 결과는 **APS 학습 데이터 내부 교차검증에서도 동일 Cost Layer 구조의 적용 가능성을 확인**한 것으로 해석한다(= 교차검증 기반 타 제조 데이터 적용 프로브).

## 6. 평가 범위 표시

- APS AUC {aps['oof_auc']:.4f}은 **고정 모델 5-fold OOF**(fold 내부 전처리)이며, **nested 구조는 비용 임계값 평가에만** 적용된다. "nested OOF AUC"로 표기 금지.
- 발표 표현: **"APS 학습 데이터 내부 교차검증에서 동일 Cost Layer 구조의 적용 가능성 확인"** / "교차검증 기반 타 제조 데이터 적용 프로브".
- ❌ "공식 독립 Test set 외부검증 완료" · ❌ "교차검증 기반 도메인 일반화 실증"(외부 도메인 일반화 완료로 오해) — 사용 금지.

## 7. 발표용 핵심 문장

> APS 전용 모델을 별도로 학습한 결과, 공식 비용비 50:1에서 선별검사가 전수검사 대비 가정된 비용지수를 {aps_sel_vs_inspect_50:.1f}% 낮췄고, 20개 seed 모두에서 동일한 선별 전략이 선택됐습니다.

> 데이터와 예측 모델이 달라도 위험점수를 비용조건별 검사정책으로 변환하는 Cost Layer는 동일하게 적용할 수 있었습니다.

## 산출물
- 그림: `figI_secom_vs_aps` · `figJ_winrate` · `figK_costlayer` (각 PNG·SVG)
- 원본 CSV: `data_secom_vs_aps.csv` · `data_winrate.csv`
- 생성 스크립트: `scripts/make_aps_figs.py` (동결 파일만 읽음 · 재학습 없음)
"""
open(os.path.join(OUT, "APS_일반화_수치정의.md"), "w", encoding="utf-8").write(md)
print("md written:", os.path.join(OUT, "APS_일반화_수치정의.md"))
