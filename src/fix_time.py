"""
fix_time.py — Time 컬럼 파싱오류(DD/MM→MM/DD 오독) 복원 유틸 + 증거 산출 (리뷰어 23차 §0).

손상 메커니즘: 원본 SECOM은 DD/MM/YYYY. month-first 파서가 DD<=12이면 '성공'해 월↔일이 뒤바뀜,
DD>12이면 실패해 dayfirst 폴백으로 올바르게 파싱됨.
→ 복원 규칙(B, 메커니즘 충실): day<=12인 모든 행의 month↔day 스왑.
  (리뷰어 검증3의 규칙 A = 복원범위 밖 497행만 스왑. 둘을 비교해 역전 적은 쪽을 채택.)

산출: results/time_bug.json, figures/fig_time_bug.png (멘토링 설명용 2패널)
사용: from fix_time import restore_time; t_fixed = restore_time(df)  # df에 'Time' 컬럼
대원칙: 분석 시간축은 타임스탬프가 아니라 **행 인덱스(생산 순서)**를 사용한다. sort_values('Time') 금지.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")


def restore_time(df, col="Time"):
    """day<=12인 행의 month↔day 스왑. 원본 행 순서 유지, 시·분 보존. 반환: 복원된 Series."""
    t = pd.to_datetime(df[col], errors="coerce")
    bad = t.dt.day <= 12
    fx = t.copy()
    fx[bad] = pd.to_datetime(dict(year=t[bad].dt.year, month=t[bad].dt.day, day=t[bad].dt.month)) \
              + pd.to_timedelta(t[bad].dt.hour, "h") + pd.to_timedelta(t[bad].dt.minute, "m")
    return fx


def _inversions(s):
    return int((s.diff().dt.total_seconds() < 0).sum())


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    t = pd.to_datetime(df["Time"])
    N = len(t)

    # ── 증거 1: 월별 등장 일(day) — 1~6·11~12월은 8·9·10뿐 ──
    month_days = {int(m): sorted(int(d) for d in t[t.dt.month == m].dt.day.unique()) for m in range(1, 13)}
    month_cnt_broken = {int(m): int((t.dt.month == m).sum()) for m in range(1, 13)}
    offrange_jan = int(((t.dt.month == 1) & (~t.dt.day.isin([8, 9, 10]))).sum())  # 반증 카운트(=0이어야)

    # ── 규칙 A(리뷰어: 복원범위 밖만) vs 규칙 B(메커니즘: day<=12 전부) ──
    lo, hi = pd.Timestamp("2008-07-19 11:55"), pd.Timestamp("2008-10-17 06:07")
    badA = ~t.between(lo, hi)
    fxA = t.copy()
    fxA[badA] = pd.to_datetime(dict(year=t[badA].dt.year, month=t[badA].dt.day, day=t[badA].dt.month)) \
                + pd.to_timedelta(t[badA].dt.hour, "h") + pd.to_timedelta(t[badA].dt.minute, "m")
    fxB = restore_time(df)
    invA, invB = _inversions(fxA), _inversions(fxB)
    fx = fxB if invB <= invA else fxA
    rule = "B(day<=12 전부 스왑)" if invB <= invA else "A(범위 밖만 스왑)"

    span = (fx.max() - fx.min()).days
    rho = pd.Series(np.arange(N)).corr(pd.Series(fx.values.astype("int64")), method="spearman")

    # ── 검증 assert (리뷰어 요구) ──
    assert 88 <= span <= 90, f"복원 후 기간 {span}일 — 89일 예상과 불일치"
    assert min(invA, invB) <= 5, f"역전 {min(invA, invB)}회 — 5 이하 예상"
    assert rho > 0.99, f"spearman(행, 시간)={rho:.4f} — 0.99 초과 예상"
    assert offrange_jan == 0, "1월에 8/9/10 외 일 존재 — 스왑 가설 반증"

    month_cnt_fixed = {int(m): int((fx.dt.month == m).sum()) for m in range(1, 13)}
    out = {
        "결론": f"Time 컬럼 DD/MM→MM/DD 파싱오류. 실제 기간 {fx.min().date()}~{fx.max().date()} ({span}일≈2.9개월). "
                "행 인덱스=생산 순서. 시간축은 행 인덱스 사용.",
        "복원규칙": {"채택": rule, "A_범위밖만_497행_역전": invA, "B_day<=12_전부_역전": invB,
                  "스왑행수": {"A": int(badA.sum()), "B": int((t.dt.day <= 12).sum())}},
        "복원후": {"min": str(fx.min()), "max": str(fx.max()), "기간_일": span,
                 "행순서_spearman": round(float(rho), 5), "역전_채택규칙": min(invA, invB)},
        "증거_월별_등장일": {f"{m:02d}월": {"행수": month_cnt_broken[m], "일": month_days[m]} for m in range(1, 13)},
        "증거_1월_8_9_10_외": offrange_jan,
        "센서측_원본일치": "1567행/590센서/불량104/결측4.54%/상수열116 — Time 컬럼만 손상",
        "월별행수": {"손상": month_cnt_broken, "복원": month_cnt_fixed},
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "time_bug.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ["결론", "복원규칙", "복원후", "증거_1월_8_9_10_외"]}, ensure_ascii=False, indent=1))

    # ══════════════ fig_time_bug.png — 멘토링 설명용 2패널 ══════════════
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    BLUE, RED, GRAY = "#4f86d9", "#d94f4f", "#8c8c8c"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("Time 컬럼 파싱오류 증거 — 유럽식 DD/MM을 MM/DD로 오독 (실제 기간: 2008-07-19 ~ 10-17, 89일)",
                 fontsize=12.5, fontweight="bold", y=0.99)

    # 좌: 월×일 존재 히트맵 — "1~6·11~12월은 8·9·10일만" 이 3초 안에 보이게
    H = np.zeros((12, 31))
    for m in range(1, 13):
        for d in t[t.dt.month == m].dt.day.value_counts().items():
            H[m - 1, d[0] - 1] = d[1]
    ax1.imshow(np.where(H > 0, np.log1p(H), np.nan), aspect="auto", cmap="Blues", origin="upper")
    ax1.set_xticks(range(0, 31, 2), [str(d) for d in range(1, 32, 2)], fontsize=8)
    ax1.set_yticks(range(12), [f"{m}월" for m in range(1, 13)], fontsize=9)
    ax1.set_xlabel("일(day)", fontsize=10); ax1.set_title("타임스탬프 그대로: 월별로 등장하는 '일'", fontsize=11)
    ax1.add_patch(Rectangle((6.5, -0.5), 3, 12, fill=False, edgecolor=RED, lw=2.2))
    ax1.text(10.2, 2.5, "← 1~6·11~12월엔 '일'이 7·8·9·10뿐\n    = 사실은 원본의 '월'(7~10월)\n    (공장이 매달 사흘만 가동? 불가능)",
             fontsize=9.5, color=RED, va="center")
    for m in [7, 8, 9, 10]:
        ax1.text(31.2, m - 1, "정상 파싱(일>12 존재)", fontsize=7.5, color=GRAY, va="center")

    # 우: 복원 전후 월별 행 수
    xs = np.arange(1, 13)
    ax2.bar(xs - 0.2, [month_cnt_broken[m] for m in xs], width=0.4, color=GRAY, alpha=0.75, label="손상 타임스탬프 (1~12월에 산재)")
    ax2.bar(xs + 0.2, [month_cnt_fixed[m] for m in xs], width=0.4, color=BLUE, label="복원 후 (7~10월만 = 89일)")
    ax2.set_xticks(xs, [f"{m}월" for m in xs], fontsize=9); ax2.set_ylabel("행 수", fontsize=10)
    ax2.set_title(f"월별 행 수: 복원 전 vs 후 (스왑 {int((t.dt.day<=12).sum())}행 · 복원 후 행순서 역전 {invB}/1566 = 완전 시간정렬)", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.text(0.98, 0.97, f"복원 = day$\\leq$12인 {int((t.dt.day<=12).sum())}행의 월↔일 교환\n무작위 순서라면 역전 약 783회 기대 → 관측 {invB}회\n⇒ 원본은 시간순 정렬 파일, 행 인덱스=생산 순서",
             transform=ax2.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e6", ec="#d9b44f"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "fig_time_bug.png"), bbox_inches="tight")
    print("저장: results/time_bug.json, figures/fig_time_bug.png")
