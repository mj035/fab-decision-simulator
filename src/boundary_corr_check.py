"""
boundary_corr_check.py — 리뷰어 25차 #4: 헤드라인 "AUC↔경계 무상관 r=0.12"의 n·CI 확인.

의혹: r=0.12가 seed 50개에서 나온 상관이면, 학생 반복실수 4번(n=66을 독립 취급)과 같은 유형인가?
확인 항목:
 (1) r의 정확한 정의(어느 두 변수) + n
 (2) 부트스트랩 95% CI (seed 단위 리샘플)
 (3) 순서뒤집힘 26/50의 부트스트랩 안정성 (비율 CI)
 (4) ⚠️ 독립성 감사: seed가 무엇을 바꾸는가 (CV 분할만? 모델 시드도?)
산출: results/boundary_corr_check.json
"""
import os, sys, json
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "seed_boundaries.json"), encoding="utf-8"))
rows = d["rows"]
n = len(rows)

auc = np.array([r["auc"] for r in rows], float)
guard = np.array([r["guard_cl"] for r in rows], float)
econ = np.array([np.nan if r["econ_cl"] is None else r["econ_cl"] for r in rows], float)
flip = np.array([str(r["order"]).startswith("econ<") for r in rows])

out = {"n_seed": n,
       "seed_varies": d.get("seed_varies"),
       "⚠️독립성감사": ("seed는 CV 분할만 바꾸고 모델 random_state는 0 고정 — 같은 1567행 데이터셋의 "
                    "재분할 50회. 따라서 n=50은 '독립 표본 50개'가 아니라 '한 데이터셋의 분할 변동 50회'. "
                    "CI는 분할 불확실성만 담고 데이터셋 불확실성은 담지 못함(과소추정 방향).")}

rng = np.random.default_rng(0)
B = 10000


def boot_r(x, y, kind="pearson"):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    f = pearsonr if kind == "pearson" else spearmanr
    pt, p = f(x, y)
    idx = rng.integers(0, len(x), (B, len(x)))          # seed 단위 리샘플
    bs = []
    for i in idx:
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        bs.append(f(x[i], y[i])[0])
    bs = np.array(bs)
    return {"n": int(len(x)), "r": round(float(pt), 3), "p_naive": round(float(p), 4),
            "ci95_boot": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "부호안정_%": round(float((np.sign(bs) == np.sign(pt)).mean() * 100), 1)}


pairs = {"AUC↔guard_cl": (auc, guard), "AUC↔econ_cl": (auc, econ), "guard_cl↔econ_cl": (guard, econ)}
out["상관"] = {k: {"pearson": boot_r(a, b, "pearson"), "spearman": boot_r(a, b, "spearman")}
              for k, (a, b) in pairs.items()}
for k, v in out["상관"].items():
    pe, sp = v["pearson"], v["spearman"]
    print(f"[{k}] n={pe['n']}  pearson r={pe['r']:+.3f} CI{pe['ci95_boot']} (부호안정 {pe['부호안정_%']}%)"
          f" | spearman ρ={sp['r']:+.3f} CI{sp['ci95_boot']}")

# ── 순서뒤집힘 비율의 부트스트랩 ──
k_flip = int(flip.sum())
bs_f = rng.binomial(n, k_flip / n, B) / n
lo_w, hi_w = np.percentile(bs_f, [2.5, 97.5])
out["순서뒤집힘"] = {"count": f"{k_flip}/{n}", "비율": round(k_flip / n, 3),
                 "ci95_boot": [round(float(lo_w), 3), round(float(hi_w), 3)],
                 "판정": ("CI가 0.5를 포함 → '어느 경계가 큰지 seed마다 뒤집힌다(동전던지기)'는 서술 유지 가능"
                        if lo_w <= 0.5 <= hi_w else
                        "CI가 0.5를 배제 → 한쪽 순서가 우세. '무작위로 뒤집힌다' 표현은 부정확")}
print(f"[순서뒤집힘] {k_flip}/{n} = {k_flip/n:.2f} CI[{lo_w:.2f},{hi_w:.2f}] → {out['순서뒤집힘']['판정']}")

# ── 헤드라인 문구 판정 ──
best = max(out["상관"].items(), key=lambda kv: abs(kv[1]["pearson"]["r"]))
out["헤드라인_판정"] = {
    "무상관주장이_성립하려면": "r의 CI가 0을 포함하고, 실용상 무시할 크기여야 함",
    "관측": {k: v["pearson"]["ci95_boot"] for k, v in out["상관"].items()},
    "권고": None}
zero_in = {k: (v["pearson"]["ci95_boot"][0] <= 0 <= v["pearson"]["ci95_boot"][1]) for k, v in out["상관"].items()}
out["헤드라인_판정"]["권고"] = (
    "AUC↔경계 CI가 0을 포함 → '무상관'이 아니라 **'상관을 검출하지 못함(CI 넓음, n=50 분할변동)'**으로 표현. "
    "'r=0.12' 단일수치 인용 금지, CI 병기 필수."
    if zero_in.get("AUC↔guard_cl") or zero_in.get("AUC↔econ_cl") else
    "CI가 0을 배제 → 무상관 주장 철회 필요")
print("→", out["헤드라인_판정"]["권고"])

json.dump(out, open(os.path.join(ROOT, "results", "boundary_corr_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/boundary_corr_check.json")
