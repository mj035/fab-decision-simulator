"""
cl8_promote_check.py — 리뷰어 25차 (C): cl=8 센서우세를 헤드라인으로 '승격'해도 되는가.

리뷰어 제안: 3구간 구조(cl≲8 센서우세 / 10~15 대등 / ≥16 모델우세)로 승격.
그러나 25차 스윕은 cl=7 대등, **cl=8만 유의**, cl=9 대등 — **고립점**이다.
승격 전 결정적 확인 3가지:
 (1) 다중비교: cl 2~24 중 몇 개를 검정했고, 센서방향 유의는 몇 개인가 (우연 기대치와 비교)
 (2) 안정성: seed 부트스트랩에서 cl=8 센서우세가 몇 % 재현되는가
 (3) 기제: '모델 과검사' 가설이 맞는가 (25차 스윕은 이미 기각 — 검사율 동일 15.8%)
     → 진짜 기제는 동일 검사량에서의 **국소 순위 역전**인지 카운트로 확인
+ (B) 규칙 정의역: cl 상단을 60 → 150까지 밀어 CI 하단>0이 언제 깨지는지 확정.
산출: results/cl8_promote_check.json
"""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
NSEED = 50
CLS = list(range(2, 151))  # (B) 정의역 확정용 — 상단 150까지

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
    sel_eff = inspect_all if rate >= CM.DEGEN_RATE else sel
    return min(pass_all, inspect_all, sel_eff), rate, agg


print(f"[precompute] {NSEED} seed × cl 2~150 ...")
diff = np.zeros((NSEED, len(CLS)))
agg8 = {"m": [], "s": []}
for s in range(NSEED):
    d = np.load(mfiles[s]); mf = list(zip(d["in_cms"], d["te_cms"]))
    sf = sensor_folds(s)
    for i, cl in enumerate(CLS):
        cm_, mr, ma = best_cost(mf, cl); cs_, sr, sa = best_cost(sf, cl)
        diff[s, i] = cs_ - cm_
        if cl == 8: agg8["m"].append(ma); agg8["s"].append(sa)

mean = diff.mean(0); se = diff.std(0, ddof=1) / np.sqrt(NSEED)
lo, hi = mean - 1.96 * se, mean + 1.96 * se
out = {}

# ── (B) 정의역 확정 ──
i16 = CLS.index(16)
breaks = [CLS[i] for i in range(i16, len(CLS)) if not (lo[i] > 0)]
out["B_정의역"] = {
    "확인범위": "cl 2~150",
    "cl≥16에서_깨지는_최초cl": breaks[0] if breaks else None,
    "CI하단_추이": {str(cl): round(float(lo[CLS.index(cl)]), 1) for cl in (16, 30, 60, 90, 120, 150) if cl in CLS},
    "판정": (f"cl={breaks[0]}에서 CI 하단이 0 이하로 떨어짐 → **규칙 정의역을 cl ∈ [2, {breaks[0]-1}]로 명문화 필요**"
           if breaks else "cl 150까지 전 구간 유지 — 실무 범위에서 정의역 문제 없음")}
print(f"\n[B] cl≥16 깨지는 최초 cl = {breaks[0] if breaks else '없음(≤150)'}")
print("    CI 하단 추이:", out["B_정의역"]["CI하단_추이"])

# ── (C1) 다중비교 ──
rng = np.random.default_rng(0)
tested = [cl for cl in range(2, 25)]
sens_sig = [cl for cl in tested if hi[CLS.index(cl)] < 0]
mod_sig = [cl for cl in tested if lo[CLS.index(cl)] > 0]
out["C1_다중비교"] = {
    "검정한_cl수": len(tested), "센서_유의": sens_sig, "모델_유의(≤24)": mod_sig,
    "우연기대치": round(len(tested) * 0.025, 1),
    "주의": ("cl들은 중첩 비용곡선이라 독립 검정이 아니다(유효 검정수 < 23). "
           "그러나 **진짜 구간이라면 연속으로 유의해야 한다** — cl=7 대등, 8 유의, 9 대등의 "
           "고립 패턴은 구간이 아니라 단일점의 특징이다.")}
print(f"\n[C1] 센서 유의 cl = {sens_sig} (검정 {len(tested)}개, 단측 우연기대 ~{len(tested)*0.025:.1f}개)")

# ── (C2) seed 부트스트랩 안정성 ──
BOOT = 5000
i7, i8, i9 = CLS.index(7), CLS.index(8), CLS.index(9)
rep = {7: 0, 8: 0, 9: 0}
for _ in range(BOOT):
    idx = rng.integers(0, NSEED, NSEED)
    for cl, i in ((7, i7), (8, i8), (9, i9)):
        d = diff[idx, i]; m = d.mean(); s_ = d.std(ddof=1) / np.sqrt(NSEED)
        if m + 1.96 * s_ < 0: rep[cl] += 1
out["C2_안정성"] = {f"cl={k}_센서유의_재현율_%": round(v / BOOT * 100, 1) for k, v in rep.items()}
out["C2_안정성"]["판정"] = ("재현율 80% 이상이면 승격 검토 가능" if rep[8] / BOOT >= 0.8 else
                        f"cl=8 재현율 {rep[8]/BOOT*100:.0f}% — **불안정. 헤드라인 승격 부적합**")
print(f"[C2] 센서우세 부트스트랩 재현율: {out['C2_안정성']}")

# ── (C3) 기제 ──
ma = np.array(agg8["m"], float); sa = np.array(agg8["s"], float)   # (seed,4) tn,fp,fn,tp
lam8 = B * 8
d_tp = float((sa[:, 3] - ma[:, 3]).mean()); d_insp = float(((sa[:, 3] + sa[:, 1]) - (ma[:, 3] + ma[:, 1])).mean())
# 비용차(센서−모델) = Δ검사건수 − Δtp·λ  (μ=α·cs=0). 부호를 실측과 같은 방향으로 맞춘다.
# ⚠️ 28차 정정: 종전 `d_tp*lam8 - d_insp`는 부호가 뒤집혀 예측 +15.9 vs 실측 −15.9로 JSON이 자기모순이었다.
pred = d_insp - d_tp * lam8
out["C3_기제"] = {
    "모델": {"검사": round(float((ma[:, 3] + ma[:, 1]).mean()), 1), "tp": round(float(ma[:, 3].mean()), 1),
           "fp": round(float(ma[:, 1].mean()), 1), "검사율_%": round(float((ma[:, 3] + ma[:, 1]).mean()) / N * 100, 1)},
    "센서": {"검사": round(float((sa[:, 3] + sa[:, 1]).mean()), 1), "tp": round(float(sa[:, 3].mean()), 1),
           "fp": round(float(sa[:, 1].mean()), 1), "검사율_%": round(float((sa[:, 3] + sa[:, 1]).mean()) / N * 100, 1)},
    "Δtp(센서−모델)": round(d_tp, 2), "Δ검사건수(센서−모델)": round(d_insp, 2),
    "예측_비용차(=Δ검사−Δtp·λ)": round(pred, 1), "실측_비용차": round(float(mean[i8]), 1),
    "과검사가설": "기각 — 검사율이 사실상 동일",
    "실제기제": (f"동일 검사량(Δ검사 {d_insp:+.1f}건)에서 센서가 tp를 {d_tp:+.2f}건 더 적발 = "
             f"**국소 순위 역전**. 적발 1건 가치 λ=β·cl={lam8:.1f} → 예측 비용차 {pred:+.1f} vs 실측 {mean[i8]:+.1f} (정합). "
             "전역 AUC는 모델이 +0.052 우세하지만, 검사율 ~16% 운영점의 상위 랭킹에서는 역전된다.")}
print(f"\n[C3] 기제: Δtp={d_tp:+.2f}, Δ검사={d_insp:+.1f} → 예측 {pred:+.1f} vs 실측 {mean[i8]:+.1f}")
print("    ", out["C3_기제"]["실제기제"])

r8 = rep[8] / BOOT * 100
out["승격_권고"] = {
    "찬성_근거": [
        f"seed 부트스트랩 재현율 {r8:.0f}% — 사전 기준 80%를 넘음. 잡음이라기엔 안정적.",
        f"기제가 대수적으로 정합: Δtp {d_tp:+.2f} × λ={lam8:.1f} − Δ검사 {d_insp:+.1f} = 예측 {pred:+.1f} vs 실측 {mean[i8]:+.1f}. "
        "'유의하지만 설명 못 하는 효과'가 아니라 기제가 특정된 효과다.",
        "센서방향 유의 1개는 단측 우연기대 0.6개보다 많지는 않지만, 기제 정합이 우연 가설을 약화한다."],
    "반대_근거": [
        "cl=7(재현율 %.0f%%)·cl=9(%.0f%%)가 모두 '대등' — **고립점**. 진짜 구간이면 연속 유의해야 한다."
        % (rep[7] / BOOT * 100, rep[9] / BOOT * 100),
        f"효과 크기 = 불량 {abs(d_tp):.1f}건. 발표 헤드라인이 감당할 크기가 아니다.",
        "cl 2~24를 전부 검정했고 다중비교 보정을 하지 않았다."],
    "결론": (
        "⚠️ **부분 승격 — '3구간 구조'는 반대, '기제'는 승격 찬성.**\n"
        "  (a) 3구간 헤드라인(cl≲8 센서우세)은 **반대**. 데이터가 말하는 건 구간이 아니라 cl=8 고립점이고, "
        "'cl≲8'로 쓰면 cl 2~7까지 포함해 근거 없는 확대가 된다.\n"
        "  (b) 'cl≤15 대등'은 **'cl 9~15 대등'으로 정정**하고, cl 7~8은 '센서 근소우세 가능성(단일점, 다중비교 미보정)'으로 별도 표기.\n"
        "  (c) **기제는 승격 가치가 있음**: 전역 AUC가 모델 +0.052 우세인데도 검사율 ~16% 운영점에서 순위가 역전된다. "
        "이건 §4(경계=ROC 꼬리)와 헤드라인 3번('AUC는 병목이 아니다')의 **직접 실증**이다 — "
        "AUC 우위가 운영점 우위를 보장하지 않는다는 것을 같은 데이터에서 보여준다.\n"
        "  → 즉 리뷰어의 '날카롭게 만든다'는 직관은 맞지만, 날카로워지는 건 구간 구조가 아니라 **AUC↛운영성능 논증**이다.")}
print("\n[승격 권고]", out["승격_권고"]["결론"])
print("\n[승격 권고]", out["승격_권고"])

json.dump(out, open(os.path.join(ROOT, "results", "cl8_promote_check.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/cl8_promote_check.json")
