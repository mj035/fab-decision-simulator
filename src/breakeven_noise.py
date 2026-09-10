"""
breakeven_noise.py — 리뷰어 25차 #1 지원: 판정기준을 '임의로' 고르지 않기 위한 사전측정.

목적: AUC 튜닝 재실행에서 "얼마나 움직이면 움직인 것인가"를 결과 보기 전에 고정해야 한다.
     그 임계를 감으로 정하면 철회 5번(임의 승률컷)과 같은 실수 → **현재 파이프라인의 재표집 잡음**에서 유도한다.

측정: 50 seed를 seed 단위로 부트스트랩(재표집)하여
  (a) 손익분기 cl (= 대응차 95%CI 하단 > 0인 최소 cl)의 표집분포
  (b) cl=15 절감률(sel_vs_inspect, 모델)의 표집분포
→ 이 분포의 95% 구간 폭이 '아무것도 안 바꿔도 생기는 흔들림'. 판정 임계는 그 바깥이어야 한다.

산출: results/breakeven_noise.json
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
NSEED, CLS = 50, list(range(2, 25))

X, y, _ = adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                         "Pass/Fail", 1, drop_cols=["Time"])
y = np.asarray(y); N = len(y); npos = int(y.sum()); nneg = N - npos
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xi = X[valid].fillna(X[valid].median()).to_numpy(); j59 = valid.index('59')
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


# ── seed×cl 행렬 사전계산: 최선전략 절대비용 차(센서−모델) + 모델 절감률 ──
print(f"[precompute] {NSEED} seed × {len(CLS)} cl ...")
diff = np.zeros((NSEED, len(CLS)))       # 센서비용 − 모델비용 (>0이면 모델 우세)
save15 = np.zeros(NSEED)                 # 모델 sel_vs_inspect @ cl=15


def best_cost(folds, cl):
    r = CM.evaluate(folds, N, npos, nneg, cl=cl, cs=0, beta=B, alpha=A)
    lam, mu, rho = CM.to_effective(cl, 0, B, A)
    sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
    pass_all = npos * cl; inspect_all = N + npos * cl - npos * lam + nneg * mu
    rate = (agg[3] + agg[1]) / N
    sel_eff = inspect_all if rate >= CM.DEGEN_RATE else sel     # 퇴화 선별 → 전수검사 폴백(정본 규칙)
    return min(pass_all, inspect_all, sel_eff), r


for s in range(NSEED):
    d = np.load(mfiles[s]); mf = list(zip(d["in_cms"], d["te_cms"]))
    sf = sensor_folds(s)
    for i, cl in enumerate(CLS):
        cm_, r = best_cost(mf, cl); cs_, _ = best_cost(sf, cl)
        diff[s, i] = cs_ - cm_
        if cl == 15: save15[s] = r["sel_vs_inspect"]
print(f"  점추정 손익분기 cl = ", end="")


def _sig(D):
    m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D))
    return m - 1.96 * se > 0


def breakeven_first(D):
    """[구 규칙] CI 하단>0인 **최초** cl. ⚠️ 격자 시작점 의존 — 진단용으로만 보존."""
    s = _sig(D)
    return CLS[int(np.argmax(s))] if s.any() else None


def breakeven_durable(D):
    """[정정 규칙] 이 cl 이상 **모든** cl에서 CI 하단>0인 최소 cl. 격자 시작점에 불변."""
    s = _sig(D)
    be = None
    for i in range(len(CLS) - 1, -1, -1):       # 뒤에서부터 단조 구간 확장
        if s[i]: be = CLS[i]
        else: break
    return be


breakeven = breakeven_durable
be_point = breakeven(diff); be_first = breakeven_first(diff)
print(f"[규칙비교] 최초교차={be_first} (격자 {CLS[0]}~{CLS[-1]} 의존) | 단조유지={be_point} (격자 불변)")
print(be_point, f"| cl=15 절감률 = {save15.mean():.1f}% (중앙 {np.median(save15):.1f}%)")

# ── seed 단위 부트스트랩 ──
rng = np.random.default_rng(0); BOOT = 5000
bes, svs = [], []
for _ in range(BOOT):
    idx = rng.integers(0, NSEED, NSEED)
    b = breakeven(diff[idx])
    if b is not None: bes.append(b)
    svs.append(save15[idx].mean())
bes = np.array(bes); svs = np.array(svs)
be_lo, be_hi = np.percentile(bes, [2.5, 97.5])
sv_lo, sv_hi = np.percentile(svs, [2.5, 97.5])
sv_half = (sv_hi - sv_lo) / 2

# ── 전 cl 진단표 (⚠️ 격자의존성·퇴화 규명용) ──
percl = {}
for i, cl in enumerate(CLS):
    d = diff[:, i]; m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    percl[str(cl)] = {"평균차": round(float(m), 1), "ci95": [round(float(m - 1.96 * se), 1), round(float(m + 1.96 * se), 1)],
                      "모델우세_seed": int((d > 0).sum()), "차이0_seed": int((d == 0).sum()),
                      "판정": ("모델 유의우세" if m - 1.96 * se > 0 else "센서 유의우세" if m + 1.96 * se < 0 else "대등")}

out = {
    "프로토콜": ("50 seed를 seed 단위로 부트스트랩(5000회). 손익분기 = 최선전략 절대비용 대응차의 "
             "95%CI 하단>0인 최소 cl (process_opt5/6 정본 규칙과 동일). "
             "⚠️ seed는 CV 분할만 바꾸므로 이 잡음폭은 **하한**(데이터셋 불확실성 미포함)."),
    "⚠️격자의존성": ("정본 규칙을 문자 그대로 적용하면 cl 격자를 어디서 시작하느냐가 답을 바꾼다. "
               "격자 10~24 → 16. 격자 2~24 → 3. 저cl(2~6)에서는 다수 seed가 '전수통과'로 폴백해 차이가 정확히 0이고, "
               "0의 질량이 SE를 눌러 경제적으로 무의미한 평균차(0.3~9)가 '유의'해진다. "
               "→ 16을 쓰려면 규칙을 '최초 교차'가 아니라 **'이후 단조 유지되는 최소 cl'**로 명문화해야 한다."),
    "⚠️cl8_센서우세": ("cl=8에서 평균차 −15.9, CI [−25.8, −6.1] — **센서가 유의하게 우세**. "
                 "KEY_NUMBERS의 'cl≤15 대등(95%CI 0 포함)'은 cl=8(및 5·6)에서 성립하지 않는다. "
                 "'대등'은 cl=10~15 구간 한정 명제로 축소 필요."),
    "per_cl": percl,
    "손익분기_cl": {"규칙": "이후 단조유지되는 최소 cl(격자 불변)", "점추정": be_point,
                "구규칙_최초교차": be_first, "boot_ci95": [int(be_lo), int(be_hi)],
                "분포": {str(int(v)): int((bes == v).sum()) for v in np.unique(bes)},
                "정의불가_비율": round(1 - len(bes) / BOOT, 3)},
    "cl15_절감률_%": {"평균": round(float(save15.mean()), 1), "중앙": round(float(np.median(save15)), 1),
                  "seed_sd": round(float(save15.std(ddof=1)), 1),
                  "boot_ci95": [round(float(sv_lo), 1), round(float(sv_hi), 1)],
                  "ci반폭_%p": round(float(sv_half), 1)},
}
print(f"  손익분기 cl 부트스트랩 CI = [{be_lo:.0f}, {be_hi:.0f}]  분포={out['손익분기_cl']['분포']}")
print(f"  cl15 절감률 부트스트랩 CI = [{sv_lo:.1f}, {sv_hi:.1f}] (반폭 ±{sv_half:.1f}%p)")

be_band = max(1, int(np.ceil(max(abs(be_hi - be_point), abs(be_point - be_lo)))))
out["사전등록_권고임계"] = {
    "손익분기_cl": f"|Δcl| ≥ {be_band + 1} 를 '움직였다'로 판정 (재표집 잡음폭 ±{be_band} 의 바깥)",
    "cl15_절감률": f"|Δ| ≥ {np.ceil(2*sv_half):.0f}%p 를 '움직였다'로 판정 (잡음 CI 반폭 ±{sv_half:.1f}%p의 약 2배)",
    "주의": "이 임계는 분할잡음만 반영 — 데이터셋 재표집 잡음은 미포함이므로 관대한(=쉽게 '움직였다'고 말하는) 방향의 하한임.",
}
print("\n[사전등록 권고]", json.dumps(out["사전등록_권고임계"], ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "breakeven_noise.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/breakeven_noise.json")
