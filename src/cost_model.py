"""
B-1 v2: 유효 파라미터로 재설계. (멘토 4차 대수 검산 반영 — 직접 재검산 완료)

★ 핵심구조: 세 전략에서 공통항 npos*cl 소거 시, β·α는 항상 아래 결합으로만 등장.
    λ = β·cl   (유효 유출비용)   μ = α·cs   (유효 오폐기비용)   ρ = β·γ·V (유효 리페어회수)
  → 의사결정은 (λ, μ[, ρ]) 2(3)차원으로 결정. β,α,γ,V 단독 추정 불필요.
  단, 절감률 '크기'는 npos*cl 때문에 scale(cl)이 추가로 필요(결정엔 무관).

전략 절대비용 (리페어 γ,V 포함. γ=0이면 기존모델로 환원 = 하위호환):
  전수통과 = npos*cl
  전수검사 = N + npos*cl - npos*λ + nneg*μ - npos*ρ
  선별검사 = (tp+fp) + npos*cl - tp*λ + fp*μ - tp*ρ
임계값 결정(공통항 소거 후): argmin_j  fn*(λ+ρ-1) + fp*(1+μ)   →  R_eff = (λ+ρ-1)/(1+μ)
실익구간(λ):  [(tp+fp)+fp*μ]/tp - ρ  <  λ  <  [(tn+fn)+tn*μ]/fn - ρ   (암묵적: 카운트가 임계값 의존)
"""
import numpy as np

def to_effective(cl, cs, beta, alpha, gamma=0.0, V=0.0):
    return beta*cl, alpha*cs, beta*gamma*V         # (λ, μ, ρ)

def best_threshold_idx(in_cm, lam, mu, rho=0.0):
    fp, fn = in_cm[:,1], in_cm[:,2]
    return int(np.argmin(fn*(lam+rho-1.0) + fp*(1.0+mu)))   # 공통항 소거형 (빠르고 안정)

def strat_costs(N, npos, nneg, tp, fp, fn, cl, lam, mu, rho=0.0):
    pass_all    = npos*cl
    inspect_all = N + npos*cl - npos*lam + nneg*mu - npos*rho
    selective   = (tp+fp) + npos*cl - tp*lam + fp*mu - tp*rho
    return pass_all, inspect_all, selective

def band_closed_form(tn, fp, fn, tp, mu, rho=0.0):
    """실익구간을 λ로 닫힌해. 카운트는 경계 임계값의 값(암묵적)→그리드와 대조검증용."""
    lower = ((tp+fp) + fp*mu)/tp - rho if tp > 0 else np.inf   # tp=0 → 선별 항상 불리
    upper = ((tn+fn) + tn*mu)/fn - rho if fn > 0 else np.inf   # fn=0 → 상한 무한
    return lower, upper

DEGEN_RATE = 0.90   # 검사율 ≥ 이 값이면 '선별'은 사실상 전수검사(퇴화) — 5(b) 가드

def _accumulate(folds, lam, mu, rho, cl):
    """fold별 안쪽선택 임계값으로 바깥 카운트·선별비용 누적."""
    sel_total = 0.0; agg = np.zeros(4); n_check = 0; pos_check = 0
    for in_cm, te_cm in folds:
        j = best_threshold_idx(in_cm, lam, mu, rho)
        tn, fp, fn, tp = te_cm[j]
        sel_total += (tp+fp) + (tp+fn)*cl - tp*lam + fp*mu - tp*rho
        agg += (tn, fp, fn, tp); n_check += tn+fp+fn+tp; pos_check += tp+fn
    return sel_total, agg, n_check, pos_check

def effective_strategy(best, rate):
    """퇴화 선별(검사율≥90%)을 전수검사로 재분류. 운영상 동일하므로 정직한 병합."""
    if best == 2 and rate >= DEGEN_RATE:
        return 1
    return best

def evaluate(folds, N, npos, nneg, cl, cs, beta, alpha, gamma=0.0, V=0.0):
    """임계값은 안쪽 fold서만 선택(낙관편향 제거). savings 음수 허용(선별 불리 노출)."""
    lam, mu, rho = to_effective(cl, cs, beta, alpha, gamma, V)
    sel_total, agg, n_check, pos_check = _accumulate(folds, lam, mu, rho, cl)
    # 🟡 스케일 일치 검증 (outer fold가 완전분할일 때만 같은 스케일)
    assert n_check == N,    f"스케일 불일치: fold합 {n_check} vs N {N}"
    assert pos_check == npos, f"양성 불일치: fold합 {pos_check} vs npos {npos}"

    tn, fp, fn, tp = agg
    rate = (tp + fp) / N
    pass_all = npos*cl
    inspect_all = N + npos*cl - npos*lam + nneg*mu - npos*rho
    costs = [pass_all, inspect_all, sel_total]
    best  = int(np.argmin(costs))
    floor = min(pass_all, inspect_all)
    return {
        "strategy": best,                                           # 0전수통과 1전수검사 2선별(원시)
        "eff_strategy": effective_strategy(best, rate),             # 퇴화선별→전수검사 병합
        "inspect_rate": rate,                                       # 선별 검사율(=flag 비율)
        "sel_vs_inspect": (inspect_all - sel_total)/inspect_all*100,  # 예선(전수검사 대비) 비교용
        "sel_vs_floor":   (floor - sel_total)/floor*100,              # ★ 음수 허용: 선별 불리시 마이너스
        "best_vs_floor":  (floor - costs[best])/floor*100,            # 최적전략 절감(항상 ≥0)
        "R_eff": (lam+rho-1.0)/(1.0+mu),
        "lam": lam, "mu": mu, "rho": rho,
        "counts": tuple(int(x) for x in agg),                        # (tn,fp,fn,tp) 경계 카운트
    }

def genuine_selective_band(folds, N, npos, nneg, cs, beta, alpha, gamma=0.0, V=0.0,
                           cl_lo=2, cl_hi=1500):
    """진짜 선별(eff_strategy==2)이 최적인 cl의 [min,max]. 깜빡임/비연속에 강건(전 구간 수집)."""
    good = [cl for cl in range(cl_lo, cl_hi)
            if CM_eff(folds, N, npos, nneg, cl, cs, beta, alpha, gamma, V) == 2]
    return (min(good), max(good)) if good else None

def CM_eff(folds, N, npos, nneg, cl, cs, beta, alpha, gamma=0.0, V=0.0):
    lam, mu, rho = to_effective(cl, cs, beta, alpha, gamma, V)
    sel_total, agg, _, _ = _accumulate(folds, lam, mu, rho, cl)
    tn, fp, fn, tp = agg; rate = (tp+fp)/N
    pass_all = npos*cl
    inspect_all = N + npos*cl - npos*lam + nneg*mu - npos*rho
    best = int(np.argmin([pass_all, inspect_all, sel_total]))
    return effective_strategy(best, rate)

def CM_raw(folds, N, npos, nneg, cl, cs, beta, alpha, gamma=0.0, V=0.0):
    """퇴화 가드 미적용 원시 전략(best). 닫힌해와의 대조검증 전용 — effective_strategy 안 씌움."""
    lam, mu, rho = to_effective(cl, cs, beta, alpha, gamma, V)
    sel_total, agg, _, _ = _accumulate(folds, lam, mu, rho, cl)
    pass_all = npos*cl
    inspect_all = N + npos*cl - npos*lam + nneg*mu - npos*rho
    return int(np.argmin([pass_all, inspect_all, sel_total]))

def raw_selective_band(folds, N, npos, nneg, cs, beta, alpha, gamma=0.0, V=0.0,
                       cl_lo=2, cl_hi=1500):
    """원시 선별(best==2)이 최적인 cl의 [min,max]. 닫힌해 상한과 1~2 단위로 일치해야 함."""
    good = [cl for cl in range(cl_lo, cl_hi)
            if CM_raw(folds, N, npos, nneg, cl, cs, beta, alpha, gamma, V) == 2]
    return (min(good), max(good)) if good else None

def decide_lmr(folds, N, npos, nneg, lam, mu, rho=0.0):
    """공통항 npos*cl 소거한 축약비용으로 (λ,μ,ρ)만으로 유효전략 판정. 실익맵용(cl 불필요)."""
    sel_red = 0.0; agg = np.zeros(4)
    for in_cm, te_cm in folds:
        j = best_threshold_idx(in_cm, lam, mu, rho)
        tn, fp, fn, tp = te_cm[j]
        sel_red += (tp+fp) - tp*lam + fp*mu - tp*rho
        agg += (tn, fp, fn, tp)
    tn, fp, fn, tp = agg; rate = (tp+fp)/N
    inspect_red = N - npos*lam + nneg*mu - npos*rho
    best = int(np.argmin([0.0, inspect_red, sel_red]))    # pass_red=0
    return effective_strategy(best, rate), rate

STRAT_NAME = ["전수통과", "전수검사", "선별검사"]

# ══════════ 역산 모드 (inverse) ══════════
# 임계값 j의 축약비용 g(j)/(1+μ) = fn_j·R + fp_j  (R=(Λ−1)/(1+μ), Λ=λ+ρ)
#   → R에 대한 직선(기울기 fn_j, 절편 fp_j). 하부 포락선=최적 임계값 = ROC 볼록껍질.
#   j가 최적인 R 구간 [R⁻,R⁺]가 곧 "그 정책이 최적이려면 유출:검사 유효비율이 이 범위"라는 역산.
def implied_R_interval(cm, j):
    """관측 임계값 j가 최적이 되는 R=(Λ−1)/(1+μ) 구간. μ 불필요.
    반환: (R_lo, R_hi) | (None, dominating_k)=지배당함(볼록껍질 내부, 어떤 R서도 비최적)."""
    fn = cm[:, 2].astype(float); fp = cm[:, 1].astype(float)
    R_lo, R_hi = 0.0, np.inf
    for k in range(len(fn)):
        if k == j:
            continue
        d_fn = fn[j] - fn[k]          # j가 최적: fn_j·R+fp_j ≤ fn_k·R+fp_k
        d_fp = fp[k] - fp[j]
        if d_fn > 0:   R_hi = min(R_hi, d_fp / d_fn)
        elif d_fn < 0: R_lo = max(R_lo, d_fp / d_fn)
        elif d_fp < 0: return None, k    # 동일 fn·더 큰 fp → k에 지배
    return (R_lo, R_hi) if R_lo <= R_hi else (None, None)

def js_for_rate(cm, N, rate, tol=0.01):
    """검사율 rate(±tol)을 내는 임계값 인덱스들. 이산성 → 밴드로 수집(없으면 최근접 1개)."""
    r = (cm[:, 3] + cm[:, 1]) / N               # (tp+fp)/N
    band = [j for j in range(len(r)) if abs(r[j] - rate) <= tol]
    if not band:
        band = [int(np.argmin(np.abs(r - rate)))]
    return band, r

def implied_R_for_rate(cm, N, rate, tol=0.01):
    """검사율 rate에서 관측정책이 최적인 R 구간(밴드 합집합) 또는 지배 판정."""
    band, r = js_for_rate(cm, N, rate, tol)
    los, his, dominated = [], [], 0
    for j in band:
        res = implied_R_interval(cm, j)
        if res[0] is None:
            dominated += 1
        else:
            los.append(res[0]); his.append(res[1])
    if not los:
        return None                              # 밴드 전부 지배 → 개선여지
    return (min(los), max(his), len(band), dominated)  # 합집합 하한/상한
