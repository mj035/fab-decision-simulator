"""
model_cmp.py — 모델 계열 비교 실험. **`사전등록_모델계열비교.md` §1~§7을 그대로 집행.**

⚠️ 격리: 정본 코드·결과·그림·보고서·시뮬레이터를 일절 수정하지 않는다. 출력은 results/model_cmp/ 아래에만.
⚠️ 판정 임계는 실행 전에 고정되었다(results/model_cmp/thresholds.json). 결과를 보고 바꾸지 않는다.

── 리뷰어 지적 반영 4건 (실행 전 수정) ──────────────────────────────────
 (1) 전처리 누수: 종전엔 outer train 전체에서 대치·분산필터·표준화를 fit한 뒤 inner CV에 넣었다.
     → **모든 전처리를 각 training subset 내부에서만 fit**한다. 전역 상수열 제거도 폐지하고
     fold 내부 VarianceThreshold가 처리한다(`Xraw`는 열 필터를 거치지 않은 원본).
 (2)+(5) 정책 정의를 **순위 기반**으로 바꾼다(SVC 척도 문제의 근본 해결).
     종전엔 inner·outer가 각각의 분위수 격자를 써서 임계값 인덱스가 어긋났고(1차 결함),
     이를 "inner에서 만든 숫자 배열을 outer에 이전"으로 고쳤으나 **여전히 불일치**였다 —
     그 배열은 3개의 서로 다른 inner SVC fit의 margin에서 나온 값인데,
     이를 네 번째 fit(outer train 전체)의 margin에 적용하므로 같은 숫자가 같은 운영점을 뜻하지 않는다.
     → **정책을 "점수 상위 q% 검사"로 정의**한다. q는 fit·척도와 무관하므로 inner에서 고른 q가
     outer에서 동일한 운영 결정을 뜻한다. calibration보다 싸고(추가 적합 0회) 도구의 실제 운용 방식과도 일치한다.
 (3) AUC 주 판정을 **outer fold별 AUC 평균**으로 바꾼다. pooled OOF는 참고값으로만 병기한다.
     (SVC decision_function은 fold마다 스케일이 달라 pooled 결합 자체가 부적절하다.)
 (4) **finalize-only 모드**(캐시 집계 전용) + **원자적 캐시 저장**을 추가한다.
 (6) 본 실행 루프를 **seed 우선**으로 바꾼다 — 중단되어도 모든 arm의 **공통 연속 seed**가 쌓인다.
 (7) 캐시 로딩 시 파일 핸들을 **즉시 닫는다**(np.load 지연 핸들 누수 방지).
 (8) **inner에서도 fit 간 점수를 합치지 않는다**(중대). 종전엔 3개 inner fit의 margin을 한 배열로 합쳐
     ① pooled AUC로 C를 고르고 ② 합친 배열을 전역 정렬해 상위 q%를 정했다.
     q는 **한 fit 내부에서만** 척도 불변이므로, 척도가 다른 세 fit을 합쳐 정렬하면
     margin 범위가 넓은 fold의 표본이 과대 선택된다. → **inner fold별로 AUC·혼동행렬을 각각 계산**하고,
     C 선택은 **fold AUC 평균**, 비용용 inner 행렬은 **fold별 q 혼동행렬의 합**으로 바꾼다.
     (outer AUC에서 pooled를 버린 것과 동일 원칙을 inner에 적용.)
 (9) **동점 처리를 중립화**한다. 안정 정렬은 결정론적이지만 동점 시 **원본 행 순서**를 우선하는데,
     SECOM 행 순서는 생산 순서를 보존하므로 시간이 보조 판단 기준으로 새어든다.
     RF 확률은 이산적이라 동점이 많다. → **cutoff 동점은 전부 포함**하고 **실제 검사율을 별도 기록**한다.
     q는 이제 "목표 검사율"이며 실제 검사율은 동점 때문에 소폭 커질 수 있다.

실행:
  CMP_PROBE=1    python src/model_cmp.py    시간 프로브만 (§7-1)
  CMP_FINALIZE=1 python src/model_cmp.py    캐시에서 집계·판정만 (신규 실행 없음)
  CMP_NSEED=10 / CMP_ARMS=base_lgbm,svc     범위 제한
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "model_cmp"); CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

B, A = 0.95, 0.02
CLS = list(range(2, 65))
NTH = 300
# (5) 정책 격자 = **검사율 q**. 점수 척도·fit에 무관하므로 inner에서 고른 q가 outer에서 같은 운영점을 뜻한다.
QS = np.linspace(0.005, 0.995, NTH)
NSEED = int(os.environ.get("CMP_NSEED", 50))
PROBE = os.environ.get("CMP_PROBE") == "1"
FINALIZE = os.environ.get("CMP_FINALIZE") == "1"
PCA_VAR = 0.95
SVC_C_GRID = [0.1, 1.0, 10.0]
TIME_BUDGET_H = 6.0
MIN_SEED = 30                                # §7-3 부분완료 하한

X, y, _ = adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                         "Pass/Fail", 1, drop_cols=["Time"])
y = np.asarray(y); N = len(y); npos = int(y.sum()); nneg = N - npos
# ⚠️ (1) 전역 열 필터 없음 — 상수열 제거는 fold 내부 VarianceThreshold가 한다
Xraw = X.to_numpy(dtype=float)

ARMS = {"base_lgbm":     ("lgbm", False, False),
        "svc":           ("svc",  False, False),
        "pca_rf":        ("rf",   True,  False),
        "pca_svc":       ("svc",  True,  False),
        "svc_smote":     ("svc",  False, True),
        "pca_svc_smote": ("svc",  True,  True)}
SEL = os.environ.get("CMP_ARMS")
if SEL: ARMS = {k: v for k, v in ARMS.items() if k in SEL.split(",")}


def prep(Xtr, Xte):
    """(1) 대치→분산필터→표준화. **전부 Xtr에서만 fit.** 호출될 때마다 새로 적합한다."""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    a, b = imp.transform(Xtr), imp.transform(Xte)
    vt = VarianceThreshold(0.0).fit(a); a, b = vt.transform(a), vt.transform(b)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b)


def mk_model(kind, seed, spw, C=1.0):
    if kind == "lgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
                              scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    return SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", random_state=seed)


def fit_predict(kind, use_pca, use_smote, Xtr, ytr, Xte, seed, C=1.0):
    """원본 배열을 받아 **fold 내부에서** 전처리·PCA·SMOTE·적합. 반환: te 점수, PCA 성분수."""
    Ztr, Zte = prep(Xtr, Xte)
    ncomp = None
    if use_pca:
        p = PCA(n_components=PCA_VAR, svd_solver="full", random_state=seed).fit(Ztr)
        Ztr, Zte, ncomp = p.transform(Ztr), p.transform(Zte), int(p.n_components_)
    Xf, yf = Ztr, ytr
    if use_smote:                     # 🔴 학습에만. 검증·테스트에는 적용하지 않는다
        Xf, yf = SMOTE(k_neighbors=5, sampling_strategy=1.0, random_state=seed).fit_resample(Ztr, ytr)
    m = mk_model(kind, seed, (yf == 0).sum() / max(yf.sum(), 1), C).fit(Xf, yf)
    s = m.predict_proba(Zte)[:, 1] if kind in ("lgbm", "rf") else m.decision_function(Zte)
    return s, ncomp


def inner_eval(kind, use_pca, use_smote, Xtr, ytr, seed, C):
    """(8) inner 3-fold를 **fold별로** 평가한다. 서로 다른 fit의 점수를 합쳐 정렬하지 않는다.

    반환: (fold AUC 평균, q별 혼동행렬 **합**, 실제 검사율(가중평균), PCA 성분수 목록)"""
    aucs = []; cmsum = np.zeros((NTH, 4), dtype=np.int64); ktot = 0; ntot = 0; nc = []
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
        v, c = fit_predict(kind, use_pca, use_smote, Xtr[a], ytr[a], Xtr[b], seed, C)
        aucs.append(roc_auc_score(ytr[b], v))       # fold별 AUC (pooled 금지)
        cm, rate = cms_rank(v, ytr[b])              # fold **내부**에서 상위 q%
        cmsum += cm; ktot = ktot + rate * len(b); ntot += len(b)
        if c: nc.append(c)
    return float(np.mean(aucs)), cmsum, ktot / max(ntot, 1), nc


def pick_C(kind, use_pca, use_smote, Xtr, ytr, seed):
    """§2.1 C를 **inner fold AUC 평균**으로 선택(nested). SVC 전용."""
    if kind != "svc": return 1.0
    best, best_a = 1.0, -1.0
    for C in SVC_C_GRID:
        a = inner_eval(kind, use_pca, use_smote, Xtr, ytr, seed, C)[0]
        if a > best_a: best, best_a = C, a
    return best


def cms_rank(score, yy):
    """(5)(9) 순위 정책: 점수 상위 q% 검사. **cutoff 동점은 전부 포함**한다.

    반환: (cm (NTH,4)=[tn,fp,fn,tp],  rate (NTH,) 실제 검사율)
    q는 한 fit **내부에서** 척도 불변이다. 서로 다른 fit의 점수를 합쳐 정렬하지 않는다((8) 참조).
    동점을 안정 정렬로 끊으면 원본 행 순서(=생산 순서)가 개입하므로, 대신 **동점을 모두 포함**하고
    실제 검사율을 기록한다 → q는 '목표 검사율', 실제는 소폭 클 수 있다."""
    o = np.argsort(-score, kind="stable")
    ss = score[o]; ys = yy[o].astype(int)
    n = len(ys); P = int(ys.sum())
    cum = np.concatenate([[0], np.cumsum(ys)])
    k0 = np.clip(np.round(QS * n).astype(int), 1, n)
    v = ss[k0 - 1]                                  # 목표 지점의 점수값
    k = np.searchsorted(-ss, -v, side="right")      # 그 값과 동점인 표본을 전부 포함
    tp = cum[k]; fp = k - tp; fn = P - tp; tn = (n - k) - fn
    return np.stack([tn, fp, fn, tp], 1).astype(int), k / n


def best_cost(folds, cl):
    lam, mu, rho = CM.to_effective(cl, 0, B, A)
    sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
    pass_all = npos * cl; inspect_all = N + npos * cl - npos * lam + nneg * mu
    rate = (agg[3] + agg[1]) / N
    return min(pass_all, inspect_all, inspect_all if rate >= CM.DEGEN_RATE else sel)


def run_seed(arm, seed):
    kind, use_pca, use_smote = ARMS[arm]
    lam15, mu15, rho15 = CM.to_effective(15, 0, B, A)        # cl=15 정책 기록용
    folds = []; pooled = np.zeros(N); fold_auc = []; comps = []; picks = []
    q_sel = []; rate_act = []; rate_in = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xraw, y):
        Xtr, Xte, ytr = Xraw[tr], Xraw[te], y[tr]
        C = pick_C(kind, use_pca, use_smote, Xtr, ytr, seed); picks.append(C)
        # (8) inner: fold별 AUC·혼동행렬. margin을 합쳐 전역 정렬하지 않는다
        _, in_cm, in_rate, nc1 = inner_eval(kind, use_pca, use_smote, Xtr, ytr, seed, C)
        te_s, nc2 = fit_predict(kind, use_pca, use_smote, Xtr, ytr, Xte, seed, C)
        te_cm, te_rate = cms_rank(te_s, y[te])               # (5) 같은 q 격자, fit 내부 순위
        comps += nc1 + ([nc2] if nc2 else [])
        folds.append((in_cm, te_cm))
        fold_auc.append(roc_auc_score(y[te], te_s))          # (3) fold별 AUC
        pooled[te] = te_s
        # cl=15에서 커널이 고른 정책 인덱스 → 목표 q와 실제 검사율 기록
        j = CM.best_threshold_idx(in_cm, lam15, mu15, rho15)
        q_sel.append(QS[j]); rate_act.append(te_rate[j]); rate_in.append(in_rate[j])
    return dict(fold_auc=np.array(fold_auc),                 # (3) 주지표
                pooled_auc=roc_auc_score(y, pooled),         # (3) 참고값
                save=CM.evaluate(folds, N, npos, nneg, cl=15, cs=0, beta=B, alpha=A)["sel_vs_inspect"],
                costs=np.array([best_cost(folds, cl) for cl in CLS]),
                comps=np.array(comps if comps else [0]), picks=np.array(picks),
                q_sel_cl15=np.array(q_sel),                  # 선택된 목표 q (fold별)
                rate_out_cl15=np.array(rate_act),            # outer 실제 검사율 (동점 포함)
                rate_in_cl15=np.array(rate_in))              # inner 실제 검사율


def save_atomic(path, d):
    """(4) 원자적 저장 — 중단 시 반쪽 파일이 남지 않게 tmp에 쓰고 교체."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d); os.replace(tmp, path)


def cache_path(arm, s): return os.path.join(CACHE, f"{arm}_s{s}.npz")


def load_cached(arm, s):
    """(7) 핸들을 즉시 닫고 배열만 반환한다."""
    p = cache_path(arm, s)
    if not os.path.exists(p): return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def common_prefix(arms):
    """(4) 모든 arm이 공통으로 가진 **연속** seed 개수 (인덱스 0부터). 결과 기반 선별 금지."""
    k = 0
    while k < NSEED and all(os.path.exists(cache_path(a, k)) for a in arms): k += 1
    return k


# ══════════ 집계·판정 ══════════
def finalize():
    n = common_prefix(list(ARMS))
    print(f"[집계] 공통 연속 seed = {n}")
    if "base_lgbm" not in ARMS:
        sys.exit("🔴 base_lgbm이 없으면 대응 비교 불가")
    if n < MIN_SEED:
        json.dump({"판정": "시도·미완", "공통연속seed": n, "하한": MIN_SEED,
                   "처리": "사전등록 §7-3에 따라 판정하지 않는다. 탐색공간·미완 사유를 부록에 기재."},
                  open(os.path.join(OUT, "model_cmp.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        sys.exit(f"🔴 공통 연속 seed {n} < {MIN_SEED} — 판정하지 않음(§7-3). 결과 파일에 '시도·미완' 기록.")

    R = {}
    for a in ARMS:
        d = [load_cached(a, s) for s in range(n)]
        R[a] = dict(fa=np.array([x["fold_auc"].mean() for x in d]),
                    pa=np.array([float(x["pooled_auc"]) for x in d]),
                    sv=np.array([float(x["save"]) for x in d]),
                    co=np.array([x["costs"] for x in d]),
                    cm=np.concatenate([x["comps"] for x in d]),
                    pk=np.concatenate([x["picks"] for x in d]),
                    q=np.concatenate([x["q_sel_cl15"] for x in d]),
                    ro=np.concatenate([x["rate_out_cl15"] for x in d]),
                    ri=np.concatenate([x["rate_in_cl15"] for x in d]))
    TH = json.load(open(os.path.join(OUT, "thresholds.json"), encoding="utf-8"))
    cost_pct = TH["대응_비용차"]["권고_실질개선_기준_%"]
    base = R["base_lgbm"]; i1020 = [CLS.index(c) for c in range(10, 21)]
    out = {"프로토콜": f"동일 {n} seed·동일 fold 대응 설계. 사전등록_모델계열비교.md 집행.",
           "AUC_정의": "주지표=outer fold별 AUC의 seed 내 평균. pooled OOF는 참고값(SVC는 fold별 스케일 상이).",
           "판정임계": TH["최종_판정기준"], "n_seed": n, "arms": {}}

    def ci(v):
        h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return float(v.mean()), float(v.mean() - h), float(v.mean() + h)

    for a in ARMS:
        r = R[a]; dA = r["fa"] - base["fa"]
        dC = ((base["co"][:, i1020] - r["co"][:, i1020]) / np.maximum(base["co"][:, i1020], 1) * 100).mean(1)
        mA, loA, hiA = ci(dA); mC, loC, hiC = ci(dC)
        try: p = float(wilcoxon(dA).pvalue) if a != "base_lgbm" and np.any(dA != 0) else 1.0
        except Exception: p = 1.0
        ok_as, ok_am = (a != "base_lgbm" and loA > 0), (mA >= 0.02)
        ok_cs, ok_cm = (a != "base_lgbm" and loC > 0), (mC >= cost_pct)
        out["arms"][a] = {
            "AUC_foldmean": [round(float(r["fa"].mean()), 4), round(float(r["fa"].std(ddof=1)), 4)],
            "AUC_pooled(참고)": round(float(r["pa"].mean()), 4),
            "대응_ΔAUC": {"평균": round(mA, 4), "95%CI": [round(loA, 4), round(hiA, 4)],
                       "wilcoxon_p": round(p, 4), "우세_seed": f"{int((dA > 0).sum())}/{n}",
                       "seed별": [round(float(v), 4) for v in dA]},
            "cl15_절감률_%": round(float(r["sv"].mean()), 1),
            "비용_cl10~20_대응Δ%": {"평균": round(mC, 2), "95%CI": [round(loC, 2), round(hiC, 2)]},
            "PCA_성분수": ([int(r["cm"].mean()), int(r["cm"].min()), int(r["cm"].max())]
                       if r["cm"].max() > 0 else None),
            "SVC_선택C_분포": ({str(c): int((r["pk"] == c).sum()) for c in SVC_C_GRID}
                          if ARMS[a][0] == "svc" else None),
            "cl15_정책": {"선택_목표q_중앙": round(float(np.median(r["q"])), 4),
                       "선택_목표q_IQR": [round(float(np.percentile(r["q"], 25)), 4),
                                       round(float(np.percentile(r["q"], 75)), 4)],
                       "실제검사율_outer_중앙": round(float(np.median(r["ro"])), 4),
                       "실제검사율_inner_중앙": round(float(np.median(r["ri"])), 4),
                       "동점_초과_%p_중앙": round(float(np.median(r["ro"] - r["q"]) * 100), 3),
                       "동점_초과_%p_최대": round(float(np.max(r["ro"] - r["q"]) * 100), 3)},
            "판정": ("기준선" if a == "base_lgbm" else
                   "교체검토 — walk-forward 확인 필요(§4-bis)" if (ok_as and ok_am and ok_cs and ok_cm) else
                   "비교보고만 — AUC 개선하나 비용 기준 미달" if (ok_as and ok_am) else
                   "유의하나 실질적이지 않음" if ok_as else "개선 검출되지 않음")}
        if a != "base_lgbm":
            print(f"[{a:<14s}] ΔAUC {mA:+.4f} CI[{loA:+.4f},{hiA:+.4f}] | "
                  f"비용Δ {mC:+.2f}% (기준 {cost_pct}%) → {out['arms'][a]['판정']}")
    out["⚠️전방검증"] = "'교체검토' arm에 한해 사전등록 §4-bis 규칙으로 별도 실행 후 −0.015 기준 확인."
    out["⚠️주의"] = ("통계적 유의성은 대응 CI로만 판단. 개별 모델 seed SD는 판정에 사용하지 않음. "
               "정책은 순위 기반(상위 q%)이며 q는 **목표** 검사율 — cutoff 동점을 전부 포함하므로 "
               "실제 검사율은 소폭 크다(동점_초과 항목 참조). base_lgbm도 이 정책으로 평가되므로 정본 수치와 직접 비교 금지.")
    json.dump(out, open(os.path.join(OUT, "model_cmp.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ results/model_cmp/model_cmp.json")


if FINALIZE:
    finalize(); sys.exit(0)

# ══════════ 시간 프로브 (§7-1) ══════════
print(f"[프로브] arm {len(ARMS)}개 × {NSEED} seed | 예산 {TIME_BUDGET_H}h")
proj = 0.0; pr = {}
for a in ARMS:
    t0 = time.time(); run_seed(a, 0); dt = time.time() - t0
    proj += dt * NSEED; pr[a] = round(dt, 1)
    print(f"  {a:<16s} seed 1개 {dt:6.1f}s → {NSEED} seed 예상 {dt*NSEED/60:6.1f}분", flush=True)
print(f"  총 예상 {proj/3600:.2f}시간")
if PROBE:
    json.dump({"seed1_초": pr, "총예상_시간": round(proj / 3600, 2), "예산_시간": TIME_BUDGET_H,
               "판정": "예산 내" if proj / 3600 <= TIME_BUDGET_H else "예산 초과 — arm 축소 승인 필요"},
              open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0)
if proj / 3600 > TIME_BUDGET_H:
    sys.exit(f"🔴 예상 {proj/3600:.1f}h > 예산 {TIME_BUDGET_H}h — §7-1에 따라 중단. arm 축소는 승인받는다.")

# ══════════ 본 실행 (캐시·재개, §7-2) ══════════
# (6) seed 우선 — 중단 시에도 모든 arm의 공통 연속 seed가 쌓인다
t0 = time.time()
for s in range(NSEED):
    for a in ARMS:
        if os.path.exists(cache_path(a, s)): continue
        save_atomic(cache_path(a, s), run_seed(a, s))
        print(f"  seed {s+1}/{NSEED} [{a}] 완료 ({time.time()-t0:.0f}s)", flush=True)
finalize()
