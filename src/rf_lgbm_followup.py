# -*- coding: utf-8 -*-
"""
rf_lgbm_followup.py — 일반 RF vs LightGBM 후속 확인실험.
**`사전등록_RF_vs_LGBM_후속.md`를 그대로 집행한다.** (판정 정본 = results/model_cmp/thresholds.json 승계)

⚠️ 격리: 기존 model_cmp 코드·사전등록·캐시·thresholds·정본 문서를 일절 수정하지 않는다.
        출력은 results/rf_lgbm_followup/ 아래에만 생성한다.
⚠️ import 안전: 이 모듈을 import해도 데이터 로딩·모델 학습이 일어나지 않는다.
        모든 동작은 `if __name__ == "__main__"` 아래 CLI 모드로만 시작된다.

모드:
  --prepare        해시·환경 수집 → run_manifest 초안 검증(신규 실행 없음)
  --probe          3 seed(0,1,2) × 2 arm 시간/메모리 프로브  [사전등록 §17]
  --run            50 seed 본 실행 (프로브 통과 + --cut 필수)  [§16 중단 조건]
  --finalize       캐시 검증 → 집계 → thresholds.json 원문 판정
  --walk-forward   AUC·비용 PASS 시에만 열림 (§13). 미충족 시 즉시 중단
  --verify-cache   캐시 무결성·해시 일치 검사만 (읽기 전용)

arm은 정확히 2개: base_lgbm(model_cmp 정본 설정) / rf_no_pca(seed_flip 400/balanced_subsample/leaf3).
PCA·SMOTE·ADASYN 없음. 기존 20-seed 결과와 통계 병합 금지(사전등록 §14).
"""
import os, sys, json, time, hashlib, warnings, argparse, datetime

# ── 경로 상수 (import 시 부작용 없음 — 디렉터리 생성도 모드 진입 후에만) ──
SRC   = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(SRC)
COMP  = os.path.dirname(ROOT)
OUT   = os.path.join(ROOT, "results", "rf_lgbm_followup")
CACHE = os.path.join(OUT, "cache")
DATA_CSV   = os.path.join(ROOT, "data", "fab_process_yield.csv")
PREREG_MD  = os.path.join(ROOT, "docs", "prereg", "사전등록_RF_vs_LGBM_후속.md")
THRESHOLDS = os.path.join(ROOT, "results", "model_cmp", "thresholds.json")

EXPERIMENT_ID = "rf_lgbm_followup_v1"
NSEED       = 50                 # 고정 (사전등록 §9) — 변경 금지
NFOLD_OUTER = 5
NFOLD_INNER = 3
MIN_SEED    = 30                 # 부분완료 하한 (§18, model_cmp §7-3 승계)
PROBE_SEEDS = (0, 1, 2)          # §17
TIME_BUDGET_H = 6.0              # §17 (model_cmp §7-1 승계)
CUT_MARGIN_MIN = 30              # 컷 30분 전 안전 완료 불가 시 중단 (§16)
MEM_FRAC_LIMIT = 0.80            # 가용 메모리 80% 초과 시 중단 (§16)
SLOW_FACTOR = 2.0                # seed당 시간이 프로브 대비 2배↑면 비정상 (§16)

B, A, CS = 0.95, 0.02, 0        # 비용 커널 상수 — model_cmp §3 승계
CLS = list(range(2, 65))         # cl 2~64
I1020 = [CLS.index(c) for c in range(10, 21)]   # 판정 축 cl 10~20 (thresholds 원문)
NTH = 300
CL_POLICY = 15                   # cl=15 정책 기록용 (model_cmp 동일)

# 스레드 = 실행환경 설정(통계 파라미터 아님, §16). manifest에 실제값 기록.
NJOBS = int(os.environ.get("RFL_NJOBS", "1"))

ARMS = ("base_lgbm", "rf_no_pca")   # 고정 2 arm — 추가 금지 (§4)

# ── 모델 설정: 기본값 의존 제거 위해 전부 명시 (사전등록 §5·§6) ──
RF_NO_PCA_PARAMS = dict(          # 귀속: src/seed_flip.py:40-41 (+0.023 우위 구성)
    n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=3,
    criterion="gini", max_depth=None, max_features="sqrt", min_samples_split=2,
    bootstrap=True, max_leaf_nodes=None, min_weight_fraction_leaf=0.0,
    min_impurity_decrease=0.0, oob_score=False, ccp_alpha=0.0, max_samples=None,
    random_state=0,               # seed_flip 원문: 모델 rs=0 고정(seed는 CV 분할만) — §5 ⚠️
)
BASE_LGBM_PARAMS = dict(          # 귀속: src/model_cmp.py:101-103 (scale_pos_weight·random_state는 런타임)
    boosting_type="gbdt", n_estimators=300, learning_rate=0.03, num_leaves=7,
    max_depth=-1, min_child_samples=25, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.3, reg_lambda=10.0, verbose=-1,
)
PREP_SPEC = "SimpleImputer(median) -> VarianceThreshold(0.0) -> StandardScaler (fit: training subset only)"


# ══════════ 해시 유틸 ══════════
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj):
    """설정 dict/배열의 canonical JSON 해시."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def q_grid():
    import numpy as np
    return np.linspace(0.005, 0.995, NTH)


def config_hashes():
    """모델·전처리·q grid·비용 설정 해시 (캐시 유효성 판단 §15)."""
    return {
        "model_config_hash": sha256_obj({"rf_no_pca": RF_NO_PCA_PARAMS,
                                         "base_lgbm": BASE_LGBM_PARAMS,
                                         "lgbm_runtime": "scale_pos_weight=fold Nneg/Npos, random_state=seed"}),
        "prep_hash":  sha256_obj(PREP_SPEC),
        "qgrid_hash": sha256_obj([round(float(v), 10) for v in q_grid()]),
        "cost_hash":  sha256_obj({"CLS": CLS, "cs": CS, "beta": B, "alpha": A,
                                  "판정축": "cl10~20 대응 비용차 (thresholds 원문)"}),
    }


def source_hashes():
    """실행 시점에 재계산해 대조하는 정본 해시 (§21). 누락·불일치 시 중단."""
    req = {"code": os.path.abspath(__file__), "data": DATA_CSV,
           "prereg": PREREG_MD, "thresholds": THRESHOLDS}
    out = {}
    for k, p in req.items():
        if not os.path.exists(p):
            die(f"필수 파일 없음: {k} = {p} — 해시 계산 불가, 중단(사전등록 §21)")
        out[k + "_sha256"] = sha256_file(p)
    return out


def env_info():
    import numpy, pandas, sklearn, lightgbm, scipy
    return {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "n_jobs(실행환경)": NJOBS}


def die(msg):
    print(f"🔴 중단: {msg}", flush=True)
    sys.exit(1)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ══════════ 데이터·fold (모드 진입 후에만 호출) ══════════
def load_data():
    """정본 로딩 경로 그대로 (model_cmp 동일). 전역 열 필터 없음."""
    import numpy as np
    sys.path.insert(0, SRC)
    from preprocess_adapter import adapt_from_csv
    X, y, _ = adapt_from_csv(DATA_CSV, "Pass/Fail", 1, drop_cols=["Time"])
    y = np.asarray(y)
    return X.to_numpy(dtype=float), y


def outer_folds(y, seed):
    """두 arm이 공유하는 outer 분할 (§9). 호출마다 동일 재현."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    idx = np.arange(len(y))
    return list(StratifiedKFold(NFOLD_OUTER, shuffle=True, random_state=seed).split(idx, y))


def fold_index_hash(y, seed):
    folds = outer_folds(y, seed)
    return sha256_obj([[tr.tolist(), te.tolist()] for tr, te in folds])


# ══════════ 파이프라인 (model_cmp §2.4·§3 승계 구현) ══════════
def prep(Xtr, Xte):
    """대치→분산필터→표준화. 전부 Xtr에서만 fit (§8)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    imp = SimpleImputer(strategy="median").fit(Xtr)
    a, b = imp.transform(Xtr), imp.transform(Xte)
    vt = VarianceThreshold(0.0).fit(a)
    a, b = vt.transform(a), vt.transform(b)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b)


def mk_model(arm, seed, spw):
    if arm == "base_lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**BASE_LGBM_PARAMS, scale_pos_weight=spw,
                              random_state=seed, n_jobs=NJOBS)
    if arm == "rf_no_pca":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_NO_PCA_PARAMS, n_jobs=NJOBS)   # rs=0은 파라미터에 포함(§5)
    die(f"알 수 없는 arm: {arm} — arm 고정 위반(사전등록 §4)")


def fit_predict(arm, Xtr, ytr, Xte, seed):
    """fold 내부 전처리 후 적합. 반환: te 위험점수(ranking score — 확률 표현 금지, §11)."""
    Ztr, Zte = prep(Xtr, Xte)
    spw = (ytr == 0).sum() / max(ytr.sum(), 1)
    m = mk_model(arm, seed, spw).fit(Ztr, ytr)
    return m.predict_proba(Zte)[:, 1]


def cms_rank(score, yy):
    """순위 정책: 상위 q% 검사, cutoff 동점 전부 포함 (model_cmp cms_rank 동일 구현)."""
    import numpy as np
    QS = q_grid()
    o = np.argsort(-score, kind="stable")
    ss = score[o]; ys = yy[o].astype(int)
    n = len(ys); P = int(ys.sum())
    cum = np.concatenate([[0], np.cumsum(ys)])
    k0 = np.clip(np.round(QS * n).astype(int), 1, n)
    v = ss[k0 - 1]
    k = np.searchsorted(-ss, -v, side="right")
    tp = cum[k]; fp = k - tp; fn = P - tp; tn = (n - k) - fn
    return np.stack([tn, fp, fn, tp], 1).astype(int), k / n


def inner_eval(arm, Xtr, ytr, seed):
    """inner 3-fold를 fold별로 평가. AUC=fold 평균 / 비용=q별 혼동행렬 합 (§11 혼용 금지)."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    aucs = []; cmsum = np.zeros((NTH, 4), dtype=np.int64)
    for a, b in StratifiedKFold(NFOLD_INNER, shuffle=True, random_state=seed).split(Xtr, ytr):
        v = fit_predict(arm, Xtr[a], ytr[a], Xtr[b], seed)
        aucs.append(roc_auc_score(ytr[b], v))
        cm, _ = cms_rank(v, ytr[b])
        cmsum += cm
    # inner 실제 검사율은 정책 인덱스 확정 후 outer에서 기록 — fold 합 행렬만 반환 (§11)
    return float(np.mean(aucs)), cmsum


def run_seed(arm, seed, X, y):
    """한 (arm, seed)의 전체 계산. 반환 dict는 캐시에 저장된다."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, SRC)
    import cost_model as CM
    N = len(y); npos = int(y.sum()); nneg = N - npos
    QS = q_grid()
    lam15, mu15, rho15 = CM.to_effective(CL_POLICY, CS, B, A)
    folds = []; pooled = np.zeros(N); fold_auc = []
    q_sel = []; rate_act = []
    for tr, te in outer_folds(y, seed):
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        _, in_cm = inner_eval(arm, Xtr, ytr, seed)
        te_s = fit_predict(arm, Xtr, ytr, Xte, seed)
        te_cm, te_rate = cms_rank(te_s, y[te])
        folds.append((in_cm, te_cm))
        fold_auc.append(roc_auc_score(y[te], te_s))
        pooled[te] = te_s
        j = CM.best_threshold_idx(in_cm, lam15, mu15, rho15)   # 정책 선택 = inner만 (§8 금지 준수)
        q_sel.append(QS[j]); rate_act.append(te_rate[j])

    def best_cost(cl):
        lam, mu, rho = CM.to_effective(cl, CS, B, A)
        sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
        pass_all = npos * cl
        inspect_all = N + npos * cl - npos * lam + nneg * mu
        rate = (agg[3] + agg[1]) / N
        return min(pass_all, inspect_all, inspect_all if rate >= CM.DEGEN_RATE else sel)

    return dict(fold_auc=np.array(fold_auc),
                pooled_auc=roc_auc_score(y, pooled),      # 참고지표 (§10 — 판정 사용 금지)
                save=CM.evaluate(folds, N, npos, nneg, cl=CL_POLICY, cs=CS, beta=B, alpha=A)["sel_vs_inspect"],
                costs=np.array([best_cost(cl) for cl in CLS]),
                q_sel_cl15=np.array(q_sel), rate_out_cl15=np.array(rate_act))


# ══════════ 캐시 (해시 검증 포함, §15) ══════════
def cache_paths(arm, seed):
    return (os.path.join(CACHE, f"{arm}_s{seed}.npz"),
            os.path.join(CACHE, f"{arm}_s{seed}.meta.json"))


def write_cache(arm, seed, result, src_h, cfg_h, fih, t_sec):
    """tmp 기록 → flush → 완료 상태 → atomic rename → 재검증 (§15)."""
    import numpy as np
    npz_path, meta_path = cache_paths(arm, seed)
    tmp_npz, tmp_meta = npz_path + ".tmp", meta_path + ".tmp"
    with open(tmp_npz, "wb") as f:
        np.savez(f, **result); f.flush(); os.fsync(f.fileno())
    meta = {"experiment_id": EXPERIMENT_ID, "arm": arm, "seed": seed,
            "outer_fold_spec": f"StratifiedKFold({NFOLD_OUTER},shuffle,rs=seed)",
            "inner_fold_spec": f"StratifiedKFold({NFOLD_INNER},shuffle,rs=seed)",
            "fold_index_hash": fih, "status": "complete",
            "created_at": now_iso(), "elapsed_sec": round(t_sec, 1),
            "env": env_info(), **src_h, **cfg_h,
            "exception": None}
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp_npz, npz_path); os.replace(tmp_meta, meta_path)
    with np.load(npz_path) as z:                          # 최종 파일 검증
        assert "fold_auc" in z.files, "캐시 재검증 실패"
    return npz_path


def cache_valid(arm, seed, src_h, cfg_h, fih):
    """모든 해시·상태 일치 시에만 재사용 (§15 — 파일명만으로 판단 금지)."""
    npz_path, meta_path = cache_paths(arm, seed)
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        return False, "파일 없음"
    try:
        meta = json.load(open(meta_path, encoding="utf-8"))
    except Exception as e:
        return False, f"meta 손상: {e}"
    checks = {"experiment_id": EXPERIMENT_ID, "arm": arm, "seed": seed,
              "status": "complete", "fold_index_hash": fih, **src_h, **cfg_h}
    for k, want in checks.items():
        if meta.get(k) != want:
            return False, f"불일치 {k}"
    return True, "ok"


def load_cache(arm, seed):
    import numpy as np
    with np.load(cache_paths(arm, seed)[0]) as z:
        return {k: z[k] for k in z.files}


# ══════════ 공통 가드 ══════════
def guard_common(need_dirs=False):
    """모든 실행 모드 공통: 정본 존재·해시 계산·arm/seed/grid 고정 확인."""
    if tuple(ARMS) != ("base_lgbm", "rf_no_pca"):
        die("arm 고정 위반")
    if NSEED != 50 or NFOLD_OUTER != 5 or NFOLD_INNER != 3 or NTH != 300:
        die("seed/fold/q grid 고정 위반 — 사전등록과 다름")
    src_h = source_hashes()                      # 누락 시 내부에서 중단
    cfg_h = config_hashes()
    th = json.load(open(THRESHOLDS, encoding="utf-8"))
    if "최종_판정기준" not in th:
        die("thresholds.json에 최종_판정기준 없음 — 판정 정본 아님")
    if need_dirs:
        os.makedirs(CACHE, exist_ok=True)
    return src_h, cfg_h, th


def no_overwrite(path):
    if os.path.exists(path):
        die(f"기존 파일 덮어쓰기 금지: {path} (지우거나 이동하지 않는다 — 책임자 승인 필요)")


def mem_guard():
    """가용 메모리 80% 초과 시 True(중단 신호). psutil 없으면 검사 불가를 기록만."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.percent >= MEM_FRAC_LIMIT * 100, f"mem {vm.percent:.0f}%"
    except ImportError:
        return False, "psutil 없음 — 메모리 가드 비활성(manifest에 기록)"


# ══════════ 모드 구현 ══════════
def mode_prepare():
    """해시·환경 수집, manifest 초안 콘솔 출력. 신규 실행 없음. 파일 생성 없음."""
    src_h, cfg_h, th = guard_common()
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "env": env_info(),
                      **src_h, **cfg_h,
                      "판정기준(원문 승계)": th["최종_판정기준"]},
                     ensure_ascii=False, indent=1))
    print("✅ prepare 검증 통과 — 실행은 하지 않았음")


def mode_probe():
    src_h, cfg_h, _ = guard_common(need_dirs=True)
    probe_path = os.path.join(OUT, "probe_result.json")
    no_overwrite(probe_path)
    X, y = load_data()
    times = {}
    for arm in ARMS:
        per = []
        for s in PROBE_SEEDS:
            fih = fold_index_hash(y, s)
            ok, why = cache_valid(arm, s, src_h, cfg_h, fih)
            t0 = time.time()
            if not ok:
                bad, msg = mem_guard()
                if bad: die(f"메모리 한계: {msg}")
                res = run_seed(arm, s, X, y)
                write_cache(arm, s, res, src_h, cfg_h, fih, time.time() - t0)
            per.append(time.time() - t0)
            print(f"  probe [{arm}] seed {s}: {per[-1]:.1f}s", flush=True)
        times[arm] = per
    proj_h = sum(sum(v) / len(v) * NSEED for v in times.values()) / 3600
    verdict = "통과" if proj_h <= TIME_BUDGET_H else "예산 초과 — 본 실행 금지, 승인 필요"
    out = {"experiment_id": EXPERIMENT_ID, "probe_seeds": list(PROBE_SEEDS),
           "seed당_초": {a: [round(x, 1) for x in v] for a, v in times.items()},
           "50seed_예상_시간_h": round(proj_h, 2), "예산_h": TIME_BUDGET_H,
           "판정": verdict, "created_at": now_iso(), **src_h}
    json.dump(out, open(probe_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[프로브] 예상 {proj_h:.2f}h / 예산 {TIME_BUDGET_H}h → {verdict}")


def mode_run(cut_iso):
    src_h, cfg_h, _ = guard_common(need_dirs=True)
    probe_path = os.path.join(OUT, "probe_result.json")
    if not os.path.exists(probe_path):
        die("프로브 미완료 — --probe 먼저 (사전등록 §17)")
    probe = json.load(open(probe_path, encoding="utf-8"))
    if probe.get("판정") != "통과":
        die(f"프로브 미통과({probe.get('판정')}) — 본 실행 금지")
    for k, v in src_h.items():                              # 프로브 시점과 해시 동일해야 함
        if probe.get(k) != v:
            die(f"프로브 이후 {k} 변경됨 — 프로브 재실행 필요")
    if not cut_iso:
        die("--cut <ISO시각> 필수 — 실행 컷 없이 본 실행 금지 (사전등록 §16)")
    cut = datetime.datetime.fromisoformat(cut_iso)
    if cut.tzinfo is None:
        cut = cut.astimezone()
    t_arm = {a: sum(v) / len(v) for a, v in probe["seed당_초"].items()}   # arm별 프로브 평균(초)
    X, y = load_data()
    manifest_path = os.path.join(OUT, "run_manifest.json")
    manifest = {"experiment_id": EXPERIMENT_ID, "started_at": now_iso(), "cut": cut_iso,
                "env": env_info(), **src_h, **cfg_h, "seeds_done": [], "seeds_failed": [],
                "중단": None}
    for s in range(NSEED):
        fih = fold_index_hash(y, s)
        for arm in ARMS:                                    # seed 우선 — 공통 연속 seed 축적
            ok, _ = cache_valid(arm, s, src_h, cfg_h, fih)
            if ok: continue
            now = datetime.datetime.now().astimezone()
            eta = now + datetime.timedelta(seconds=t_arm[arm])
            if eta + datetime.timedelta(minutes=CUT_MARGIN_MIN) > cut:
                manifest["중단"] = f"컷 30분 전 안전 완료 불가 (seed {s})"
                break
            bad, msg = mem_guard()
            if bad:
                manifest["중단"] = f"메모리 한계 {msg} (seed {s})"; break
            t0 = time.time()
            try:
                res = run_seed(arm, s, X, y)
                dt = time.time() - t0
                write_cache(arm, s, res, src_h, cfg_h, fih, dt)   # 완료분은 보존(재개 가능)
                print(f"  seed {s+1}/{NSEED} [{arm}] {dt:.0f}s", flush=True)
                if dt > SLOW_FACTOR * max(t_arm[arm], 1):
                    manifest["중단"] = f"seed당 시간 비정상 증가 [{arm}] {dt:.0f}s vs 프로브 {t_arm[arm]:.0f}s (seed {s})"
                    break                                    # §16: 비정상 증가 → 안전 중단
            except MemoryError:
                manifest["seeds_failed"].append({"arm": arm, "seed": s, "err": "MemoryError"})
                manifest["중단"] = f"OOM (seed {s})"; break
            except Exception as e:
                manifest["seeds_failed"].append({"arm": arm, "seed": s, "err": repr(e)})
                manifest["중단"] = f"비정상 예외 (seed {s})"; break
        else:
            manifest["seeds_done"].append(s)
            continue
        break                                               # 안쪽 중단 → 바깥도 중단
    manifest["finished_at"] = now_iso()
    tmp = manifest_path + ".tmp"
    json.dump(manifest, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, manifest_path)
    print(f"→ {manifest_path}" + (f"  🔴 중단: {manifest['중단']}" if manifest["중단"] else "  ✅ 완주"))


def _common_prefix(y, src_h, cfg_h):
    k = 0
    while k < NSEED:
        fih = fold_index_hash(y, k)
        if not all(cache_valid(a, k, src_h, cfg_h, fih)[0] for a in ARMS):
            break
        k += 1
    return k


def mode_finalize():
    import numpy as np
    from scipy.stats import wilcoxon, t as tdist
    src_h, cfg_h, th = guard_common()
    X, y = load_data()
    n = _common_prefix(y, src_h, cfg_h)
    print(f"[집계] 공통 연속 유효 seed = {n}")
    out_json = os.path.join(OUT, "rf_lgbm_followup.json")
    no_overwrite(out_json)
    if n < MIN_SEED:
        json.dump({"판정": "시도·미완", "공통연속seed": n, "하한": MIN_SEED,
                   "처리": "사전등록 §18에 따라 판정하지 않는다."},
                  open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        die(f"공통 연속 seed {n} < {MIN_SEED} — 판정하지 않음(§18)")

    R = {}
    for a in ARMS:
        d = [load_cache(a, s) for s in range(n)]
        R[a] = dict(fa=np.array([x["fold_auc"].mean() for x in d]),
                    fold_raw=[x["fold_auc"].tolist() for x in d],
                    pa=np.array([float(x["pooled_auc"]) for x in d]),
                    sv=np.array([float(x["save"]) for x in d]),
                    co=np.array([x["costs"] for x in d]),
                    q=np.concatenate([x["q_sel_cl15"] for x in d]),
                    ro=np.concatenate([x["rate_out_cl15"] for x in d]))

    base, arm = R["base_lgbm"], R["rf_no_pca"]
    dA = arm["fa"] - base["fa"]                             # paired ΔAUC = RF − LGBM
    dC = ((base["co"][:, I1020] - arm["co"][:, I1020])
          / np.maximum(base["co"][:, I1020], 1) * 100).mean(1)

    def ci(v):
        h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return float(v.mean()), float(v.mean() - h), float(v.mean() + h)

    mA, loA, hiA = ci(dA); mC, loC, hiC = ci(dC)
    p_raw = float(wilcoxon(dA).pvalue) if np.any(dA != 0) else 1.0
    crit = th["최종_판정기준"]
    cost_pct = th["대응_비용차"]["권고_실질개선_기준_%"]      # 파일에서 직접 읽음 — 재유도 금지
    auc_pass = bool(loA > 0 and mA >= 0.02)
    cost_pass = bool(loC > 0 and mC >= cost_pct)

    per_seed_csv = os.path.join(OUT, "per_seed_results.csv")
    per_fold_csv = os.path.join(OUT, "per_fold_results.csv")
    cost_csv = os.path.join(OUT, "cost_by_ratio.csv")
    for p in (per_seed_csv, per_fold_csv, cost_csv):
        no_overwrite(p)
    with open(per_seed_csv, "w", encoding="utf-8") as f:
        f.write("seed,lgbm_foldmean_auc,rf_foldmean_auc,dA,lgbm_pooled,rf_pooled\n")
        for s in range(n):
            f.write(f"{s},{base['fa'][s]:.6f},{arm['fa'][s]:.6f},{dA[s]:.6f},"
                    f"{base['pa'][s]:.6f},{arm['pa'][s]:.6f}\n")
    with open(per_fold_csv, "w", encoding="utf-8") as f:
        f.write("seed,fold,lgbm_auc,rf_auc\n")
        for s in range(n):
            for k in range(NFOLD_OUTER):
                f.write(f"{s},{k},{base['fold_raw'][s][k]:.6f},{arm['fold_raw'][s][k]:.6f}\n")
    with open(cost_csv, "w", encoding="utf-8") as f:
        f.write("cl,lgbm_cost_mean,rf_cost_mean,paired_diff_mean,improve_pct_mean,ci_lo,ci_hi\n")
        for i, cl in enumerate(CLS):
            diff = base["co"][:, i] - arm["co"][:, i]
            imp = diff / np.maximum(base["co"][:, i], 1) * 100
            m_, lo_, hi_ = ci(imp)
            f.write(f"{cl},{base['co'][:,i].mean():.2f},{arm['co'][:,i].mean():.2f},"
                    f"{diff.mean():.2f},{m_:.3f},{lo_:.3f},{hi_:.3f}\n")

    out = {"experiment_id": EXPERIMENT_ID, "n_seed": n,
           "프로토콜": "동일 seed·동일 fold 대응 2-arm. 사전등록_RF_vs_LGBM_후속.md 집행. "
                    "판정 정본=thresholds.json 승계(재유도 없음).",
           "AUC_정의": "주지표=outer fold별 AUC의 seed 내 평균. pooled OOF는 참고값.",
           "판정임계(원문)": crit, **src_h,
           "base_lgbm": {"AUC_foldmean": [round(float(base["fa"].mean()), 4), round(float(base["fa"].std(ddof=1)), 4)],
                        "AUC_pooled(참고)": round(float(base["pa"].mean()), 4),
                        "cl15_절감률_%": round(float(base["sv"].mean()), 1)},
           "rf_no_pca": {"AUC_foldmean": [round(float(arm["fa"].mean()), 4), round(float(arm["fa"].std(ddof=1)), 4)],
                        "AUC_pooled(참고)": round(float(arm["pa"].mean()), 4),
                        "cl15_절감률_%": round(float(arm["sv"].mean()), 1)},
           "대응_ΔAUC(RF−LGBM)": {"평균": round(mA, 4), "95%CI": [round(loA, 4), round(hiA, 4)],
                               "sd": round(float(dA.std(ddof=1)), 4),
                               "우세_동률_열세": [int((dA > 0).sum()), int((dA == 0).sum()), int((dA < 0).sum())],
                               "wilcoxon_p_raw": p_raw, "wilcoxon_p_sci": f"{p_raw:.3e}",
                               "wilcoxon_p_표시": round(p_raw, 4),
                               "p_주의": "보조 통계 — thresholds 원문의 통과 조건 아님(사전등록 §12)",
                               "seed별": [round(float(v), 4) for v in dA]},
           "비용_cl10~20_대응Δ%": {"평균": round(mC, 2), "95%CI": [round(loC, 2), round(hiC, 2)],
                               "실질기준_%": cost_pct},
           "cl15_정책": {"q_중앙_lgbm": round(float(np.median(base["q"])), 4),
                      "q_중앙_rf": round(float(np.median(arm["q"])), 4),
                      "실검사율_중앙_lgbm": round(float(np.median(base["ro"])), 4),
                      "실검사율_중앙_rf": round(float(np.median(arm["ro"])), 4),
                      "동점초과_%p_중앙_rf": round(float(np.median(arm["ro"] - arm["q"]) * 100), 3),
                      "동점초과_%p_최대_rf": round(float(np.max(arm["ro"] - arm["q"]) * 100), 3)},
           "판정": {"AUC": "PASS" if auc_pass else "FAIL",
                  "비용": "PASS" if cost_pass else "FAIL",
                  "walk_forward": "실행조건 충족 — --walk-forward 가능" if (auc_pass and cost_pass)
                                 else "미실행 — AUC·비용 기준 미충족(사전등록 §13)",
                  "교체후보": bool(auc_pass and cost_pass) and "wf 통과 시 지정(3축 모두 필요)" or "아니오"},
           "⚠️비병합": "기존 20-seed 결과와 통계 병합 금지(사전등록 §14). 별도 표 병기만.",
           "⚠️해석범위": "본 판정은 사전등록된 2 arm에만 적용. 전 모델공간 확대 해석 금지(사전등록 §19)."}
    if int((dA == 0).sum()) == n:
        die("전 seed ΔAUC=0 — 캐시 오염 의심, finalize 차단")
    tmp = out_json + ".tmp"
    json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, out_json)
    # 원자료↔집계 일치 재검증
    chk = np.loadtxt(per_seed_csv, delimiter=",", skiprows=1, usecols=3)
    assert abs(float(chk.mean()) - mA) < 1e-9, "CSV↔JSON 집계 불일치"
    print(f"ΔAUC {mA:+.4f} CI[{loA:+.4f},{hiA:+.4f}] p={p_raw:.3e} | 비용Δ {mC:+.2f}% "
          f"| AUC {out['판정']['AUC']} / 비용 {out['판정']['비용']}")
    print(f"→ {out_json}")


WF_NSEED = 20            # §4-bis 승계 — 고정, 변경 금지
WF_INNER_FOLD = 5        # fwd_cost.py 원문(학습 블록 내부 5-fold)의 fold 수 승계


def wf_steps(y):
    """fwd_cost.py:42 원문 승계: 행 인덱스(생산 순서) quartile → 3 forward step.
    손상 Time 열 미사용. 정렬 금지. 신규 분할 정의 금지(사전등록 §13)."""
    import numpy as np
    N = len(y)
    q = np.floor(np.arange(N) / N * 4).astype(int); q[q == 4] = 3
    return [(np.where(q <= k)[0], np.where(q == k + 1)[0], f"Q1~{k+1}→Q{k+2}") for k in range(3)]


def wf_inner_cm(arm, Xtr, ytr, seed):
    """학습 블록 내부 inner 평가 — fwd_cost 원문의 5-fold 분할 수 + §4-bis 규칙
    (q 격자·fold별 혼동행렬 합, fit 간 점수 결합 금지 = model_cmp 수정 (8) 동일 원칙)."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    cmsum = np.zeros((NTH, 4), dtype=np.int64)
    for a, b in StratifiedKFold(WF_INNER_FOLD, shuffle=True, random_state=seed).split(Xtr, ytr):
        v = fit_predict(arm, Xtr[a], ytr[a], Xtr[b], seed)
        cm, _ = cms_rank(v, ytr[b])
        cmsum += cm
    return cmsum


def mode_walk_forward():
    """§13 집행. 게이트 전부 통과 전에는 **모델 fit이 한 건도 시작되지 않는다**."""
    import numpy as np
    from scipy.stats import t as tdist
    src_h, cfg_h, th = guard_common()
    # ── 게이트 (모델 fit 이전 — 사전등록 §13 / 보완 지시 조건 9) ──
    allow = -float(th["walk_forward"]["seed_잡음"])          # 정본에서 직독
    if abs(allow + 0.015) > 1e-12:
        die(f"thresholds.json walk_forward.seed_잡음={-allow} ≠ 사전등록 고정값 0.015 — 정본 훼손 의심, 중단")
    fin = os.path.join(OUT, "rf_lgbm_followup.json")
    if not os.path.exists(fin):
        die("finalize 결과 없음 — walk-forward 진입 불가(사전등록 §13)")
    d = json.load(open(fin, encoding="utf-8"))
    if not (d.get("판정", {}).get("AUC") == "PASS" and d.get("판정", {}).get("비용") == "PASS"):
        die(f"walk-forward 실행 조건 미충족(AUC={d.get('판정',{}).get('AUC')}, "
            f"비용={d.get('판정',{}).get('비용')}) — 사전등록 §13에 따라 차단. 미실행 사유 기록됨.")
    for k, v in src_h.items():
        if d.get(k) != v:
            die(f"finalize 이후 {k} 변경 — wf 진입 금지")
    out_path = os.path.join(OUT, "wf_result.json")
    no_overwrite(out_path)
    # ── 게이트 통과 — 이 아래에서만 모델 fit 발생 ──
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, SRC)
    import cost_model as CM
    X, y = load_data()
    NM = ["전수통과", "전수검사", "선별"]
    res_steps = []; fail_steps = []
    for tr, te, name in wf_steps(y):
        npos_te = int(y[te].sum()); nte = len(te)
        per = {}
        for arm in ARMS:                                     # base_lgbm은 항상 함께 실행(§4-bis)
            aucs, saves, raws = [], [], []
            for s in range(WF_NSEED):
                bad, msg = mem_guard()
                if bad: die(f"메모리 한계: {msg} (wf {name} [{arm}] seed {s})")
                in_cm = wf_inner_cm(arm, X[tr], y[tr], s)    # 임계 선택 = 학습 블록 내부만(전방 누수 0)
                te_s = fit_predict(arm, X[tr], y[tr], X[te], s)
                te_cm, _ = cms_rank(te_s, y[te])
                folds = [(in_cm, te_cm)]
                r = CM.evaluate(folds, nte, npos_te, nte - npos_te,
                                cl=CL_POLICY, cs=CS, beta=B, alpha=A)     # fwd_cost.run 동일 커널 호출
                raw = CM.CM_raw(folds, nte, npos_te, nte - npos_te,
                                cl=CL_POLICY, cs=CS, beta=B, alpha=A)     # 배포권고 = 원시 argmin(28차 (a))
                aucs.append(roc_auc_score(y[te], te_s)); saves.append(r["sel_vs_inspect"]); raws.append(int(raw))
                print(f"  wf {name} [{arm}] seed {s+1}/{WF_NSEED}", flush=True)
            per[arm] = dict(auc=np.array(aucs), save=np.array(saves), raws=raws)
        dA = per["rf_no_pca"]["auc"] - per["base_lgbm"]["auc"]           # 대응 Δ (동일 seed·동일 블록)
        h = tdist.ppf(0.975, WF_NSEED - 1) * dA.std(ddof=1) / np.sqrt(WF_NSEED)
        lo, hi = float(dA.mean() - h), float(dA.mean() + h)
        worse = bool(hi < allow)                             # §4-bis: CI 상한 < −0.015 → 악화
        if worse: fail_steps.append(name)

        def arm_block(a):
            return {"AUC_mean": round(float(per[a]["auc"].mean()), 4),
                    "AUC_sd": round(float(per[a]["auc"].std(ddof=1)), 4),
                    "AUC_seed별": [round(float(v), 4) for v in per[a]["auc"]],
                    "cl15_절감률_mean_%": round(float(per[a]["save"].mean()), 1),
                    "커널권고_raw": {NM[i]: f"{per[a]['raws'].count(i)}/{WF_NSEED}"
                                  for i in range(3) if per[a]["raws"].count(i)}}
        res_steps.append({"step": name, "test_n": nte, "test_npos": npos_te,
                          "유병률_%": round(npos_te / nte * 100, 1),
                          "base_lgbm": arm_block("base_lgbm"), "rf_no_pca": arm_block("rf_no_pca"),
                          "대응_ΔAUC(RF−LGBM)": {"평균": round(float(dA.mean()), 4),
                                              "95%CI": [round(lo, 4), round(hi, 4)],
                                              "seed별": [round(float(v), 4) for v in dA]},
                          "악화": worse})
    out = {"experiment_id": EXPERIMENT_ID,
           "프로토콜": ("행 인덱스 quartile 전방 3스텝(fwd_cost.py:42 원문). 임계 선택=학습 블록 내부 "
                    f"{WF_INNER_FOLD}-fold q격자 혼동행렬 합(§4-bis). {WF_NSEED} seed. "
                    f"cl={CL_POLICY}, cs={CS}, β={B}, α={A}. 커널=CM.evaluate/CM_raw(fwd_cost 동일)."),
           "seed_의미": ("외부(시간) 분할 고정. seed는 내부 임계값 선택 fold + 모델 난수만 변동 "
                      "→ ±·CI는 분산 과소추정(시간분할·데이터셋 불확실성 미포함). 인용 시 병기."),
           "허용기준": allow,
           "steps": res_steps,
           "판정": {"악화_스텝": fail_steps,
                  "전방": "FAIL" if fail_steps else "PASS",
                  "규칙": "어느 한 스텝이라도 대응 Δ 95%CI 상한 < −0.015면 악화(§4-bis 원문). 신규 기준 없음."},
           **src_h}
    tmp = out_path + ".tmp"
    json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    print(f"전방 판정: {out['판정']['전방']} (악화 스텝: {fail_steps or '없음'}) → {out_path}")


def mode_verify_cache():
    src_h, cfg_h, _ = guard_common()
    X, y = load_data()
    ok_n, bad = 0, []
    for s in range(NSEED):
        fih = fold_index_hash(y, s)
        for a in ARMS:
            npz, _ = cache_paths(a, s)
            if not os.path.exists(npz): continue
            ok, why = cache_valid(a, s, src_h, cfg_h, fih)
            if ok: ok_n += 1
            else: bad.append(f"{a}_s{s}: {why}")
    print(f"유효 캐시 {ok_n}개 / 무효 {len(bad)}개")
    for b in bad: print("  🔴", b)


# ══════════ CLI — import만으로는 아무것도 실행되지 않는다 ══════════
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="rf_lgbm_followup — 사전등록_RF_vs_LGBM_후속.md 집행")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--finalize", action="store_true")
    g.add_argument("--walk-forward", action="store_true", dest="wf")
    g.add_argument("--verify-cache", action="store_true", dest="vc")
    ap.add_argument("--cut", type=str, default=None, help="본 실행 컷 (ISO, 예: 2026-07-21T22:00) — --run 필수 인자")
    args = ap.parse_args()
    if args.prepare: mode_prepare()
    elif args.probe: mode_probe()
    elif args.run: mode_run(args.cut)
    elif args.finalize: mode_finalize()
    elif args.wf: mode_walk_forward()
    elif args.vc: mode_verify_cache()
