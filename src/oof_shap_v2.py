# -*- coding: utf-8 -*-
"""
oof_shap_v2.py — LightGBM OOF SHAP 안정성·센서군 분석 v2. **Revision 1.**
`사전등록_OOF_SHAP_v2.md`(Revision 1, 해시 코드 내 고정)를 그대로 집행한다. 결과 확인 후 규칙 변경 금지.

⚠️ import 안전: import만으로 데이터 로딩·학습·SHAP 없음(전 동작 __main__ CLI).
⚠️ 격리: 출력은 results/oof_shap_v2/ 에만. 기존 파일 무수정.

모드: --probe(seed0 → .probe_tmp → 전수검증 후 공식 승격, §R1-3) · --run(10항 게이트 후 seeds 0~19)
      · --aggregate · --groups · --manifest · --verify
"""
import os, sys, json, time, hashlib, warnings, argparse, datetime, platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "results", "oof_shap_v2")
PROBE_TMP = os.path.join(OUT, ".probe_tmp")
DATA_CSV = os.path.join(ROOT, "data", "fab_process_yield.csv")
PREREG = os.path.join(ROOT, "docs", "prereg", "사전등록_OOF_SHAP_v2.md")

# ── R1-1: 고정 해시 (직접 비교 — 불일치 시 전 모드 중단) ──
EXPECTED_DATA_SHA256 = "e8b8187ce276db148707b5103463aa0f35cb8e521da2bc697c0fd45a6b06ad0f"
EXPECTED_PREREG_SHA256 = "95bfc0ce8380c65f6be6bc7181dbff236bdfc568239b7539ce8bdc977b96360d"   # Revision 2만 허용

SEEDS = list(range(20)); NFOLD = 5; EXPECT_NFEAT = 474; N_ROWS = 1567
THRESHOLDS = [0.85, 0.90, 0.95]; TOPKS = [1, 3, 5, 10, 20]
ATOL = RTOL = 1e-4                      # §7 additivity
LOGIT_EPS = 1e-15; LOGIT_TOL = 1e-8     # R1-11
SIM_POST_TOL = 1e-9                     # R1-6
C_CLIP_TOL = 1e-8                       # R1-9
PROBE_LIMIT_MIN = 30; STORAGE_LIMIT_MB = 150
NJOBS = int(os.environ.get("OOF_NJOBS", "1"))

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, verbose=-1)

PROTECTED = ["results/xai_stability.json", "results/s59_corr_sensitivity/s59_corr_sensitivity.json",
             "results/rf_lgbm_followup/rf_lgbm_followup.json", "results/rf_lgbm_followup/wf_result.json",
             "results/model_cmp/thresholds.json", "results/report.json", "figures/fig4_shap_top20.png"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def die(msg):
    print(f"🔴 중단: {msg}", flush=True); sys.exit(1)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def env_fingerprint():
    """R1-2: 버전+OS+CPU를 정렬 JSON→SHA-256."""
    import numpy, pandas, sklearn, lightgbm, shap as shap_lib, scipy
    env = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
           "scipy": scipy.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
           "shap": shap_lib.__version__, "os": platform.platform(), "cpu_arch": platform.machine()}
    return env, hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()


def source_hashes():
    """R1-1: 고정값 직접 비교 — 불일치 시 즉시 중단(전 모드 공통 게이트)."""
    for p in (DATA_CSV, PREREG):
        if not os.path.exists(p):
            die(f"필수 파일 없음: {p}")
    dh, ph = sha256_file(DATA_CSV), sha256_file(PREREG)
    if dh != EXPECTED_DATA_SHA256:
        die(f"데이터 해시 불일치(R1-1)\n  현재: {dh}\n  기대: {EXPECTED_DATA_SHA256}")
    if ph != EXPECTED_PREREG_SHA256:
        die(f"사전등록 해시 불일치(R1-1 — Revision 1만 허용)\n  현재: {ph}\n  기대: {EXPECTED_PREREG_SHA256}")
    env, fp = env_fingerprint()
    return {"data_sha256": dh, "prereg_sha256": ph,
            "code_sha256": sha256_file(os.path.abspath(__file__)),
            "environment_fingerprint": fp, "_env": env}


def atomic_json(path, obj):
    """R1-3: tmp→fsync→재열람 검증→rename."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1); f.flush(); os.fsync(f.fileno())
    json.load(open(tmp, encoding="utf-8"))
    os.replace(tmp, path)


def load_data():
    import numpy as np, pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess_adapter import adapt
    df = pd.read_csv(DATA_CSV)
    X, y, _ = adapt(df.drop(columns=["Time"]), "Pass/Fail", 1)
    y = np.asarray(y)
    Xf = X.fillna(X.median(numeric_only=True))
    var = Xf.to_numpy(dtype=float).var(axis=0)
    master = [c for c, v in zip(Xf.columns, var) if v > 0.0]
    if len(master) != EXPECT_NFEAT or len(y) != N_ROWS:
        die(f"계보 불일치: 피처 {len(master)}(기대 {EXPECT_NFEAT}), 행 {len(y)}(기대 {N_ROWS})")
    return X[master], y, master


def normalize_shap(sv, ev):
    import numpy as np
    raw_type = type(sv).__name__
    if isinstance(sv, list):
        if len(sv) != 2: die(f"SHAP list 길이 {len(sv)} 미지원(§6)")
        out, branch, raw_shape = np.asarray(sv[1]), "list[2]→class1", f"list[2]x{np.asarray(sv[0]).shape}"
    else:
        a = np.asarray(sv); raw_shape = str(a.shape)
        if a.ndim == 2:   out, branch = a, "2D→positive-class"
        elif a.ndim == 3: out, branch = a[:, :, -1], "3D→마지막축 class1"
        else: die(f"SHAP ndim {a.ndim} 미지원(§6)"); return
    e = np.asarray(ev)
    base = float(e.ravel()[-1]) if e.size > 1 else float(e.ravel()[0])
    ev_meta = {"raw_ev_type": type(ev).__name__, "raw_ev_shape": str(e.shape), "normalized_ev": base}
    return out, base, {"raw_shap_type": raw_type, "raw_shap_shape": raw_shape,
                       "normalized_shape": str(out.shape), "class_branch": branch, **ev_meta}


# ══════════ seed 1개 실행 ══════════
def run_seed(seed, X, y, master, src_h):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    import shap as shap_lib
    from scipy.special import logit

    N = len(y)
    shap_mat = np.zeros((N, EXPECT_NFEAT), np.float32)
    prob = np.full(N, np.nan, np.float32); margin = np.full(N, np.nan, np.float32)
    base_row = np.full(N, np.nan, np.float32); fold_id = np.full(N, -1, np.int16)
    feature_active = np.zeros((NFOLD, EXPECT_NFEAT), bool)
    seen = np.zeros(N, int); fold_meta = []
    t0 = time.time(); t_start = now()
    Xv = X.to_numpy(dtype=float)

    for k, (tr, va) in enumerate(StratifiedKFold(NFOLD, shuffle=True, random_state=seed).split(Xv, y)):
        if len(set(y[tr].tolist())) < 2 or len(set(y[va].tolist())) < 2:
            die(f"seed {seed} fold {k}: 단일 클래스(§17)")
        seen[va] += 1
        imp = SimpleImputer(strategy="median").fit(Xv[tr])
        A_tr, A_va = imp.transform(Xv[tr]), imp.transform(Xv[va])
        vt = VarianceThreshold(0.0).fit(A_tr); mask = vt.get_support()
        sc = StandardScaler().fit(A_tr[:, mask])
        Ztr, Zva = sc.transform(A_tr[:, mask]), sc.transform(A_va[:, mask])
        active_idx = np.where(mask)[0]; feature_active[k, active_idx] = True
        spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))
        m = LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw, random_state=seed, n_jobs=NJOBS).fit(Ztr, y[tr])
        expl = shap_lib.TreeExplainer(m)
        sv, base, shap_qc = normalize_shap(expl.shap_values(Zva, check_additivity=False), expl.expected_value)
        if sv.shape != (len(va), int(mask.sum())):
            die(f"seed {seed} fold {k}: SHAP shape {sv.shape} 정렬 불일치(§17)")
        raw_margin = np.asarray(m.predict(Zva, raw_score=True), float)
        proba = m.predict_proba(Zva)
        if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(raw_margin))):
            die(f"seed {seed} fold {k}: NaN/Inf(§17)")
        # §7 additivity (외부 검증)
        recon = base + sv.sum(1)
        add_err = float(np.abs(recon - raw_margin).max())
        if not np.allclose(recon, raw_margin, atol=ATOL, rtol=RTOL):
            die(f"seed {seed} fold {k}: additivity 실패 max_err={add_err:.2e}(§7) — 전체 중단")
        # R1-11 logit QC (성능 아님 — raw margin 의미 확인)
        lg = logit(np.clip(proba[:, 1], LOGIT_EPS, 1 - LOGIT_EPS))
        logit_err = float(np.abs(lg - raw_margin).max())
        if not np.allclose(lg, raw_margin, atol=LOGIT_TOL, rtol=LOGIT_TOL):
            die(f"seed {seed} fold {k}: logit QC 실패 max_err={logit_err:.2e}(R1-11)")
        shap_mat[np.ix_(va, active_idx)] = sv.astype(np.float32)
        prob[va] = proba[:, 1].astype(np.float32); margin[va] = raw_margin.astype(np.float32)
        base_row[va] = np.float32(base); fold_id[va] = k
        fold_meta.append({"fold": k, "n_train": len(tr), "n_val": len(va),
                          "train_idx_sha256": sha256_obj(tr.tolist()), "val_idx_sha256": sha256_obj(va.tolist()),
                          "n_active": int(mask.sum()),
                          "removed_features": [master[i] for i in np.where(~mask)[0]],
                          "scale_pos_weight": round(spw, 6),
                          "model_params_actual": m.get_params(deep=True),        # R1-10 실제 fit 객체
                          "shap_qc": shap_qc,                                    # R1-11 원형·분기
                          "raw_margin_shape": str(raw_margin.shape), "predict_proba_shape": str(proba.shape),
                          "additivity_max_err": add_err, "logit_qc_max_err": logit_err})
        print(f"  seed {seed} fold {k+1}/{NFOLD}: active {int(mask.sum())}/474, "
              f"add {add_err:.1e}, logit {logit_err:.1e} [{time.time()-t0:.0f}s]", flush=True)
    if not np.all(seen == 1): die(f"seed {seed}: coverage 위반(§17)")
    arrays = dict(shap_values=shap_mat, oof_probability=prob, oof_raw_margin=margin,
                  base_value_per_row=base_row, fold_id=fold_id,
                  original_row_index=np.arange(N, dtype=np.int32), y_true=y.astype(np.int8),
                  feature_active=feature_active)
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier as _L
    meta = {"seed": seed, "experiment_id": "oof_shap_v2_rev2", "started": t_start, "finished": now(),
            "elapsed_sec": round(time.time() - t0, 1), "folds": fold_meta,
            "master_n_features": EXPECT_NFEAT, "master_features_sha256": sha256_obj(master),
            "template_params": _L(**LGBM_PARAMS).get_params(),                    # R1-10 템플릿은 별도 명명
            "prep": "fold 내부: SimpleImputer(median)→VarianceThreshold(0.0)→StandardScaler (train fit only)",
            "shap_basis": "raw margin (log-odds)", "additivity_rule": f"allclose atol=rtol={ATOL}",
            "logit_qc_rule": f"eps={LOGIT_EPS}, atol=rtol={LOGIT_TOL}",
            "oof_auc_QC전용": round(float(roc_auc_score(y, prob)), 4),
            "QC주의": "OOF AUC는 QC 전용(§11) — 성능 성과·모델 선택 사용 금지",
            "validation_coverage_pass": True, "shape_checks_pass": True,        # R2-1(위 assert 통과 시에만 도달)
            "env": src_h["_env"],
            **{k: v for k, v in src_h.items() if not k.startswith("_")}}
    return arrays, meta


# ══════════ R1-12 전수 검증 ══════════
def validate_arrays(arrays, y, seed, where):
    import numpy as np
    a = arrays
    req = {"shap_values": (N_ROWS, EXPECT_NFEAT), "oof_probability": (N_ROWS,), "oof_raw_margin": (N_ROWS,),
           "base_value_per_row": (N_ROWS,), "fold_id": (N_ROWS,), "original_row_index": (N_ROWS,),
           "y_true": (N_ROWS,), "feature_active": (NFOLD, EXPECT_NFEAT)}
    for k, shp in req.items():
        if k not in a: die(f"{where}: key '{k}' 부재(R1-12)")
        if tuple(a[k].shape) != shp: die(f"{where}: {k} shape {a[k].shape} ≠ {shp}(R1-12)")
    # R2-2: dtype 8종 고정 검증
    for k in ("shap_values", "oof_probability", "oof_raw_margin", "base_value_per_row"):
        if a[k].dtype != np.float32: die(f"{where}: {k} dtype {a[k].dtype} ≠ float32(R2-2)")
    for k in ("fold_id", "original_row_index", "y_true"):
        if not np.issubdtype(a[k].dtype, np.integer): die(f"{where}: {k} dtype {a[k].dtype} 정수형 아님(R2-2)")
    if a["feature_active"].dtype != np.bool_: die(f"{where}: feature_active dtype {a['feature_active'].dtype} ≠ bool(R2-2)")
    for k in ("shap_values", "oof_probability", "oof_raw_margin", "base_value_per_row"):
        if not np.all(np.isfinite(a[k])): die(f"{where}: {k} 비유한(R1-12)")
    if sorted(a["original_row_index"].tolist()) != list(range(N_ROWS)): die(f"{where}: row_index 순열 아님(R1-12)")
    if not (a["fold_id"].min() >= 0 and a["fold_id"].max() <= NFOLD - 1): die(f"{where}: fold_id 범위(R1-12)")
    if not np.array_equal(a["y_true"], y.astype(np.int8)): die(f"{where}: y_true 불일치(R1-12)")
    if not (a["oof_probability"].min() >= 0 and a["oof_probability"].max() <= 1): die(f"{where}: proba 범위(R1-12)")
    for k in range(NFOLD):                       # inactive 열이 해당 fold val 행에서 0
        rows = np.where(a["fold_id"] == k)[0]; inact = np.where(~a["feature_active"][k])[0]
        if len(inact) and not np.all(a["shap_values"][np.ix_(rows, inact)] == 0):
            die(f"{where}: fold {k} inactive 열 SHAP≠0(R1-12)")


def paths(seed, base=None):
    b = base or OUT
    return (os.path.join(b, f"oof_shap_seed{seed}.npz"), os.path.join(b, f"oof_shap_seed{seed}.meta.json"))


def save_npz_validated(path, arrays, y, seed):
    import numpy as np
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    with np.load(tmp) as z:                       # 재열람 후 동일 검증(R1-12)
        validate_arrays({k: z[k] for k in z.files}, y, seed, f"reopen({os.path.basename(path)})")
    os.replace(tmp, path)


def commit_path(seed):
    return os.path.join(OUT, f"seed_{seed:02d}.commit.json")


def load_json_or_die(p, what):
    """R2-5: malformed JSON은 명확한 메시지로 중단(게이트 경로 전용)."""
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        die(f"{what} JSON 손상/파싱 실패: {p} ({e})")


def validate_seed_metadata(seed, src_h, base=None):
    """R2-1: metadata 의미 검증 + NPZ 교차 검증. 반환 (ok, 사유). 재파싱만으로 통과 처리하지 않는다."""
    import numpy as np
    npz_p, meta_p = paths(seed, base)
    if not (os.path.exists(npz_p) and os.path.exists(meta_p)):
        return False, "NPZ/metadata 파일 부재(부분 승격 금지)"
    try:
        m = json.load(open(meta_p, encoding="utf-8"))
    except Exception as e:
        return False, f"metadata JSON 손상({e})"
    req = ["seed", "folds", "master_n_features", "npz_sha256", "data_sha256", "prereg_sha256",
           "code_sha256", "environment_fingerprint", "additivity_rule", "logit_qc_rule",
           "validation_coverage_pass", "shape_checks_pass"]
    miss = [k for k in req if k not in m]
    if miss: return False, f"metadata key 누락 {miss}"
    if m["seed"] != seed: return False, f"metadata seed {m['seed']} ≠ 기대 {seed}(파일명)"
    if len(m["folds"]) != NFOLD: return False, f"fold metadata {len(m['folds'])}개 ≠ {NFOLD}"
    for k in ("data_sha256", "prereg_sha256", "code_sha256", "environment_fingerprint"):
        if m.get(k) != src_h[k]: return False, f"{k} 현재 정본 불일치"
    if m["npz_sha256"] != sha256_file(npz_p): return False, "NPZ 해시 ≠ metadata 기록"
    if not (m["validation_coverage_pass"] and m["shape_checks_pass"]): return False, "QC 플래그 실패"
    if m["master_n_features"] != EXPECT_NFEAT: return False, "메타 피처 수 ≠ 474"
    for f in m["folds"]:
        for k in ("model_params_actual", "train_idx_sha256", "val_idx_sha256",
                  "additivity_max_err", "logit_qc_max_err", "n_val"):
            if k not in f: return False, f"fold {f.get('fold')} '{k}' 누락"
    with np.load(npz_p) as z:
        a = {k: z[k] for k in z.files}
    if a["shap_values"].shape != (N_ROWS, EXPECT_NFEAT): return False, "NPZ shape ≠ (1567,474)"
    for f in m["folds"]:                              # fold 구조 교차 검증
        cnt = int((a["fold_id"] == f["fold"]).sum())
        if cnt != f["n_val"]: return False, f"fold {f['fold']}: meta n_val {f['n_val']} ≠ NPZ {cnt}"
    # R2-1 허용범위 = allclose 의미론적 상한(임의 숫자 아님)
    mm = float(np.abs(a["oof_raw_margin"]).max())
    add_bound = ATOL + RTOL * mm; lg_bound = LOGIT_TOL * (1 + mm)
    errs = [f["additivity_max_err"] for f in m["folds"]]
    lgs = [f["logit_qc_max_err"] for f in m["folds"]]
    if not all(np.isfinite(errs)) or max(errs) > add_bound:
        return False, f"additivity 오차 이탈(max {max(errs):.2e} > {add_bound:.2e})"
    if not all(np.isfinite(lgs)) or max(lgs) > lg_bound:
        return False, f"logit 오차 이탈(max {max(lgs):.2e} > {lg_bound:.2e})"
    return True, "ok"


def write_commit(seed, src_h):
    """R2-4: 승격 마지막 단계 — commit marker 원자 저장."""
    npz_p, meta_p = paths(seed)
    atomic_json(commit_path(seed), {
        "seed": seed, "verdict": "PASS",
        "npz_sha256": sha256_file(npz_p), "meta_sha256": sha256_file(meta_p),
        "data_sha256": src_h["data_sha256"], "prereg_sha256": src_h["prereg_sha256"],
        "code_sha256": src_h["code_sha256"], "environment_fingerprint": src_h["environment_fingerprint"],
        "created_at": now()})


def cache_valid(seed, src_h):
    """R2-4: commit marker + 4해시 + 파일 해시 + metadata 의미 검증 전부 통과 시에만 공식 인정."""
    cp = commit_path(seed)
    if not os.path.exists(cp): return False
    try:
        c = json.load(open(cp, encoding="utf-8"))
    except Exception:
        return False
    npz_p, meta_p = paths(seed)
    if not (os.path.exists(npz_p) and os.path.exists(meta_p)): return False
    if c.get("verdict") != "PASS" or c.get("seed") != seed: return False
    for k in ("data_sha256", "prereg_sha256", "code_sha256", "environment_fingerprint"):
        if c.get(k) != src_h[k]: return False
    if c.get("npz_sha256") != sha256_file(npz_p) or c.get("meta_sha256") != sha256_file(meta_p): return False
    ok, _ = validate_seed_metadata(seed, src_h)
    return ok


# ══════════ 모드 ══════════
def mode_probe():
    src_h = source_hashes()
    os.makedirs(PROBE_TMP, exist_ok=True)
    X, y, master = load_data()
    t0 = time.time()
    arrays, meta = run_seed(0, X, y, master, src_h)
    elapsed = time.time() - t0
    validate_arrays(arrays, y, 0, "probe(메모리)")
    tnpz, tmeta = paths(0, PROBE_TMP)
    save_npz_validated(tnpz, arrays, y, 0)
    size = os.path.getsize(tnpz); proj = size * len(SEEDS)
    fail = []
    if elapsed > PROBE_LIMIT_MIN * 60: fail.append(f"시간 {elapsed/60:.1f}분 > {PROBE_LIMIT_MIN}분")
    if proj > STORAGE_LIMIT_MB * 1e6: fail.append(f"예상 용량 {proj/1e6:.0f}MB > {STORAGE_LIMIT_MB}MB")
    verdict = "FAIL" if fail else "PASS"
    meta_ok = False; commit_h = None
    if verdict == "PASS":                          # R1-4/R2-4: 전수검증→이동→metadata 검증→commit marker
        meta["npz_sha256"] = sha256_file(tnpz)
        atomic_json(tmeta, meta)
        os.replace(tnpz, paths(0)[0]); os.replace(tmeta, paths(0)[1])
        ok, why = validate_seed_metadata(0, src_h)
        if not ok:
            die(f"승격 직후 metadata 검증 실패: {why}(R2-1) — commit marker 미생성")
        meta_ok = True
        write_commit(0, src_h)
        commit_h = sha256_file(commit_path(0))
        npz_h, meta_h = sha256_file(paths(0)[0]), sha256_file(paths(0)[1])
    else:
        npz_h = meta_h = None                      # 공식 seed 파일·marker 미생성
    summary = {"verdict": verdict, "seed": 0,
               "data_hash": src_h["data_sha256"], "prereg_hash": src_h["prereg_sha256"],
               "code_hash": src_h["code_sha256"], "environment_fingerprint": src_h["environment_fingerprint"],
               "seed0_npz_hash": npz_h, "seed0_metadata_hash": meta_h, "seed0_commit_hash": commit_h,
               "seed0_elapsed_seconds": round(elapsed, 1), "seed0_file_size_bytes": int(size),
               "projected_total_size_bytes": int(proj),
               "storage_note": "R2-7: 150MB 추정은 seed NPZ 중심 — metadata·aggregate·groups 파일 별도(총량은 manifest 실측)",
               "additivity_max_error": max(f["additivity_max_err"] for f in meta["folds"]),
               "logit_qc_pass": True, "logit_max_error": max(f["logit_qc_max_err"] for f in meta["folds"]),
               "metadata_validation_pass": meta_ok,
               "validation_coverage_pass": True, "shape_checks_pass": True,
               "fail_reasons": fail, "created_at": now()}
    atomic_json(os.path.join(OUT, "probe_summary.json"), summary)
    if fail:
        die(f"probe FAIL: {fail} — 공식 seed 파일 미생성, summary 기록됨")
    print(f"[프로브] PASS — seed0 {elapsed/60:.1f}분 · {size/1e6:.1f}MB → 20 seed 예상 "
          f"{elapsed/60*20:.0f}분 · {proj/1e6:.0f}MB. 규칙 무변경 시 공식 승계(§R1-3)")


def _run_gate(src_h):
    """R1-3 10항 + R2-5 확장 게이트."""
    p = os.path.join(OUT, "probe_summary.json")
    if not os.path.exists(p): die("probe_summary 없음 — --probe 먼저(R1-3)")
    s = load_json_or_die(p, "probe_summary")
    npz0, mj0 = paths(0)
    checks = [
        ("1 verdict PASS", s.get("verdict") == "PASS"),
        ("2 data hash", s.get("data_hash") == src_h["data_sha256"]),
        ("3 prereg hash", s.get("prereg_hash") == src_h["prereg_sha256"]),
        ("4 code hash", s.get("code_hash") == src_h["code_sha256"]),
        ("5 env fingerprint", s.get("environment_fingerprint") == src_h["environment_fingerprint"]),
        ("6 seed0 파일 해시", os.path.exists(npz0) and os.path.exists(mj0)
         and sha256_file(npz0) == s.get("seed0_npz_hash") and sha256_file(mj0) == s.get("seed0_metadata_hash")),
        ("7 metadata↔summary", os.path.exists(mj0)
         and json.load(open(mj0, encoding="utf-8")).get("npz_sha256") == s.get("seed0_npz_hash")),
        ("8 QC 통과", s.get("validation_coverage_pass") and s.get("shape_checks_pass")
         and s.get("additivity_max_error") is not None),   # additivity 통과는 실행 중 §7 allclose가 강제(실패 시 verdict 자체가 안 나옴)
        ("9 용량", s.get("projected_total_size_bytes", 1e18) <= STORAGE_LIMIT_MB * 1e6),
        ("10 시간", s.get("seed0_elapsed_seconds", 1e9) <= PROBE_LIMIT_MIN * 60)]
    # R2-5 확장: metadata 의미검증 · cache_valid · logit QC · commit marker 정합
    ok_meta, why = validate_seed_metadata(0, src_h)
    checks += [
        ("11 metadata 의미검증(seed==0·4해시·QC·범위)", ok_meta),
        ("12 cache_valid(0)", cache_valid(0, src_h)),
        ("13 logit_qc_pass", bool(s.get("logit_qc_pass")) and s.get("logit_max_error") is not None),
        ("14 metadata_validation_pass", bool(s.get("metadata_validation_pass"))),
        ("15 commit marker 정합", os.path.exists(commit_path(0))
         and s.get("seed0_commit_hash") == sha256_file(commit_path(0)))]
    bad = [n for n, ok in checks if not ok]
    if bad: die(f"--run 게이트 실패 {bad}" + (f" | metadata 사유: {why}" if not ok_meta else "") + " — 새 --probe 필요")


def mode_run():
    src_h = source_hashes(); os.makedirs(OUT, exist_ok=True)
    _run_gate(src_h)
    run_started = now()
    X, y, master = load_data()
    for s in SEEDS:
        if cache_valid(s, src_h):
            print(f"  seed {s}: 유효 캐시 승계(marker 정합)"); continue
        arrays, meta = run_seed(s, X, y, master, src_h)
        validate_arrays(arrays, y, s, f"seed{s}")
        npz, mj = paths(s)
        save_npz_validated(npz, arrays, y, s)
        meta["npz_sha256"] = sha256_file(npz)
        atomic_json(mj, meta)
        ok, why = validate_seed_metadata(s, src_h)          # R2-1: 승격 검증 후에만 marker
        if not ok: die(f"seed {s} metadata 검증 실패: {why}")
        write_commit(s, src_h)
        print(f"  → seed {s} 저장+commit {os.path.getsize(npz)/1e6:.1f}MB")
    atomic_json(os.path.join(OUT, "run_times.json"),         # R2-6: manifest용 시작/종료 분리 기록
                {"run_started_at": run_started, "run_completed_at": now()})
    print("→ 전 seed 완료")


def _load_all(src_h):
    import numpy as np
    mats = []
    for s in SEEDS:
        if not cache_valid(s, src_h): die(f"seed {s} 캐시 무효/부재")
        with np.load(paths(s)[0]) as z:
            mats.append({k: z[k] for k in z.files})
    return mats


def _rank(imp):
    import numpy as np
    order = np.argsort(-imp, kind="stable")       # §10 동점=마스터 순서
    r = np.empty(len(imp), int); r[order] = np.arange(1, len(imp) + 1)
    return r, order


def mode_aggregate():
    import numpy as np
    src_h = source_hashes()
    X, y, master = load_data()
    mats = _load_all(src_h)
    METRICS = ["mean_abs_all", "mean_abs_pos", "mean_abs_neg", "signed_all", "signed_pos", "signed_neg"]
    vec = {m: [] for m in METRICS}
    for d in mats:
        sm = d["shap_values"].astype(np.float64); yt = d["y_true"]
        v = {"mean_abs_all": np.abs(sm).mean(0), "mean_abs_pos": np.abs(sm[yt == 1]).mean(0),
             "mean_abs_neg": np.abs(sm[yt == 0]).mean(0), "signed_all": sm.mean(0),
             "signed_pos": sm[yt == 1].mean(0), "signed_neg": sm[yt == 0].mean(0)}
        for m in METRICS:
            vec[m].append(v[m])
    # R2-3: 정렬 필드 — mean_abs*는 값 내림차순 / signed*는 signed_desc(값 내림차순)·absmag(|값| 내림차순) 이중 분리
    order_fields = {}
    for m in METRICS:
        if m.startswith("mean_abs"):
            order_fields[m] = [np.argsort(-v, kind="stable") for v in vec[m]]
        else:
            order_fields[f"{m}_signed_desc"] = [np.argsort(-v, kind="stable") for v in vec[m]]
            order_fields[f"{m}_absmag"] = [np.argsort(-np.abs(v), kind="stable") for v in vec[m]]
    arrs = {}
    for m in METRICS:
        V = np.stack(vec[m])
        arrs[f"{m}_per_seed"] = V.astype(np.float32)
        arrs[f"{m}_mean"] = V.mean(0).astype(np.float32); arrs[f"{m}_sd"] = V.std(0, ddof=1).astype(np.float32)
        arrs[f"{m}_median"] = np.median(V, 0).astype(np.float32)
        arrs[f"{m}_q25"] = np.percentile(V, 25, 0).astype(np.float32)
        arrs[f"{m}_q75"] = np.percentile(V, 75, 0).astype(np.float32)
        arrs[f"{m}_min"] = V.min(0).astype(np.float32); arrs[f"{m}_max"] = V.max(0).astype(np.float32)
    for fld, orders in order_fields.items():             # R2-3: 필드별 rank 배열 실저장
        R = []
        for o in orders:
            r = np.empty(EXPECT_NFEAT, int); r[o] = np.arange(1, EXPECT_NFEAT + 1); R.append(r)
        arrs[f"{fld}_rank_per_seed"] = np.stack(R).astype(np.int16)
    seed_topk = {fld: {str(k): [[master[i] for i in o[:k]] for o in orders] for k in TOPKS}
                 for fld, orders in order_fields.items()}   # R2-3: 6종(signed는 이중) × Top-1/3/5/10/20 × 20 seed
    ap = os.path.join(OUT, "aggregate_arrays.npz")
    tmp = ap + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, feature_names=np.array(master), **arrs); f.flush(); os.fsync(f.fileno())
    with np.load(tmp) as z: assert "mean_abs_all_per_seed" in z.files
    os.replace(tmp, ap)

    R_main = arrs["mean_abs_all_rank_per_seed"].astype(int)
    headline = np.stack(vec["mean_abs_all"]).mean(0)
    h_rank, h_order = _rank(headline)
    topsets = {k: [set(np.argsort(-v, kind="stable")[:k].tolist()) for v in vec["mean_abs_all"]] for k in TOPKS}
    jac = {k: [round(len(a & b) / len(a | b), 4) for i, a in enumerate(topsets[k]) for b in topsets[k][i + 1:]]
           for k in TOPKS}
    # R1-7 fold 참고 분석 (seed 내부 10쌍 × 20 = 200쌍)
    fold_records = []; fold_pos_top10 = []
    for si, d in enumerate(mats):
        sm = d["shap_values"].astype(np.float64); fid = d["fold_id"]; yt = d["y_true"]
        fv = []
        for k in range(NFOLD):
            rows = np.where(fid == k)[0]
            fv.append(np.abs(sm[rows]).mean(0))
            prow = rows[yt[rows] == 1]
            fold_pos_top10.append({"seed": si, "fold": k, "exploratory_small_sample": True,
                                   "n_pos": int(len(prow)),
                                   "top10": [master[i] for i in np.argsort(-np.abs(sm[prow]).mean(0), kind="stable")[:10]]})
        for a in range(NFOLD):
            for b in range(a + 1, NFOLD):
                ordA_full = np.argsort(-fv[a], kind="stable")   # R2-7: 결정적 정렬(set 순회 금지)
                ordB_full = np.argsort(-fv[b], kind="stable")
                for k in TOPKS:
                    ordA, ordB = ordA_full[:k], ordB_full[:k]
                    A, B = set(ordA.tolist()), set(ordB.tolist())
                    fold_records.append({"seed": si, "fold_a": a, "fold_b": b, "k": k,
                                         "intersection": len(A & B), "union": len(A | B),
                                         "jaccard": round(len(A & B) / len(A | B), 4),
                                         "fold_a_topk": [master[i] for i in ordA],
                                         "fold_b_topk": [master[i] for i in ordB]})
    sensors = {master[j]: {"median_rank": float(np.median(R_main[:, j])),
                           "iqr": [float(np.percentile(R_main[:, j], 25)), float(np.percentile(R_main[:, j], 75))],
                           "min": int(R_main[:, j].min()), "max": int(R_main[:, j].max()),
                           **{f"top{k}_rate": round(float(np.mean(R_main[:, j] <= k)), 3) for k in TOPKS}}
               for j in range(EXPECT_NFEAT)}
    out = {"n_seed": len(SEEDS), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "dtype": "float32(npz)", "저장": "seed별 6종 벡터·순위·20seed 통계 = aggregate_arrays.npz(R1-8)",
           "집계원칙": "seed별 mean|SHAP|→순위→분포. 행별 seed 평균 금지(§9). 합격선 없음(§10).",
           "클래스별_주의": "전체 평균만으로 불량 우선순위 결론 금지(§9)",
           "headline_top30": [{"피처": master[i], "mean_abs": round(float(headline[i]), 6),
                             "median_rank": sensors[master[i]]["median_rank"]} for i in h_order[:30]],
           "sensor_stats": sensors,
           "seed_topk": seed_topk,                 # R2-3: 6종(signed 이중 정렬 분리) × Top-1/3/5/10/20 × 20 seed
           "seed_topk_의미": {"mean_abs_*": "값 내림차순", "signed_*_signed_desc": "signed 값 자체 내림차순(불량 방향)",
                          "signed_*_absmag": "|signed mean| 내림차순(순방향 크기)"},
           "jaccard_seed_분포": jac,
           "fold_참고분석": {"주의": "소표본 참고 — 주 결론에 사용 금지(R1-7)", "n_pairs": len(fold_records) // len(TOPKS),
                        "records": fold_records},
           "fold_양성전용_top10": fold_pos_top10,
           "oof_auc_QC": [json.load(open(paths(s)[1], encoding="utf-8"))["oof_auc_QC전용"] for s in SEEDS],
           "QC주의": "OOF AUC는 QC 전용(§11)",
           "aggregate_arrays_sha256": sha256_file(ap),
           "seed_npz_sha256": {str(s): sha256_file(paths(s)[0]) for s in SEEDS},     # R2-6 계보 내장
           "seed_meta_sha256": {str(s): sha256_file(paths(s)[1]) for s in SEEDS},
           **{k: v for k, v in src_h.items() if not k.startswith("_")}}
    atomic_json(os.path.join(OUT, "aggregate.json"), out)
    print(f"→ aggregate.json + aggregate_arrays.npz | headline top5: {[master[i] for i in h_order[:5]]}")


def _cluster_info(labels, sim, master, method_id, thr, simdef):
    """R1-5: 전 군집 정보."""
    import numpy as np
    clusters = []
    for c in sorted(set(labels.tolist())):
        mem = np.where(labels == c)[0]
        sub = sim[np.ix_(mem, mem)]
        off = sub[~np.eye(len(mem), dtype=bool)] if len(mem) > 1 else None
        clusters.append({"cluster_id": int(c), "size": int(len(mem)),
                         "singleton": bool(len(mem) == 1), "multi": bool(len(mem) > 1),
                         "member_names": [master[i] for i in mem.tolist()],
                         "member_indices": [int(i) for i in mem.tolist()],
                         "min_pairwise_sim": (round(float(off.min()), 6) if off is not None else None),
                         "max_pairwise_sim": (round(float(off.max()), 6) if off is not None else None)})
    tj = master.index("59")
    sizes = [c["size"] for c in clusters]
    return {"method_id": method_id, "threshold": thr, "similarity_definition": simdef,
            "cluster_labels": [int(v) for v in labels.tolist()],
            "n_clusters": len(clusters), "n_singleton": sum(c["singleton"] for c in clusters),
            "n_multi": sum(c["multi"] for c in clusters), "max_size": max(sizes),
            "S59_cluster_id": int(labels[tj]),
            "S59_members": [master[i] for i in np.where(labels == labels[tj])[0].tolist()],
            "clusters": clusters}


def mode_groups():
    import numpy as np
    from scipy.stats import rankdata
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    src_h = source_hashes()
    X, y, master = load_data()
    M = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    Z = (M - M.mean(0)) / M.std(0); P = (Z.T @ Z) / len(M)
    Rk = np.apply_along_axis(rankdata, 0, M); ZR = (Rk - Rk.mean(0)) / Rk.std(0); S_ = (ZR.T @ ZR) / len(M)

    def prep_sim(mat):
        """R1-6: clip·대칭·대각·finite 보증 + assert."""
        s = np.clip(np.abs(mat), 0.0, 1.0); s = (s + s.T) / 2; np.fill_diagonal(s, 1.0)
        assert np.all(np.isfinite(s)), "sim NaN/Inf(R1-6)"
        d = 1.0 - s; np.fill_diagonal(d, 0.0)
        assert d.min() >= 0 and np.allclose(d, d.T), "distance 음수/비대칭(R1-6)"
        return s, d

    sim, D = prep_sim(np.maximum(np.abs(P), np.abs(S_)))
    mats = _load_all(src_h)
    shaps = [d["shap_values"].astype(np.float64) for d in mats]; ys = [d["y_true"] for d in mats]

    contrib_arrays = {}

    def contributions(cfg_key, info):
        """R1-9: 전 군집 × seed × 클래스 A/B/C."""
        labels = np.array(info["cluster_labels"]); stats = {}
        ids = [c["cluster_id"] for c in info["clusters"]]
        for cls, sel in (("all", None), ("pos", 1), ("neg", 0)):
            A = np.zeros((len(ids), len(SEEDS))); B = np.zeros_like(A)
            for si, (sm, yt) in enumerate(zip(shaps, ys)):
                rows = sm if sel is None else sm[yt == sel]
                for ci, c in enumerate(ids):
                    mem = np.where(labels == c)[0]
                    A[ci, si] = np.abs(rows[:, mem].sum(1)).mean()
                    B[ci, si] = np.abs(rows[:, mem]).mean(0).sum()
            C = np.full_like(A, np.nan)
            nz = B > 0; C[nz] = A[nz] / B[nz]
            bad = nz & ((C < -C_CLIP_TOL) | (C > 1 + C_CLIP_TOL))
            if bad.any(): die(f"{cfg_key}/{cls}: C 범위 이탈 max={np.nanmax(C):.10f}(R1-9)")
            C[nz] = np.clip(C[nz], 0.0, 1.0)      # R1-9: |이탈|≤1e-8만 clip(위에서 초과분은 이미 중단)
            contrib_arrays[f"{cfg_key}|{cls}|A"] = A.astype(np.float32)
            contrib_arrays[f"{cfg_key}|{cls}|B"] = B.astype(np.float32)
            contrib_arrays[f"{cfg_key}|{cls}|C"] = C.astype(np.float32)
            def st(V):
                return {"mean": np.nanmean(V, 1).round(6).tolist(), "sd": np.nanstd(V, 1, ddof=1).round(6).tolist(),
                        "median": np.nanmedian(V, 1).round(6).tolist(),
                        "q25": np.nanpercentile(V, 25, 1).round(6).tolist(),
                        "q75": np.nanpercentile(V, 75, 1).round(6).tolist(),
                        "min": np.nanmin(V, 1).round(6).tolist(), "max": np.nanmax(V, 1).round(6).tolist()}
            stats[cls] = {"cluster_ids": ids, "A": st(A), "B": st(B), "C": st(C),
                          "valid_C_seeds": np.sum(~np.isnan(C), 1).tolist(),
                          "ratio_of_means": np.divide(A.mean(1), B.mean(1),
                                                      out=np.full(len(ids), np.nan), where=B.mean(1) > 0).round(6).tolist()}
        # R1-4 표시 순서(계산은 전 군집): B(all) seed평균 ↓ → A ↓ → size ↓ → 마스터 최소 인덱스
        Ball = contrib_arrays[f"{cfg_key}|all|B"].mean(1); Aall = contrib_arrays[f"{cfg_key}|all|A"].mean(1)
        sz = np.array([c["size"] for c in info["clusters"]])
        first = np.array([min(c["member_indices"]) for c in info["clusters"]])
        disp = np.lexsort((first, -sz, -Aall, -Ball))[:30]
        stats["display_top30_cluster_ids"] = [ids[i] for i in disp]
        return stats

    out = {"similarity": "max(|Pearson|,|Spearman|), clip[0,1], 전체 1567행·라벨 미사용(R1-6)",
           "configs": {}, **{k: v for k, v in src_h.items() if not k.startswith("_")}}
    lk = linkage(squareform(D, checks=False), method="complete")
    for thr in THRESHOLDS:
        lab = fcluster(lk, t=1 - thr, criterion="distance")
        info = _cluster_info(np.array(lab), sim, master, "complete_linkage_combined", thr,
                             "max(|Pearson|,|Spearman|)")
        for c in info["clusters"]:                    # R1-6+R2-7: 반올림 전 similarity로 사후검증(반올림값은 출력 전용)
            if c["multi"]:
                mem = np.array(c["member_indices"])
                off = sim[np.ix_(mem, mem)][~np.eye(len(mem), dtype=bool)]
                if float(off.min()) < thr - SIM_POST_TOL:
                    die(f"complete-linkage 사후검증 실패 thr={thr} (비반올림 min={off.min():.12f})(R1-6/R2-7)")
        info["contrib"] = contributions(f"CL|{thr}", info)
        out["configs"][f"complete_linkage_combined_{thr}"] = info
        # 부록 CC (연쇄 경고)
        adj = sim >= thr
        comp = -np.ones(EXPECT_NFEAT, int); cid = 0
        for i in range(EXPECT_NFEAT):
            if comp[i] >= 0: continue
            stack = [i]; comp[i] = cid
            while stack:
                u = stack.pop()
                for v in np.where(adj[u] & (comp < 0))[0]: comp[v] = cid; stack.append(v)
            cid += 1
        info_cc = _cluster_info(comp + 1, sim, master, "connected_component_combined", thr,
                                "max(|Pearson|,|Spearman|) — 연쇄 연결 경고: 군 내 전 쌍 보장 없음")
        info_cc["contrib"] = contributions(f"CC|{thr}", info_cc)
        out["configs"][f"connected_component_combined_{thr}"] = info_cc
    for name, mat in (("pearson_only", np.abs(P)), ("spearman_only", np.abs(S_))):
        s1, d1 = prep_sim(mat)
        lab = fcluster(linkage(squareform(d1, checks=False), method="complete"), t=0.10, criterion="distance")
        info = _cluster_info(np.array(lab), s1, master, f"{name}_complete_linkage", 0.90, f"|{name}|")
        info["contrib"] = contributions(f"{name}|0.90", info)
        out["configs"][f"{name}_complete_linkage_0.90"] = info
    xs = json.load(open(os.path.join(ROOT, "results", "xai_stability.json"), encoding="utf-8"))
    out["legacy_greedy_대응"] = {"기존": "Pearson>0.95 greedy(정본 290, S59 크기 1)",
                          "본분석_CL_0.95_S59": out["configs"]["complete_linkage_combined_0.95"]["S59_members"],
                          "기존_S59_cluster_size": xs["B_cluster"]["s59_cluster_size"]}
    gp = os.path.join(OUT, "groups_contrib.npz")
    tmp = gp + ".tmp"
    import numpy as _np
    with open(tmp, "wb") as f:
        _np.savez_compressed(f, **contrib_arrays); f.flush(); os.fsync(f.fileno())
    with _np.load(tmp) as z: assert len(z.files) == len(contrib_arrays)
    os.replace(tmp, gp)
    out["groups_contrib_sha256"] = sha256_file(gp)
    out["seed_npz_sha256"] = {str(s): sha256_file(paths(s)[0]) for s in SEEDS}       # R2-6 계보 내장
    out["seed_meta_sha256"] = {str(s): sha256_file(paths(s)[1]) for s in SEEDS}
    atomic_json(os.path.join(OUT, "groups.json"), out)
    print("→ groups.json + groups_contrib.npz (전 군집 저장, 표시 30은 별도 기준)")


def mode_manifest():
    src_h = source_hashes()
    need = ["probe_summary.json", "aggregate.json", "aggregate_arrays.npz", "groups.json",
            "groups_contrib.npz", "run_times.json"]
    for f in need:
        if not os.path.exists(os.path.join(OUT, f)): die(f"manifest 불가 — {f} 없음")
    for s in SEEDS:
        if not cache_valid(s, src_h): die(f"manifest 불가 — seed {s} 무효(marker/검증)")
    # R2-6: 존재 여부가 아니라 내장 계보를 현재 source·실제 해시와 비교 — stale이면 차단
    cur_npz = {str(s): sha256_file(paths(s)[0]) for s in SEEDS}
    cur_meta = {str(s): sha256_file(paths(s)[1]) for s in SEEDS}
    for fname, npz_key, npz_file in (("aggregate.json", "aggregate_arrays_sha256", "aggregate_arrays.npz"),
                                     ("groups.json", "groups_contrib_sha256", "groups_contrib.npz")):
        j = load_json_or_die(os.path.join(OUT, fname), fname)
        for k in ("data_sha256", "prereg_sha256", "code_sha256", "environment_fingerprint"):
            if j.get(k) != src_h[k]: die(f"manifest 차단 — {fname} {k} stale(현재 source와 불일치)")
        if j.get("seed_npz_sha256") != cur_npz or j.get("seed_meta_sha256") != cur_meta:
            die(f"manifest 차단 — {fname}의 seed 계보가 현재 20 seed 해시와 불일치(stale)")
        if j.get(npz_key) != sha256_file(os.path.join(OUT, npz_file)):
            die(f"manifest 차단 — {fname} ↔ {npz_file} 해시 불일치(stale)")
    rt = load_json_or_die(os.path.join(OUT, "run_times.json"), "run_times")
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)
                if os.path.isfile(os.path.join(OUT, f)))
    man = {"experiment_id": "oof_shap_v2_rev2",
           "run_started_at": rt["run_started_at"], "run_completed_at": rt["run_completed_at"],  # R2-6 분리
           "created_at": now(),
           "prereg_sha256": src_h["prereg_sha256"], "code_sha256": src_h["code_sha256"],
           "render_code_sha256": sha256_file(os.path.join(ROOT, "src", "render_oof_shap_v2.py")),
           "data_sha256": src_h["data_sha256"], "environment_fingerprint": src_h["environment_fingerprint"],
           "env": src_h["_env"], "seeds": SEEDS,
           "seed_npz_sha256": {s: sha256_file(paths(s)[0]) for s in SEEDS},
           "seed_meta_sha256": {s: sha256_file(paths(s)[1]) for s in SEEDS},
           "aggregate_sha256": sha256_file(os.path.join(OUT, "aggregate.json")),
           "aggregate_arrays_sha256": sha256_file(os.path.join(OUT, "aggregate_arrays.npz")),
           "groups_sha256": sha256_file(os.path.join(OUT, "groups.json")),
           "groups_contrib_sha256": sha256_file(os.path.join(OUT, "groups_contrib.npz")),
           "total_size_bytes": int(total),
           "QC_summary": {"additivity_rule": f"atol=rtol={ATOL}", "logit_qc": f"eps={LOGIT_EPS}, tol={LOGIT_TOL}",
                       "oof_auc_QC전용": [json.load(open(paths(s)[1], encoding="utf-8"))["oof_auc_QC전용"] for s in SEEDS]},
           "protected_snapshot": {p: sha256_file(os.path.join(ROOT, p)) for p in PROTECTED
                               if os.path.exists(os.path.join(ROOT, p))}}
    atomic_json(os.path.join(OUT, "run_manifest.json"), man)   # manifest 자기 해시는 내부에 넣지 않음(R1-13)
    print(f"→ run_manifest.json (총 {total/1e6:.1f}MB)")


def mode_verify():
    src_h = source_hashes()
    ok = sum(cache_valid(s, src_h) for s in SEEDS)
    print(f"유효 seed 캐시 {ok}/{len(SEEDS)} · env fingerprint {src_h['environment_fingerprint'][:16]}…")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    ap = argparse.ArgumentParser(description="oof_shap_v2 Rev1 — 사전등록_OOF_SHAP_v2.md(R1) 집행")
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("probe", "run", "aggregate", "groups", "manifest", "verify"):
        g.add_argument(f"--{m}", action="store_true")
    a = ap.parse_args()
    (mode_probe if a.probe else mode_run if a.run else mode_aggregate if a.aggregate
     else mode_groups if a.groups else mode_manifest if a.manifest else mode_verify)()
