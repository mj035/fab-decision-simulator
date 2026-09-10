# -*- coding: utf-8 -*-
"""
final_preprocessing_compare.py — 발표 정본용 전처리(결측 대치) 8종 비교 (고정 LightGBM · 50 seed)

멘토 사양 반영:
  · ⭐ 모델은 8종 모두 **동일한 LightGBM** (운영 모델) → 전처리만 변수로 분리
    (기존 imputation.py는 RF 기반이라 최종 운영 모델과 불일치 — 그래서 새로 짬)
  · 대치·상수제거·고결측판단을 전부 outer training fold 안에서만 결정 (test fold 통계 금지)
  · seed 0~49, 동일 outer 5-fold · ROC-AUC·PR-AUC 둘 다 fold-mean 주지표
  · 중앙값 대비 paired 차이·95% CI·승/동/패

방법 8종: 중앙값 / 평균 / 0채움 / 최빈값 / KNN(k=5) / 중앙값+결측표시 /
          고결측(>50%)제거+중앙값 / LightGBM 네이티브(NaN 그대로)

격리: 정본·캐시·그림 미수정. 신규 경로만. 출력 → results/final_preprocessing_compare/result.json
재현: python src/final_preprocessing_compare.py --run  (프로브 --probe · KNN 제외 --no-knn)
⚠️ KNN 대치는 50×5=250회라 느림. 시간 부족 시 --no-knn 로 7종만.
"""
import os, sys, json, argparse, warnings

SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
COMP = os.path.dirname(ROOT)
DATA_CSV = os.path.join(ROOT, "data", "fab_process_yield.csv")
OUT_DIR  = os.path.join(ROOT, "results", "final_preprocessing_compare")

NFOLD = 5
DEFAULT_NSEED = 50
NJOBS = int(os.environ.get("FPC_NJOBS", "-1"))
REF = "① 중앙값"

METHODS = ["① 중앙값", "② 평균", "③ 0채움", "④ 최빈값", "⑤ KNN(k=5)",
           "⑥ 중앙값+결측표시", "⑦ 고결측제거+중앙값", "⑧ LGBM 네이티브"]


def load_xy():
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    y = (df["Pass/Fail"] == 1).astype(int).to_numpy()
    X = df.drop(columns=["Time", "Pass/Fail"]).astype(float)
    return X, y


def lgbm(spw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
                          scale_pos_weight=spw, random_state=0, n_jobs=NJOBS, verbose=-1)


def transform(method, Xtr_df, Xte_df):
    """training fold에서만 fit. (train행렬, test행렬) 반환. 방법별 분기."""
    import numpy as np
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    Xtr = Xtr_df.to_numpy(float); Xte = Xte_df.to_numpy(float)

    if method == "⑧ LGBM 네이티브":
        return Xtr, Xte                                    # NaN 그대로 LightGBM에

    # 결측표시(⑥): 대치 전 마스크 (train fold 결측열만)
    ind_tr = ind_te = None
    if method == "⑥ 중앙값+결측표시":
        keep = np.isnan(Xtr).any(0)
        ind_tr = np.isnan(Xtr)[:, keep].astype(float)
        ind_te = np.isnan(Xte)[:, keep].astype(float)

    # 고결측 센서 제거(⑦): train 결측률 기준
    if method == "⑦ 고결측제거+중앙값":
        keep_col = np.isnan(Xtr).mean(0) <= 0.5
        Xtr, Xte = Xtr[:, keep_col], Xte[:, keep_col]

    imp = {"① 중앙값": SimpleImputer(strategy="median"),
           "② 평균": SimpleImputer(strategy="mean"),
           "③ 0채움": SimpleImputer(strategy="constant", fill_value=0),
           "④ 최빈값": SimpleImputer(strategy="most_frequent"),
           "⑤ KNN(k=5)": KNNImputer(n_neighbors=5),
           "⑥ 중앙값+결측표시": SimpleImputer(strategy="median"),
           "⑦ 고결측제거+중앙값": SimpleImputer(strategy="median"),
           }[method].fit(Xtr)
    Atr, Ate = imp.transform(Xtr), imp.transform(Xte)

    var = VarianceThreshold(0.0).fit(Atr)
    sc  = StandardScaler().fit(var.transform(Atr))
    Ztr = sc.transform(var.transform(Atr))
    Zte = sc.transform(var.transform(Ate))

    if ind_tr is not None:                                 # 결측표시 열 append (스케일 안 함)
        Ztr = np.hstack([Ztr, ind_tr]); Zte = np.hstack([Zte, ind_te])
    return Ztr, Zte


def paired(delta):
    import numpy as np
    d = np.asarray(delta, float); n = len(d)
    mean = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    return dict(mean=mean, sd=sd, ci95=[mean - 1.96*se, mean + 1.96*se],
                win=int((d > 1e-12).sum()), tie=int((abs(d) <= 1e-12).sum()),
                loss=int((d < -1e-12).sum()), n=n)


def run(nseed, methods):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_xy()
    base = float(y.mean())
    print(f"데이터 {X.shape[0]}행 × {X.shape[1]}열 · 양성 {int(y.sum())} · 방법 {len(methods)}종 · seed {nseed}", flush=True)

    seed_roc = {m: [] for m in methods}; seed_pr = {m: [] for m in methods}
    raw = {m: {"roc": [], "pr": []} for m in methods}

    for s in range(nseed):
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=s)
        f_roc = {m: [] for m in methods}; f_pr = {m: [] for m in methods}
        for tr, te in skf.split(X, y):
            Xtr_df, Xte_df, ytr = X.iloc[tr], X.iloc[te], y[tr]
            spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
            for m in methods:
                Ztr, Zte = transform(m, Xtr_df, Xte_df)
                p = lgbm(spw).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
                f_roc[m].append(roc_auc_score(y[te], p))
                f_pr[m].append(average_precision_score(y[te], p))
        for m in methods:
            seed_roc[m].append(float(np.mean(f_roc[m])))
            seed_pr[m].append(float(np.mean(f_pr[m])))
            raw[m]["roc"].append([float(v) for v in f_roc[m]])
            raw[m]["pr"].append([float(v) for v in f_pr[m]])
        print(f"seed {s:2d} done", flush=True)

    ref = REF if REF in methods else methods[0]
    summary = {}
    for m in methods:
        summary[m] = dict(
            roc_mean=float(np.mean(seed_roc[m])), roc_sd=float(np.std(seed_roc[m], ddof=1)),
            pr_mean=float(np.mean(seed_pr[m])),   pr_sd=float(np.std(seed_pr[m], ddof=1)),
            d_roc_vs_median=paired(np.array(seed_roc[m]) - np.array(seed_roc[ref])),
            d_pr_vs_median=paired(np.array(seed_pr[m]) - np.array(seed_pr[ref])),
        )
    result = dict(protocol="고정 LightGBM · 50 seed · outer 5-fold · fold-mean · 전처리 fold내부 fit",
                  nseed=nseed, ref=ref, pr_baseline=base, summary=summary,
                  per_seed={"roc": seed_roc, "pr": seed_pr}, raw_fold=raw)
    path = os.path.join(OUT_DIR, "result.json")
    json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 74)
    print(f"{'방법':<20}{'ROC(fold-mean)':>18}{'PR(fold-mean)':>16}{'ΔROC vs 중앙값':>16}")
    for m in sorted(methods, key=lambda k: -summary[k]["roc_mean"]):
        s = summary[m]; d = s["d_roc_vs_median"]
        dtxt = "기준" if m == ref else f"{d['mean']:+.4f}"
        print(f"{m:<20}{s['roc_mean']:>10.4f}±{s['roc_sd']:.3f}{s['pr_mean']:>9.4f}±{s['pr_sd']:.3f}{dtxt:>14}")
    print("=" * 74)
    print(f"→ {path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--no-knn", action="store_true", help="KNN 제외(빠름)")
    a = ap.parse_args()
    ms = [m for m in METHODS if not (a.no_knn and "KNN" in m)]
    if a.probe: run(a.seeds or 3, ms)
    elif a.run: run(a.seeds or DEFAULT_NSEED, ms)
    else: print(__doc__); print("→ --probe 또는 --run")
