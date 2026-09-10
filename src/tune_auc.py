"""
tune_auc.py — AUC 튜닝 실험. **사전등록 문서(`사전등록_AUC튜닝_판정기준.md`) §1~§4를 그대로 집행.**

⚠️ 판정 임계는 이 스크립트를 처음 실행하기 **전에** 고정되었다. 결과를 보고 임계를 바꾸지 않는다.
   ΔAUC ≥ +0.02 / |Δ손익분기cl| ≥ 3 (주지표) / |Δ절감률| ≥ 2%p / |Δ경계중앙| ≥ 8 (부지표)

프로토콜 (§4):
 · nested — 하이퍼파라미터 탐색은 **inner fold에서만**. outer는 평가 전용. (`optuna_vs_leakage.py`의 교훈)
 · 베이스라인도 **동일 코드 경로**로 재계산한다. 캐시(cms_seeds)와 비교하면 코드 차이가 교란된다.
 · 동일 50 seed, 동일 CM.evaluate, 손익분기는 "이후 단조유지" 규칙 + 정의역 cl∈[2,64].

산출: results/tune_auc.json
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
NSEED = int(os.environ.get("TUNE_NSEED", 50))
CLS = list(range(2, 65))                     # 정의역 cl ∈ [2,64]
THS = np.linspace(0.005, 0.995, 300)

X, y, _ = adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                         "Pass/Fail", 1, drop_cols=["Time"])
y = np.asarray(y); N = len(y); npos = int(y.sum()); nneg = N - npos
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xv = X[valid].reset_index(drop=True)

BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0)
# 탐색공간 (사전 고정, 6개). 부록에 그대로 공개할 것.
GRID = [
    BASE,
    dict(BASE, num_leaves=15, min_child_samples=15),
    dict(BASE, num_leaves=31, min_child_samples=10, reg_lambda=1.0),
    dict(BASE, n_estimators=800, learning_rate=0.01, colsample_bytree=0.2),
    dict(BASE, n_estimators=150, learning_rate=0.08, num_leaves=15),
    dict(BASE, colsample_bytree=0.6, reg_lambda=30.0, min_child_samples=40),
]


def mk(cfg, spw, seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()),
        ("m", LGBMClassifier(**cfg, scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1))])


def cms(score, yy):
    return np.array([[int(((score < t) & (yy == 0)).sum()), int(((score >= t) & (yy == 0)).sum()),
                      int(((score < t) & (yy == 1)).sum()), int(((score >= t) & (yy == 1)).sum())] for t in THS])


def run_arm(seed, tune):
    """한 seed의 5 outer fold. tune=True면 inner에서 config 선택(nested). 반환: folds, oof, 선택된 config들."""
    folds = []; oof = np.zeros(N); picked = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xv, y):
        spw = (y[tr] == 0).sum() / max(y[tr].sum(), 1)
        if tune:
            best, best_auc = None, -1
            for ci, cfg in enumerate(GRID):
                p = cross_val_predict(mk(cfg, spw, seed), Xv.iloc[tr], y[tr],
                                      cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                      method="predict_proba", n_jobs=1)[:, 1]   # ⚠️ 바깥 병렬 1 고정
                a = roc_auc_score(y[tr], p)
                if a > best_auc: best, best_auc, bi = cfg, a, ci
            picked.append(bi)
        else:
            best = BASE; picked.append(0)
        in_oof = cross_val_predict(mk(best, spw, seed), Xv.iloc[tr], y[tr],
                                   cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                   method="predict_proba", n_jobs=1)[:, 1]
        sc = mk(best, spw, seed).fit(Xv.iloc[tr], y[tr]).predict_proba(Xv.iloc[te])[:, 1]
        oof[te] = sc
        folds.append((cms(in_oof, y[tr]), cms(sc, y[te])))
    return folds, oof, picked


def best_cost(folds, cl):
    lam, mu, rho = CM.to_effective(cl, 0, B, A)
    sel, agg, _, _ = CM._accumulate(folds, lam, mu, rho, cl)
    pass_all = npos * cl; inspect_all = N + npos * cl - npos * lam + nneg * mu
    rate = (agg[3] + agg[1]) / N
    return min(pass_all, inspect_all, inspect_all if rate >= CM.DEGEN_RATE else sel)


def durable_be(D):
    m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D)); sig = m - 1.96 * se > 0
    be = None
    for i in range(len(CLS) - 1, -1, -1):
        if sig[i]: be = CLS[i]
        else: break
    return be


# ── seed별 캐시 (장시간 실행 중 죽어도 재개 가능) ──
CACHE = os.path.join(ROOT, "results", "tune_cache"); os.makedirs(CACHE, exist_ok=True)
res = {}
t0 = time.time()
for arm in ("baseline", "tuned"):
    aucs, saves, costs, picks = [], [], [], []
    for s in range(NSEED):
        cf = os.path.join(CACHE, f"{arm}_s{s}.npz")
        if os.path.exists(cf):
            d = np.load(cf)
            aucs.append(float(d["auc"])); saves.append(float(d["save"]))
            costs.append(d["costs"]); picks += list(d["picks"])
            continue
        folds, oof, pk = run_arm(s, tune=(arm == "tuned"))
        a = roc_auc_score(y, oof)
        r = CM.evaluate(folds, N, npos, nneg, cl=15, cs=0, beta=B, alpha=A)
        cst = np.array([best_cost(folds, cl) for cl in CLS])
        np.savez(cf, auc=a, save=r["sel_vs_inspect"], costs=cst, picks=np.array(pk))
        aucs.append(a); saves.append(r["sel_vs_inspect"]); costs.append(cst); picks += pk
        print(f"  [{arm}] seed {s+1}/{NSEED} 완료  누적AUC {np.mean(aucs):.4f}  절감 {np.mean(saves):.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    res[arm] = {"auc": [float(np.mean(aucs)), float(np.std(aucs))],
                "save15": [float(np.mean(saves)), float(np.std(saves))],
                "costs": np.array(costs), "picks": picks}
    print(f"[{arm}] AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f} · cl15 절감 {np.mean(saves):.1f}±{np.std(saves):.1f}%")

# 손익분기: 대응 절대비용차 (baseline − tuned) > 0 이면 tuned 우위 … 가 아니라
# 주지표는 "단일센서 대비 손익분기 cl"이므로 각 arm을 **센서 비용**과 비교해야 한다.
# → 센서 비용은 arm과 무관하므로 breakeven_domain.py의 센서 곡선을 재사용한다.
import glob
from sklearn.model_selection import StratifiedKFold as SKF
Xi = Xv.fillna(Xv.median()).to_numpy()


def auc_cols(idx):
    sub = Xi[idx]; yy = y[idx]; p = yy == 1; np_ = p.sum(); nn_ = len(idx) - np_
    return ((sub.argsort(0).argsort(0) + 1)[p].sum(0) - np_ * (np_ + 1) / 2) / (np_ * nn_)


def sensor_cost_row(seed):
    folds = []
    for tr, te in SKF(5, shuffle=True, random_state=seed).split(Xi, y):
        a = auc_cols(tr); j = int(np.argmax(np.maximum(a, 1 - a))); sign = 1.0 if a[j] >= 0.5 else -1.0
        s_tr, s_te = Xi[tr, j] * sign, Xi[te, j] * sign
        ths = np.quantile(s_tr, np.linspace(0.01, 0.99, 300))
        cm = lambda sc, idx: np.array([[int(((sc < t) & (y[idx] == 0)).sum()), int(((sc >= t) & (y[idx] == 0)).sum()),
                                        int(((sc < t) & (y[idx] == 1)).sum()), int(((sc >= t) & (y[idx] == 1)).sum())] for t in ths])
        folds.append((cm(s_tr, tr), cm(s_te, te)))
    return [best_cost(folds, cl) for cl in CLS]


sens = np.array([sensor_cost_row(s) for s in range(NSEED)])
out = {"프로토콜": f"nested({len(GRID)} config, inner 3-fold) vs 베이스라인, 동일 코드경로·동일 {NSEED} seed. "
                f"손익분기=단조유지 규칙, 정의역 cl∈[2,64].",
       "탐색공간": [{k: v for k, v in c.items()} for c in GRID],
       "사전등록_임계": {"ΔAUC": 0.02, "Δ손익분기cl(주지표)": 3, "Δ절감률%p": 2}}
for arm in ("baseline", "tuned"):
    be = durable_be(sens - res[arm]["costs"])
    out[arm] = {"auc": [round(res[arm]["auc"][0], 4), round(res[arm]["auc"][1], 4)],
                "save15_%": [round(res[arm]["save15"][0], 1), round(res[arm]["save15"][1], 1)],
                "손익분기_cl": be,
                "선택된_config_분포": {str(i): res[arm]["picks"].count(i) for i in range(len(GRID))}}

d_auc = out["tuned"]["auc"][0] - out["baseline"]["auc"][0]
d_be = (out["tuned"]["손익분기_cl"] - out["baseline"]["손익분기_cl"]
        if out["tuned"]["손익분기_cl"] and out["baseline"]["손익분기_cl"] else None)
d_sv = out["tuned"]["save15_%"][0] - out["baseline"]["save15_%"][0]
auc_up = d_auc >= 0.02
be_moved = (d_be is not None and abs(d_be) >= 3)

if d_be is None:
    scen = "판정보류"; act = ("주지표(손익분기 cl)가 한쪽 arm에서 **미정의** — 단조유지 조건이 정의역 내에서 성립하지 않음. "
                          "사전등록에 이 경우가 없으므로 **임의 판정하지 않고 리뷰어에게 반려**한다. "
                          f"(baseline={out['baseline']['손익분기_cl']}, tuned={out['tuned']['손익분기_cl']})")
elif not auc_up:
    scen = "A"; act = ("헤드라인 3번 유지. 서술은 **'우리가 시도한 튜닝 범위에서는 AUC가 오르지 않았다'**로 한정하고 "
                       "'AUC를 올릴 수 없다'로 확대 금지. 탐색공간을 부록에 명시.")
elif not be_moved:
    scen = "B"; act = (f"헤드라인 3번 **강화**. 본문 서술 정본: **'AUC {d_auc:+.3f}에도 관측 Δ손익분기cl = {d_be} "
                       f"(< 임계 3, 분할변동만 반영한 잡음 기준 — 데이터셋 불확실성 미포함)'**. "
                       "⚠️ 'cl 불변'이라고 쓰지 않는다(사전등록 §1 개정).")
else:
    scen = "C"; act = ("사전등록 §3-C 집행: (1) 헤드라인 3번을 '이 데이터의 AUC 범위에서는 결정이 AUC에 둔감하다'로 **축소**, "
                       "(2) (ΔAUC, Δcl) 쌍을 본문 표로 **공개**, (3) 역방향('AUC 올리면 비용 개선') 주장 **금지**, "
                       "(4) 12절 체크리스트 + 교정질문 3개 통과 후 본문 반영.")
out["판정"] = {"ΔAUC": round(d_auc, 4), "Δ손익분기cl": d_be, "Δ절감률_%p": round(d_sv, 1),
              "AUC_올랐나": bool(auc_up), "주지표_움직였나": bool(be_moved),
              "시나리오": scen, "사전등록_대응": act}
print(f"\n[판정] 시나리오 {scen} · ΔAUC {d_auc:+.4f} · Δ손익분기cl {d_be} · Δ절감률 {d_sv:+.1f}%p")
print("      ", act)
json.dump(out, open(os.path.join(ROOT, "results", "tune_auc.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/tune_auc.json")
