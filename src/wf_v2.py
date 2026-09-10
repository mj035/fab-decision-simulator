"""
wf_v2.py — walk-forward + 이상탐지③ 재실행 (리뷰어 23차 §3). 시간축=행 인덱스(생산 순서).

(a) walk-forward: Q1→Q2, Q1+2→Q3, Q1~3→Q4 (행 순서 기준, 5 seed)
    + 학습크기 맞춘 랜덤 부분집합 대조(테스트 구간 제외 전체서 추출, 5 draw)
    → 전방 페널티(=대조 − 전방)와 학습량 효과 분리
(b) 이상탐지 ③ novelty 전제 재검증: Q1~3 학습 → 홀드아웃 Q1~3 vs Q4 (AUC 주지표, recall 보조)
    + Q4 양성 n 명시 + AUC 95% CI(Hanley–McNeil) 병기 → '판별 불가'인지 명시
구버전(손상시간 Q4: AUC 0.625 vs 0.731)은 폐기. 산출: results/wf_v2.json, results/anomaly3_v2.json
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_adapter import adapt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
raw = pd.read_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"))
# ⚠️ 정렬 금지 — 행 순서 = 생산 순서
X, y, info = adapt(raw.drop(columns=["Time"]), "Pass/Fail", 1); y = np.asarray(y); N = len(y)
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xv = X[valid]
q = np.floor(np.arange(N) / N * 4).astype(int); q[q == 4] = 3

def mkpipe(spw, seed=0):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()),
        ("m", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbose=-1))])

def auc_ci(auc, n1, n0):
    """Hanley–McNeil 95% CI 반폭."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return 1.96 * se

# ── (a) walk-forward + 크기맞춤 랜덤 대조 ──
rng = np.random.default_rng(0)
steps = [(np.where(q <= k)[0], np.where(q == k + 1)[0], f"Q1~{k+1}→Q{k+2}") for k in range(3)]
wf = []
print("[(a)] walk-forward (행 순서) vs 크기맞춤 랜덤 대조:")
for tr, te, name in steps:
    n_tr = len(tr)
    fwd = []
    for s in range(5):
        spw = (y[tr] == 0).sum() / max(y[tr].sum(), 1)
        sc = mkpipe(spw, s).fit(Xv.iloc[tr], y[tr]).predict_proba(Xv.iloc[te])[:, 1]
        fwd.append(roc_auc_score(y[te], sc))
    pool = np.setdiff1d(np.arange(N), te)
    ctl = []
    for s in range(5):
        sub = rng.choice(pool, n_tr, replace=False)
        if y[sub].sum() < 2: continue
        spw = (y[sub] == 0).sum() / max(y[sub].sum(), 1)
        sc = mkpipe(spw, s).fit(Xv.iloc[sub], y[sub]).predict_proba(Xv.iloc[te])[:, 1]
        ctl.append(roc_auc_score(y[te], sc))
    fm, fs, cm, cs_ = np.mean(fwd), np.std(fwd), np.mean(ctl), np.std(ctl)
    n1 = int(y[te].sum()); ci = auc_ci(max(fm, 0.5), n1, len(te) - n1)
    wf.append({"step": name, "train_n": n_tr, "test_npos": n1,
               "fwd_auc": [round(fm, 3), round(fs, 3)], "ctrl_auc": [round(cm, 3), round(cs_, 3)],
               "penalty": round(cm - fm, 3), "auc_ci95_halfwidth": round(ci, 3)})
    print(f"  {name}: 전방 {fm:.3f}±{fs:.3f} · 대조 {cm:.3f}±{cs_:.3f} · 페널티 {cm-fm:+.3f} (양성 {n1}, CI반폭 ±{ci:.3f})")
pen = [w["penalty"] for w in wf]
concl_a = (f"평균 전방 페널티 {np.mean(pen):+.3f}. 페널티가 CI 반폭보다 작으면 '유의한 전방 열화 없음'으로 서술. "
           "대조가 미래 행을 포함하므로 페널티=순수 전방성 효과(학습량 통제됨).")
print(f"  → {concl_a}")
json.dump({"프로토콜": "행 인덱스 quartile. 전방=과거만 학습(5 seed). 대조=테스트 제외 전체서 같은 크기 무작위(5 draw). "
                   "구버전 시간홀드아웃(2008-01~08/09~12, 962/605)은 실재하지 않는 분할 — 철회.",
           "steps": wf, "평균페널티": round(float(np.mean(pen)), 3), "결론": concl_a},
          open(os.path.join(ROOT, "results", "wf_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→ results/wf_v2.json")

# ── (b) 이상탐지 ③ novelty 전제 재검증 ──
print("\n[(b)] 이상탐지③ — Q4가 '새 모드'인가 (행 순서):")
tr123, q4m = q < 3, q == 3
X123, y123 = Xv[tr123], y[tr123]; Xq4, yq4 = Xv[q4m], y[q4m]
n123p, nq4p = int(y123.sum()), int(yq4.sum())
def rec_bud(score, yt, b):
    kk = max(1, int(round(b * len(yt)))); return yt[np.argsort(-score)[:kk]].sum() / max(yt.sum(), 1)
rA, rB = {0.10: [], 0.20: []}, {0.10: [], 0.20: []}; aucA, aucB = [], []
spw = (y123 == 0).sum() / max(n123p, 1)
for s in range(5):
    oof123 = cross_val_predict(mkpipe(spw, s), X123, y123, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                               method="predict_proba", n_jobs=-1)[:, 1]
    sq4 = mkpipe(spw, s).fit(X123, y123).predict_proba(Xq4)[:, 1]
    for b in (0.10, 0.20):
        rA[b].append(rec_bud(oof123, y123, b)); rB[b].append(rec_bud(sq4, yq4, b))
    aucA.append(roc_auc_score(y123, oof123)); aucB.append(roc_auc_score(yq4, sq4))
a123, aq4 = float(np.mean(aucA)), float(np.mean(aucB))
ci4 = auc_ci(max(aq4, 0.5), nq4p, int((yq4 == 0).sum()))
print(f"  AUC: 홀드아웃 Q1~3 {a123:.3f} | Q4 {aq4:.3f} (Δ{a123-aq4:+.3f}) · Q4 양성 n={nq4p} · Q4 AUC CI반폭 ±{ci4:.3f}")
p3 = {}
for b in (0.10, 0.20):
    p3[f"{int(b*100)}%"] = {"holdout_q123": [round(float(np.mean(rA[b])), 3), round(float(np.std(rA[b])), 3)],
                            "q4": [round(float(np.mean(rB[b])), 3), round(float(np.std(rB[b])), 3)]}
    print(f"  recall@{int(b*100)}%: 홀드아웃 {np.mean(rA[b]):.3f}±{np.std(rA[b]):.3f} | Q4 {np.mean(rB[b]):.3f}±{np.std(rB[b]):.3f} (보조, 유병률 {y123.mean()*100:.1f}% vs {yq4.mean()*100:.1f}%)")
delta = a123 - aq4
verdict = (f"Δ{delta:+.3f}이 Q4 AUC CI반폭 ±{ci4:.3f}{'보다 작음 → 판별 불가(구버전과 동일 결론)' if abs(delta) < ci4 else '보다 큼 → 전방 열화 시사(단 새모드 vs covariate shift는 여전히 판별 불가)'}. "
           f"Q4 양성 n={nq4p}. 파트①(IForest가 임계내리기 못 이김)은 시간 무관이라 불변.")
print(f"  판정: {verdict}")
json.dump({"프로토콜": "행 인덱스 Q1~3 학습 → 홀드아웃 Q1~3(OOF) vs Q4. 5 seed. AUC 주지표(유병률 둔감), recall 보조. "
                   "구버전(손상시간 Q4: 0.625 vs 0.731, n=14)은 폐기.",
           "auc": {"holdout_q123": round(a123, 3), "q4": round(aq4, 3), "delta": round(delta, 3),
                   "q4_npos": nq4p, "q4_ci95_halfwidth": round(ci4, 3)},
           "recall": p3, "판정": verdict},
          open(os.path.join(ROOT, "results", "anomaly3_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("→ results/anomaly3_v2.json")
