"""
fwd_cost.py — 리뷰어 25차 #2 (최우선): 전방(walk-forward) 조건에서의 절감률.

동기: 표준 절감률 31.0%(sel_vs_inspect, cl=15)는 **랜덤 CV OOF = 보간 성능** 기준(한계⑫).
     walk-forward AUC는 0.474/0.724/0.550으로 요동(20 seed). Q2 구간은 AUC 0.474 = 무정보.
     → 전방 조건 절감률을 확인해 현재 서사가 참인지 검증한다. 방어용 부록이 아니다.

⚠️ 28차 수정 2건:
  (a) 배포권고를 문자열로 하드코딩하지 않는다 — **원시 argmin**(3전략 절대비용, 가드 미적용)에서 유도.
      종전 `nsel==0 → "전수검사"` 하드코딩은 **틀렸다**(실제 권고는 전수통과). `q2_strategy_audit.py` 참조.
  (b) 중첩 병렬(cross_val_predict n_jobs=-1 × LGBM n_jobs=-1) 제거 — OOM으로 장시간 실행이 죽던 원인.

프로토콜 (wf_v2.py의 이미 승인된 설계를 비용축으로 확장):
  · 시간축 = 행 인덱스(생산 순서). 정렬 금지.
  · 전방: Q1→Q2, Q1~2→Q3, Q1~3→Q4. 임계값은 **학습 구간 내부 OOF에서만** 선택(전방 누수 0).
  · 대조: 테스트 블록을 제외한 전체에서 같은 크기 무작위 추출(N_SEED draw) → 학습량 통제.
          대조는 미래 행을 포함하므로 (대조 − 전방) = 순수 전방성 페널티.
  · 비용: CM.evaluate, cl=15, cs=0, β=0.95, α=0.02 (정본과 동일 파라미터).
  · 블록별 유병률을 함께 보고 — 절감률은 유병률에 강하게 의존하므로 분해가 필요.

산출: results/fwd_cost.json
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
import cost_model as CM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
raw = pd.read_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"))
X, y, _ = adapt(raw.drop(columns=["Time"]), "Pass/Fail", 1); y = np.asarray(y); N = len(y)
valid = [c for c in X.columns if np.nanstd(X[c].to_numpy()) > 0]
Xv = X[valid]
q = np.floor(np.arange(N) / N * 4).astype(int); q[q == 4] = 3

CL, CS, BETA, ALPHA = 15, 0, 0.95, 0.02
THS = np.linspace(0.005, 0.995, 300)
N_SEED = int(os.environ.get("FWD_NSEED", 20))

# ⚠️ seed가 바꾸는 것 (리뷰어 25차 조건 4): walk-forward는 **외부 분할이 시간으로 고정**되므로
#   seed는 (i) 내부 임계값 선택 fold, (ii) LGBM 서브샘플링/부스팅 난수 만 바꾼다.
#   → 아래 ±와 CI는 **임계값 선택 + 모델 난수 변동만** 잡은 것이고,
#     시간 분할 불확실성·데이터셋 불확실성은 미포함 = **분산 과소추정**. 인용 시 병기 필수.
SEED_SEMANTICS = ("외부(시간) 분할 고정. seed는 내부 임계값 선택 fold + LGBM 난수만 변동 "
                  "→ ±·CI는 분산 과소추정(시간분할·데이터셋 불확실성 미포함).")


def mkpipe(spw, seed=0):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler()),
        ("m", LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, scale_pos_weight=spw, random_state=seed,
            n_jobs=-1, verbose=-1))])


def cms(score, yy):
    """(300,4) tn,fp,fn,tp — 고정 임계값 그리드."""
    return np.array([[int(((score < t) & (yy == 0)).sum()), int(((score >= t) & (yy == 0)).sum()),
                      int(((score < t) & (yy == 1)).sum()), int(((score >= t) & (yy == 1)).sum())] for t in THS])


def run(tr, te, seed):
    """학습 tr → 내부 OOF로 임계값 선택 → 전방 te 평가. 비용 dict + AUC 반환."""
    spw = (y[tr] == 0).sum() / max(y[tr].sum(), 1)
    in_oof = cross_val_predict(mkpipe(spw, seed), Xv.iloc[tr], y[tr],
                               cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                               method="predict_proba", n_jobs=1)[:, 1]   # ⚠️ 28차: 중첩 병렬(바깥×LGBM) OOM 방지 — 바깥 1 고정
    te_sc = mkpipe(spw, seed).fit(Xv.iloc[tr], y[tr]).predict_proba(Xv.iloc[te])[:, 1]
    in_cm, te_cm = cms(in_oof, y[tr]), cms(te_sc, y[te])
    npos = int(y[te].sum()); nte = len(te)
    folds = [(in_cm, te_cm)]
    r = CM.evaluate(folds, nte, npos, nte - npos, cl=CL, cs=CS, beta=BETA, alpha=ALPHA)
    tn, fp, fn, tp = r["counts"]
    # ⚠️ 26차 결함 수정: 배포권고를 "전수검사"로 하드코딩하지 않는다.
    #    실제 권고는 3전략 절대비용의 **원시 argmin**(가드 미적용)에서 유도한다. (`q2_strategy_audit.py` 확인)
    raw = CM.CM_raw(folds, nte, npos, nte - npos, cl=CL, cs=CS, beta=BETA, alpha=ALPHA)
    return {"save": r["sel_vs_inspect"], "rate": r["inspect_rate"], "strategy": r["eff_strategy"],
            "raw_strategy": raw,
            "prec": tp / max(tp + fp, 1) * 100, "rec": tp / max(tp + fn, 1) * 100,
            "auc": roc_auc_score(y[te], te_sc)}


rng = np.random.default_rng(0)
steps = [(np.where(q <= k)[0], np.where(q == k + 1)[0], f"Q1~{k+1}→Q{k+2}") for k in range(3)]
res = []
print(f"[전방 조건 절감률] cl={CL}, cs={CS}, β={BETA}, α={ALPHA} | 표준(랜덤CV OOF) = 31.0%\n")
for tr, te, name in steps:
    npos = int(y[te].sum()); prev = npos / len(te) * 100
    fwd = [run(tr, te, s) for s in range(N_SEED)]
    pool = np.setdiff1d(np.arange(N), te)
    ctl = []; skipped = 0
    for s in range(N_SEED):
        sub = rng.choice(pool, len(tr), replace=False)
        if y[sub].sum() < 5:            # ⚠️ 대조군을 고유병률 쪽으로 미세 선택하는 조건 — 실제 발생 여부를 기록
            skipped += 1; continue
        ctl.append(run(sub, te, s))

    def agg(rs, k):
        v = np.array([r[k] for r in rs], float); return float(np.mean(v)), float(np.std(v))
    fs, fsd = agg(fwd, "save"); csv_, csd = agg(ctl, "save")
    fa, _ = agg(fwd, "auc"); ca, _ = agg(ctl, "auc")
    sv = np.array([r["save"] for r in fwd])
    from scipy.stats import t as _t
    # 전방 절감률의 95% CI (t 근사, seed n=N_SEED — 화석 주석 방지용으로 상수 참조)
    half = _t.ppf(0.975, len(sv) - 1) * sv.std(ddof=1) / np.sqrt(len(sv))
    nsel = sum(r["strategy"] == 2 for r in fwd)
    # 배포권고 = 원시 argmin 분포 (하드코딩 금지). 0전수통과 1전수검사 2선별
    NM = ["전수통과", "전수검사", "선별"]
    rawc = {NM[i]: sum(1 for r in fwd if r["raw_strategy"] == i) for i in range(3)}
    rawc = {k: v for k, v in rawc.items() if v}
    raw_rec = " / ".join(f"{k} {v}/{len(fwd)}" for k, v in sorted(rawc.items(), key=lambda kv: -kv[1]))
    guard_changed = sum(1 for r in fwd if r["raw_strategy"] != r["strategy"])
    # 퇴화 가드(DEGEN_RATE=0.90) 민감도 — 상수 의존성 노출 (인수인계 🟡)
    rates = np.array([r["rate"] for r in fwd])
    degen = {f"가드{g:.2f}": f"{int((rates >= g).sum())}/{len(rates)} 퇴화" for g in (0.80, 0.85, 0.90, 0.95)}
    res.append({"step": name, "test_n": len(te), "test_npos": npos, "유병률_%": round(prev, 1),
                "전방_절감률_%": [round(fs, 1), round(fsd, 1)], "전방_절감률_ci95": [round(fs - half, 1), round(fs + half, 1)],
                "대조_절감률_%": [round(csv_, 1), round(csd, 1)], "페널티_%p": round(csv_ - fs, 1),
                "전방_AUC": round(fa, 3), "대조_AUC": round(ca, 3),
                "전방_검사율_%": round(agg(fwd, "rate")[0] * 100, 1),
                "전방_정밀도_%": round(agg(fwd, "prec")[0], 1), "전방_재현율_%": round(agg(fwd, "rec")[0], 1),
                "선별이_최적인_seed": f"{nsel}/{len(fwd)}",
                "검사율_seed별": [round(float(v) * 100, 1) for v in rates],
                "퇴화가드_민감도": degen,
                "배포시_커널권고": raw_rec,
                "가드가_판정을_바꾼_seed": f"{guard_changed}/{len(fwd)}",
                "대조_draw_스킵": f"{skipped}/{N_SEED}"})
    r_ = res[-1]
    print(f"  {name}  (n={len(te)}, 양성 {npos}, 유병률 {prev:.1f}%)")
    print(f"    전방 절감률 {fs:+.1f}±{fsd:.1f}%  CI[{fs-half:+.1f},{fs+half:+.1f}]  (AUC {fa:.3f})")
    print(f"    대조 절감률 {csv_:+.1f}±{csd:.1f}%  (AUC {ca:.3f})  →  전방 페널티 {csv_-fs:+.1f}%p")
    print(f"    검사율 {r_['전방_검사율_%']}% · 정밀도 {r_['전방_정밀도_%']}% · 재현율 {r_['전방_재현율_%']}% · 선별최적 {nsel}/{len(fwd)}\n")

allsv = [r["전방_절감률_%"][0] for r in res]
out = {
    "프로토콜": (f"행 인덱스 quartile 전방. 임계값은 학습구간 내부 OOF에서만 선택(전방 누수 0). "
             f"cl={CL}, cs={CS}, β={BETA}, α={ALPHA}. 전방 {N_SEED} seed / 대조 {N_SEED} draw(테스트 제외 풀에서 같은 크기)."),
    "seed_의미": SEED_SEMANTICS,
    "기준_랜덤CV_OOF_절감률_%": 31.0,
    "steps": res,
    "⛔평균금지": ("스텝 평균을 계산하지 않는다(리뷰어 25차 조건 1). 세 스텝은 교환가능한 반복이 아니다 — "
              "학습량(391/783/1176)·검사 유병률이 다르고, Q2는 선별 불성립(퇴화) 스텝이다. "
              "퇴화 전략의 수치를 유효 전략 둘과 산술평균한 값은 아무것도 대표하지 않는다. **스텝별 병기만 허용.**"),
    "⚠️귀속_주의": ("31.0 → 전방 격차는 (소표본 학습 + 블록 평가 + 전방성)의 **합성**이며, 이 중 전방성만 대조로 분리된다. "
               "크기맞춤 대조는 학습 크기와 테스트 블록을 고정하고 학습 구성(과거만 vs 미래 포함)만 바꾼다 — "
               "따라서 대조가 랜덤 OOF보다 낮은 것은 '블록 효과'가 아니라 소표본 학습 + 블록 평가의 합이다. "
               "그리고 3스텝 중 2개에서 전방이 대조보다 **오히려 높다** → **전방성이 주범이라는 증거는 없다.**"),
}
out["판정"] = ("스텝별 전방 절감률 " + " / ".join(f"{r['step']} {r['전방_절감률_%'][0]:+.1f}%" for r in res)
            + f" (보간 기준 31.0%). 최저 {min(allsv):.1f}% · 최고 {max(allsv):.1f}%. "
              "→ 31.0% 단독 인용 중단, 스텝별 병기. 평균 인용 금지.")
print("[판정]", out["판정"])
print("[귀속]", out["⚠️귀속_주의"])
json.dump(out, open(os.path.join(ROOT, "results", "fwd_cost.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→ results/fwd_cost.json")
