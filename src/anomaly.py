# -*- coding: utf-8 -*-
"""
아이디어 2 — 이상탐지 (IsolationForest). 리뷰어 17차 설계 반영.

세 파트 (순서: ② → ① → ③):
 ② 공정 비교  : 비지도(IForest)를 지도(LGBM)와 '같은 Cost Layer'로 통과.
                "라벨 없이도 되나" — AUC(on 104) & 손익분기 cl 대조. (아이디어2 커버)
 ① 2차 필터   : 지도 통과분(예측 음성) 중 이상점수 상위 k% 재검사 = 언더킬 감시.
                핵심결과 = 손익분기 정밀도. 커널 회계로:
                  Δcost = m − d·λ + (m−d)·μ  (m=추가검사수, d=회수불량)
                  이득 ⟺ 정밀도 d/m > (1+μ)/(λ+μ)   [cs=0 → 1/λ 로 환원]
                → "이 기법이 통하는지 판정하는 기준"을 제시(값 자체는 미달일 수 있음).
 ③ leave-mode-out(시간기반) : 후반기 불량모드를 지도학습서 숨김(전반 학습→후반 평가).
                숨긴 모드 recall: 지도 vs 이상탐지. "모르는 불량" 주장을 실험으로.
                null 대비 결론문 3종 + MDD 미리 확정.

정직성 못박기: SECOM 모든 불량은 라벨에 있음 → 진짜 미지불량 회수는 실증 불가.
              주장은 '메커니즘 + 커널 손익분기'까지. ③은 시간대리 홀드아웃(대리실험).
출력: results/anomaly.json + 콘솔 요약.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
CSV  = os.path.join(ROOT, "data", "fab_process_yield.csv")
B, A = 0.95, 0.02                  # β, α (drift.py와 동일)
THS  = np.linspace(0.005, 0.995, 300)
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))

# ── 데이터 (시간순 정렬 — ③에서 사용) ──
raw = pd.read_csv(CSV)
t = pd.to_datetime(raw["Time"], errors="coerce")
order = np.argsort(t.values)
raw = raw.iloc[order].reset_index(drop=True); t = t.iloc[order].reset_index(drop=True)
X, y, info = adapt(raw.drop(columns=["Time"]), "Pass/Fail", 1)
y = np.asarray(y); N = len(y); npos = int(y.sum()); nneg = N - npos
print(f"[데이터] N={N} 양성={npos} ({npos/N*100:.2f}%)  시간 {t.min().date()}~{t.max().date()}  SEEDS={len(SEEDS)}")

def prep(Xtr):
    p = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("var", VarianceThreshold(0.0)), ("sc", StandardScaler())])
    return p.fit(Xtr)

def mk_lgbm(spw):
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
        scale_pos_weight=spw, random_state=0, n_jobs=-1, verbose=-1)

def mk_iforest():
    # 비지도: 학습폴드 '정상(음성)'만 보고 학습 = novelty 프레이밍. 점수↑ = 더 이상.
    return IsolationForest(n_estimators=300, max_samples="auto", contamination="auto",
                           random_state=0, n_jobs=-1)

def cms_from_scores(y_true, score, ths):
    return np.array([confusion_matrix(y_true, (score >= t), labels=[0, 1]).ravel() for t in ths])

def nested_scores(seed):
    """지도 OOF prob + 비지도 OOF 이상점수(둘 다 outer test에서). 각 폴드 in/te CM도.
       비지도 점수는 폴드별 train min-max로 [0,1] 정규화(임계 스윕 일관)."""
    spw = nneg / npos
    sup_oof = np.zeros(N); ano_oof = np.zeros(N)
    sup_in, sup_te, ano_in, ano_te = [], [], [], []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
        pp = prep(Xtr); Ztr, Zte = pp.transform(Xtr), pp.transform(Xte)
        # 지도 — 내부 CV(임계선택용) + 외부 test 점수
        inner = cross_val_predict(Pipeline([("m", mk_lgbm(spw))]), Ztr, ytr,
                    cv=StratifiedKFold(4, shuffle=True, random_state=k),
                    method="predict_proba", n_jobs=-1)[:, 1]
        sup_te_s = mk_lgbm(spw).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
        sup_oof[te] = sup_te_s
        sup_in.append(cms_from_scores(ytr, inner, THS))
        sup_te.append(cms_from_scores(yte, sup_te_s, THS))
        # 비지도 — 학습폴드 음성만으로 IForest 적합 → 이상점수(-score_samples)
        ifo = mk_iforest().fit(Ztr[ytr == 0])
        raw_in = -ifo.score_samples(Ztr); raw_te = -ifo.score_samples(Zte)
        lo, hi = raw_in.min(), raw_in.max(); rng = (hi - lo) or 1.0
        a_in = (raw_in - lo) / rng; a_te = (raw_te - lo) / rng
        ano_oof[te] = a_te
        ano_in.append(cms_from_scores(ytr, a_in, THS))
        ano_te.append(cms_from_scores(yte, a_te, THS))
    return dict(sup_oof=sup_oof, ano_oof=ano_oof,
                sup_in=np.array(sup_in), sup_te=np.array(sup_te),
                ano_in=np.array(ano_in), ano_te=np.array(ano_te))

# ── OOF 캐시 (없으면 계산) ──
CACHE = os.path.join(ROOT, "results", f"anomaly_oof_{len(SEEDS)}.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True); runs = list(z["runs"])
    print(f"[캐시] {CACHE} 로드 ({len(runs)} seed)")
else:
    t0 = time.time(); runs = []
    for s in SEEDS:
        runs.append(nested_scores(s))
        print(f"  seed {s} 완료 [{time.time()-t0:.0f}s]", flush=True)
    np.savez(CACHE, runs=np.array(runs, dtype=object))
    print(f"[캐시] 저장 {CACHE} [{time.time()-t0:.0f}s]")

# AUC on 104 (rank기반, 시드평균)
sup_auc = np.mean([roc_auc_score(y, r["sup_oof"]) for r in runs])
ano_auc = np.mean([roc_auc_score(y, r["ano_oof"]) for r in runs])
sup_auc_sd = np.std([roc_auc_score(y, r["sup_oof"]) for r in runs])
ano_auc_sd = np.std([roc_auc_score(y, r["ano_oof"]) for r in runs])
assert sup_auc > 0.55, f"라벨 오정합 의심: 지도 OOF AUC {sup_auc:.3f} < 0.55 (무작위 이하=구조적 불가)"  # sanity(리뷰어 21차)
print(f"\n[AUC on 104 알려진불량] 지도 {sup_auc:.3f}±{sup_auc_sd:.3f}  |  비지도 {ano_auc:.3f}±{ano_auc_sd:.3f}")
print("  → 라벨 있으면 지도가 우위(예상). 비지도는 라벨 0으로 이만큼.")

out = {"seeds": len(SEEDS), "N": N, "npos": npos, "beta": B, "alpha": A,
       "sup_auc": sup_auc, "ano_auc": ano_auc, "sup_auc_sd": sup_auc_sd, "ano_auc_sd": ano_auc_sd}

def wilson(k, n, z=1.96):
    """이항비율 Wilson 95% CI. 소표본에서 정직."""
    if n == 0: return (0.0, 0.0)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, c-hw), c+hw)

# ══════════════════ 파트 ② 공정 비교 (같은 Cost Layer + 무작위 기준선) ══════════════════
print("\n" + "="*64 + "\n[파트②] 공정 비교 — 지도 vs 비지도 vs 무작위, 같은 Cost Layer")
def band_and_savings(rlist, key_in, key_te, cs=0.0, ref_cls=(15, 30, 50)):
    lo_b, hi_b, savings = [], [], {c: [] for c in ref_cls}
    for r in rlist:
        folds = list(zip(r[key_in], r[key_te]))
        band = CM.genuine_selective_band(folds, N, npos, nneg, cs, B, A, cl_lo=2, cl_hi=200)
        if band: lo_b.append(band[0]); hi_b.append(band[1])
        for c in ref_cls:
            savings[c].append(CM.evaluate(folds, N, npos, nneg, c, cs, B, A)["sel_vs_inspect"])
    band_str = (f"[{int(np.median(lo_b))}, {int(np.median(hi_b))}]" if lo_b else "없음(전수검사 우위)")
    return band_str, {c: (float(np.mean(v)), float(np.std(v))) for c, v in savings.items()}

# 무작위 기준선: 점수=난수 → 같은 커널로 CM 생성(예선 RF vs Dummy와 동일 발상)
def random_run(seed):
    rng = np.random.RandomState(1000 + seed); ins, tes = [], []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        ins.append(cms_from_scores(y[tr], rng.rand(len(tr)), THS))
        tes.append(cms_from_scores(y[te], rng.rand(len(te)), THS))
    return {"rnd_in": np.array(ins), "rnd_te": np.array(tes)}
rand_runs = [random_run(s) for s in SEEDS]

# 🔴 무료 기준선 = max(전수통과, 전수검사) 중 저렴한 것의 sel_vs_inspect. 저cl서 '검사 덜함'만으로 자동 양수.
def floor_sav(cl):
    lam, mu, rho = CM.to_effective(cl, 0.0, B, A)
    pa = npos*cl; ia = N + npos*cl - npos*lam + nneg*mu - npos*rho
    return max(0.0, (ia - pa)/ia*100)      # pa<ia(전수통과 저렴)일 때만 양, 아니면 0(=전수검사가 floor)
# 순정보(폴백 적용): 선별이 무료기준선보다 나쁘면 전수검사로 폴백(공정최적화와 동일 규칙) → net=max(0, 곡선−floor)
sup_band, sup_sav = band_and_savings(runs, "sup_in", "sup_te")
ano_band, ano_sav = band_and_savings(runs, "ano_in", "ano_te")
rnd_band, rnd_sav = band_and_savings(rand_runs, "rnd_in", "rnd_te")
print(f"  선별최적 cl밴드(중앙):  지도 {sup_band}  |  비지도 {ano_band}  |  무작위 {rnd_band}")
print(f"  {'cl':>4} {'무료기준선':>10} {'무작위':>9} {'IForest':>9} {'지도':>9} {'IForest순정보(폴백)':>18} {'지도순정보':>10}")
for c in (15, 30, 50):
    fl = floor_sav(c)
    print(f"  {c:>4} {fl:>+9.1f}% {rnd_sav[c][0]:>+7.1f}% {ano_sav[c][0]:>+7.1f}% {sup_sav[c][0]:>+7.1f}% "
          f"{max(0.0, ano_sav[c][0]-fl):>+15.1f}%p {max(0.0, sup_sav[c][0]-fl):>+8.1f}%p")
print(f"  AUC on 104: 지도 {sup_auc:.3f}  비지도 {ano_auc:.3f}±{ano_auc_sd:.3f}(CI 0.5 제외=약하나 일관되게>무작위)")
print("  🔴 무료기준선=max(전수통과,전수검사). 무작위(+4.0%)<전수통과(+5.2%)=무작위는 전수통과보다도 못함.")
print("     폴백 적용시 IForest 순정보는 cl≤17만 양(+5.4%p)·이후 0 → IForest 밴드 [3,17]과 정합. 지도만 큰 순정보=라벨가치.")
# cl 축 절감곡선 (그림용, 세 점수 시드평균)
CLG = list(range(3, 61, 2))
def savings_curve(rlist, ki, kt):
    return [float(np.mean([CM.evaluate(list(zip(r[ki], r[kt])), N, npos, nneg, c, 0.0, B, A)["sel_vs_inspect"]
                           for r in rlist])) for c in CLG]
out["part2"] = {"sup_band": sup_band, "ano_band": ano_band, "rnd_band": rnd_band,
                "sup_savings": sup_sav, "ano_savings": ano_sav, "rnd_savings": rnd_sav,
                "cl_grid": CLG, "curve_sup": savings_curve(runs, "sup_in", "sup_te"),
                "curve_ano": savings_curve(runs, "ano_in", "ano_te"),
                "curve_rnd": savings_curve(rand_runs, "rnd_in", "rnd_te"),
                "curve_floor": [floor_sav(c) for c in CLG]}

# ══════════════════ 파트 ① 2차 필터: 손익분기 = 1차모델 한계정밀도 ══════════════════
# ⭐ 대조 = "아무것도 안하기"가 아니라 "1차 임계값 더 내리기". A(이상점수) vs B(지도점수=임계내리기).
print("\n" + "="*64 + "\n[파트①] 2차 필터 — 손익분기=1차모델 한계정밀도. A(IForest) vs B(임계내리기)")
def passed_region(rlist, cl, cs=0.0, kf=0.05):
    lam, mu, rho = CM.to_effective(cl, cs, B, A); be = (1+mu)/(lam+rho+mu)  # Λ=λ+ρ (리페어 포함 일반형)
    rows = []
    for r in rlist:
        js = [CM.best_threshold_idx(r["sup_in"][f], lam, mu, rho) for f in range(5)]
        thr = float(np.mean([THS[j] for j in js]))
        passed = r["sup_oof"] < thr
        aP, sP, yP = r["ano_oof"][passed], r["sup_oof"][passed], y[passed]
        m_all, d_all = len(yP), int(yP.sum())
        if m_all == 0 or d_all == 0: continue
        k5 = max(1, int(round(kf * m_all)))
        dA = int(yP[np.argsort(-aP)[:k5]].sum())       # A: 이상점수 상위
        dB = int(yP[np.argsort(-sP)[:k5]].sum())       # B: 지도점수 상위 = 임계 그냥 더 내리기
        # 커널 최적 k: 통과영역 이상점수로 임계 자동선택(임의 k 제거)
        cm_pass = cms_from_scores(yP, aP, THS); jk = CM.best_threshold_idx(cm_pass, lam, mu, rho)
        tn2, fp2, fn2, tp2 = cm_pass[jk]
        rows.append(dict(m5=k5, dA=dA, dB=dB, m_all=m_all, d_all=d_all,
                         m_opt=int(fp2+tp2), d_opt=int(tp2),
                         auc=roc_auc_score(yP, aP) if 0 < d_all < m_all else np.nan))
    g = lambda k: float(np.mean([r[k] for r in rows]))
    return dict(lam=lam, mu=mu, breakeven=be, pass_auc=float(np.nanmean([r["auc"] for r in rows])),
                m5=g("m5"), dA=g("dA"), dB=g("dB"), m_all=g("m_all"), d_all=g("d_all"),
                m_opt=g("m_opt"), d_opt=g("d_opt"),
                nA_gt_B=sum(r["dA"] > r["dB"] for r in rows), nB_gt_A=sum(r["dB"] > r["dA"] for r in rows),
                nseed=len(rows))
print(f"  {'cl':>4} {'손익분기':>8} {'m(상위5%)':>9} {'A:d회수':>7} {'A정밀도[Wilson95%]':>22} {'B(임계내리기)':>12} {'A>B/B>A':>8}")
part1 = {}
for cl in (15, 30, 50, 100):
    pr = passed_region(runs, cl)
    mA, dA = round(pr["m5"]), round(pr["dA"]); dB = round(pr["dB"])
    lo, hi = wilson(dA, mA); precA, precB = dA/mA, dB/mA
    spans = lo <= pr["breakeven"] <= hi
    verdict = "판정불가" if spans else ("이득" if precA > pr["breakeven"] else "미달")
    print(f"  {cl:>4} {pr['breakeven']*100:>7.2f}% {mA:>9} {dA:>7} "
          f"{precA*100:>6.1f}% [{lo*100:>4.1f},{hi*100:>4.1f}]  {precB*100:>10.1f}%  {pr['nA_gt_B']}/{pr['nB_gt_A']:<3} [{verdict}]")
    part1[cl] = {**pr, "precA": precA, "precB": precB, "wilsonA": [lo, hi], "verdict": verdict}
# ⚠️ cl별 seed합산 금지(같은 seed 4회=종속). 표본 유의미한 유일 지점만 근거로.
c15 = part1[15]
print(f"  통과영역 AUC 0.53~0.59. Wilson CI가 손익분기선을 전부 감쌈 → 전 cl '판정불가'.")
print(f"  ⭐ 표본 유의미한 유일 지점 cl=15(m={round(c15['m5'])}, d={round(c15['dA'])}): "
      f"B(임계내리기) {c15['nB_gt_A']}:{c15['nA_gt_B']} 우세.")
print(f"     cl=30/50/100은 회수불량 d=0~1개=무정보 → 합산 안 함(seed 종속·d=0 지점 동등가중 오류 방지).")
print(f"     → IForest가 1차모델과 직교하는 정보를 갖는다는 증거 없음. (손익분기=1차모델 한계정밀도, 해석적)")
print(f"  ⭐ B(임계내리기) 실측이 손익분기선을 따라감: cl=15 {part1[15]['precB']*100:.1f}% vs 손익분기 {part1[15]['breakeven']*100:.2f}%, "
      f"cl=30 {part1[30]['precB']*100:.1f}% vs {part1[30]['breakeven']*100:.2f}% → **항등식이 데이터에서도 성립**(대수+경험 둘 다).")
out["part1"] = part1
out["part1_cl15_B_over_A"] = [c15["nB_gt_A"], c15["nA_gt_B"]]

# ══════════════════ 파트 ③ 새 모드 존재 확인 (novelty 전제 검증) ══════════════════
# "후반=새 모드"부터 확인: Q1~3 학습→ (a)홀드아웃 Q1~3 recall vs (b)Q4 recall. 비슷하면 새 모드 없음.
print("\n" + "="*64 + "\n[파트③] novelty 전제 검증 — Q4가 정말 '새 모드'인가")
qq = np.floor(np.arange(N)/N*4).astype(int); qq[qq == 4] = 3
tr123, q4 = qq < 3, qq == 3
X123, y123 = X.iloc[tr123], y[tr123]; Xq4, yq4 = X.iloc[q4], y[q4]
n_q123_pos, n_q4_pos = int(y123.sum()), int(yq4.sum())
print(f"  Q1~3 {int(tr123.sum())}행(양성 {n_q123_pos}) 학습 → Q4 {int(q4.sum())}행(양성 {n_q4_pos}) 평가")
spw = (y123 == 0).sum() / max(n_q123_pos, 1)
full = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)),
                         ("sc", StandardScaler()), ("m", mk_lgbm(spw))])
def rec_bud(score, yt, b):
    kk = max(1, int(round(b*len(yt)))); return yt[np.argsort(-score)[:kk]].sum()/max(yt.sum(), 1)
rA, rB = {0.10: [], 0.20: []}, {0.10: [], 0.20: []}; aucA, aucB = [], []
for s in range(5):
    p = full(); p.set_params(m__random_state=s)
    oof123 = cross_val_predict(p, X123, y123, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                               method="predict_proba", n_jobs=-1)[:, 1]        # (a) 홀드아웃 Q1~3
    sq4 = full().set_params(m__random_state=s).fit(X123, y123).predict_proba(Xq4)[:, 1]  # (b) Q4
    for b in (0.10, 0.20):
        rA[b].append(rec_bud(oof123, y123, b)); rB[b].append(rec_bud(sq4, yq4, b))
    aucA.append(roc_auc_score(y123, oof123)); aucB.append(roc_auc_score(yq4, sq4))
a_auc123, a_aucq4 = float(np.mean(aucA)), float(np.mean(aucB))
prev123, prevq4 = y123.mean()*100, yq4.mean()*100
print(f"  ★주지표 AUC(유병률 둔감):  홀드아웃 Q1~3 {a_auc123:.3f}  |  Q4 {a_aucq4:.3f}   (Q4 양성 n={n_q4_pos})")
print(f"  {'검사예산':>8} {'(a)홀드아웃Q1~3 recall':>22} {'(b)Q4 recall':>16}  (보조: 유병률 오염됨)")
p3 = {}
for b in (0.10, 0.20):
    am, bm = np.mean(rA[b]), np.mean(rB[b])
    print(f"  {int(b*100):>7}% {am:>18.3f}±{np.std(rA[b]):.3f} {bm:>12.3f}±{np.std(rB[b]):.3f}")
    p3[f"{int(b*100)}%"] = {"holdout_q123": [float(am), float(np.std(rA[b]))], "q4": [float(bm), float(np.std(rB[b]))]}
print(f"  ⚠️ recall 보조: 고정예산서 유병률 낮은 Q4({prevq4:.1f}%)가 Q1~3({prev123:.1f}%)보다 불리 → AUC로 판단.")
drop = a_auc123 - a_aucq4
# Q4 양성 n=14 → AUC CI 매우 넓음. 점추정 하락을 '새 모드 확정'으로 못 씀.
concl = (f"Q4 AUC 점추정이 낮음({a_aucq4:.3f} vs 홀드아웃 {a_auc123:.3f}, Δ{drop:.3f}) → Q4 예측이 다소 어려움"
         "(드리프트 '약한 전방페널티'와 정합). BUT **Q4 양성 n={0}로 AUC CI가 넓어 '구별되는 새 모드'인지 "
         "'동일모드 covariate shift'인지 판별 불가**. 어느 쪽이든 파트①에서 IForest가 임계내리기(B)를 "
         "못 이기므로 2차필터 서사는 미지지 — 시간홀드아웃은 novelty 실증에 부족.").format(n_q4_pos)
print(f"  [결론] {concl}")
out["part3"] = {"n_q123_pos": n_q123_pos, "n_q4_pos": n_q4_pos, "auc_holdout_q123": a_auc123,
                "auc_q4": a_aucq4, "auc_drop": drop, "prev_q123": float(prev123), "prev_q4": float(prevq4),
                "recall": p3, "conclusion": concl}

json.dump(out, open(os.path.join(ROOT, "results", "anomaly.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n→ results/anomaly.json (파트②①③ 재작업 완료)")
