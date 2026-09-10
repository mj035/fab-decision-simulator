"""
APS 일반화 검증 (1차) — SECOM과 '동일한' 규격독립 파이프라인에 APS를 태운다.
  전처리: 중앙값 대치 → VarianceThreshold(상수제거) → 표준화   (열 이름/개수 무관)
  모델  : LightGBM (배포 모델)  ·  scale_pos_weight = N_neg/N_pos
  CostLayer: Nested CV(안쪽 fold에서만 임계값 선택) → 낙관편향 제거
  공식 비용비: FN(고장 놓침)=500, FP(불필요 점검)=10  ->  50:1
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_aps")

X = pd.read_csv(os.path.join(DATA, "aps_X.csv"))
y = pd.read_csv(os.path.join(DATA, "aps_y.csv"))["fail"].to_numpy()
N_pos, N_neg = int(y.sum()), int((y == 0).sum())
SPW = N_neg / N_pos
print(f"APS train: {X.shape[0]:,}행 × {X.shape[1]}피처 | 양성 {N_pos:,}({y.mean()*100:.2f}%) | spw={SPW:.1f}")

# ── SECOM과 동일한 LGBM 설정(배포 모델) ──
LGB_KW = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=SPW, random_state=0, n_jobs=-1, verbose=-1)

# ── SECOM과 동일한 규격독립 전처리 파이프라인 ──
def mkpipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("var", VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("m",   LGBMClassifier(**LGB_KW))])

C_FP = 10.0                       # APS 공식: 불필요 점검 1건 = 10
RATIOS = [2, 5, 10, 20, 30, 50]   # 50:1 = APS 공식 비용비
THS = np.linspace(0.005, 0.995, 199)

def cost_of(yt, sc, th, R):
    tn, fp, fn, tp = confusion_matrix(yt, (sc >= th).astype(int), labels=[0,1]).ravel()
    return fp*C_FP + fn*R*C_FP, (tn, fp, fn, tp)

def baselines(yt, R):
    return int(yt.sum())*R*C_FP, int((yt==0).sum())*C_FP   # 무검사, 전수검사

# ── [1] 신호강도: 단순 5-fold OOF AUC ──
t0 = time.time()
print("\n[1] 신호강도(OOF AUC) 계산...")
oof = cross_val_predict(mkpipe(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba", n_jobs=-1)[:, 1]
auc = roc_auc_score(y, oof)
print(f"    APS OOF ROC-AUC = {auc:.4f}   (SECOM 천장 0.72~0.75)  [{time.time()-t0:.0f}s]")

# ── [2] Nested CV 비용 (임계값은 안쪽 fold에서만) ──
print("\n[2] Nested CV Cost Layer (낙관편향 제거)...")
outer = StratifiedKFold(5, shuffle=True, random_state=7)
acc = {R: {"cost":0.0,"tn":0,"fp":0,"fn":0,"tp":0,"ths":[]} for R in RATIOS}
for k,(tr,te) in enumerate(outer.split(X,y)):
    Xtr,ytr,Xte,yte = X.iloc[tr],y[tr],X.iloc[te],y[te]
    inner = cross_val_predict(mkpipe(), Xtr, ytr,
                cv=StratifiedKFold(4, shuffle=True, random_state=k),
                method="predict_proba", n_jobs=-1)[:,1]
    te_score = mkpipe().fit(Xtr,ytr).predict_proba(Xte)[:,1]
    in_cm = np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS])
    for R in RATIOS:
        j = int((in_cm[:,1]+in_cm[:,2]*R).argmin())      # 안쪽 fold에서만 임계값 선택
        c,(tn,fp,fn,tp) = cost_of(yte, te_score, THS[j], R)
        d=acc[R]; d["cost"]+=c; d["tn"]+=tn; d["fp"]+=fp; d["fn"]+=fn; d["tp"]+=tp; d["ths"].append(THS[j])
    print(f"    outer fold {k+1}/5 done [{time.time()-t0:.0f}s]")

print("\n" + "="*74)
print(f"{'비용비':>6} | {'전략':^10} | {'절감%':>7} | {'검출%':>6} | {'과검%':>6} | {'임계값':>6}")
print("-"*74)
res = {}
for R in RATIOS:
    d=acc[R]; bn,ba = baselines(y,R); floor=min(bn,ba)
    tp,fn,fp,tn = d["tp"],d["fn"],d["fp"],d["tn"]
    sel = d["cost"]
    strat = ["무검사(전수통과)","전수검사","선별검사"][int(np.argmin([bn,ba,sel]))]
    sav = (floor - sel)/floor*100
    res[R] = {"strategy":strat, "save_pct":round(sav,1),
              "detect_pct":round(tp/(tp+fn)*100,1) if (tp+fn) else 0.0,
              "overkill_pct":round(fp/(fp+tn)*100,1) if (fp+tn) else 0.0,
              "th_mean":round(float(np.mean(d["ths"])),3), "cost":round(sel,1)}
    mark = "  ← APS 공식비용비" if R==50 else ""
    print(f"{R:>4}:1 | {strat:^10} | {sav:>6.1f}% | "
          f"{res[R]['detect_pct']:>5.1f}% | {res[R]['overkill_pct']:>5.1f}% | {res[R]['th_mean']:>6}{mark}")
print("="*74)

# 저장
out = {"n_row":int(X.shape[0]), "n_feat":int(X.shape[1]), "n_pos":N_pos, "pos_rate":round(y.mean(),4),
       "oof_auc":round(float(auc),4), "cost_C_FP":C_FP, "official_ratio":50, "nested_cv":res}
os.makedirs(os.path.join(ROOT,"results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT,"results","aps_result.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n→ results/aps_result.json 저장  (총 {time.time()-t0:.0f}s)")
print("\n[해석 포인트]")
print(f"  · APS AUC {auc:.3f} vs SECOM 0.72~0.75 → 신호강도 대조")
print(f"  · 50:1(공식)에서 선별검사가 이기고 절감>0 이면 → '비용>모델' 명제 타 데이터 재현")
print(f"  · 양성 {N_pos:,}개(SECOM 104의 ~10배) → 다음 단계에서 고비용 seed 안정성(가설) 검증")
