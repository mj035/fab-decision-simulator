"""
APS 의사결정 신뢰도 (20-seed) — SECOM과 동일 방식으로 '선별검사가 최선일 확률' 곡선.
가설: 양성 많고(1000) 신호 강한(AUC~0.99) APS는 고비용(50:1)에서도 선별검사가 안정적으로 이긴다.
      → SECOM(양성104)의 고비용 흔들림은 데이터 한계지 알고리즘 한계 아님을 실데이터로 입증.
전략 코드: 0=무검사(전수통과) 1=전수검사 2=선별검사
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["figure.dpi"]=140

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_aps")
X = pd.read_csv(os.path.join(DATA,"aps_X.csv"))
y = pd.read_csv(os.path.join(DATA,"aps_y.csv"))["fail"].to_numpy()
N_pos, N_neg = int(y.sum()), int((y==0).sum()); SPW=N_neg/N_pos

LGB_KW = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=SPW, random_state=0, n_jobs=-1, verbose=-1)
def mk(): return Pipeline([("imp",SimpleImputer(strategy="median")),("var",VarianceThreshold(0.0)),
                           ("sc",StandardScaler()),("m",LGBMClassifier(**LGB_KW))])

C_FP=10.0; RATIOS=list(range(2,61)); THS=np.linspace(0.005,0.995,199); SEEDS=list(range(20))
t0=time.time()
win = {s: [] for s in SEEDS}   # seed -> ratio별 승리전략

for s in SEEDS:
    outer = StratifiedKFold(5, shuffle=True, random_state=100+s)
    # 누적 혼동행렬(선별검사용) — 임계값은 안쪽 fold에서만 선택
    acc = {R:{"fp":0,"fn":0} for R in RATIOS}
    for k,(tr,te) in enumerate(outer.split(X,y)):
        Xtr,ytr,Xte,yte = X.iloc[tr],y[tr],X.iloc[te],y[te]
        inner = cross_val_predict(mk(),Xtr,ytr,cv=StratifiedKFold(4,shuffle=True,random_state=k+s),
                                  method="predict_proba",n_jobs=-1)[:,1]
        te_score = mk().fit(Xtr,ytr).predict_proba(Xte)[:,1]
        in_cm = np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS])
        te_cm = np.array([confusion_matrix(yte,(te_score>=t),labels=[0,1]).ravel() for t in THS])
        for R in RATIOS:
            j=int((in_cm[:,1]+in_cm[:,2]*R).argmin())
            tn,fp,fn,tp = te_cm[j]; acc[R]["fp"]+=fp; acc[R]["fn"]+=fn
    for R in RATIOS:
        sel = acc[R]["fp"]*C_FP + acc[R]["fn"]*R*C_FP
        no_insp = N_pos*R*C_FP; full = N_neg*C_FP
        win[s].append(int(np.argmin([no_insp, full, sel])))
    print(f"  seed {s+1}/20 done [{time.time()-t0:.0f}s]")

# 선별검사(=2) 승률 per ratio
aps_selrate = [np.mean([win[s][i]==2 for s in SEEDS])*100 for i in range(len(RATIOS))]

# ── SECOM 비교치 로드 (동일 방식, LGBM) ──
sec = json.load(open(os.path.join(ROOT,"results","nested_multiseed.json"),encoding="utf-8"))
sec_ratios = sec["ratios"]; sec_lgbm = sec["lgbm"]
sec_selrate = []
for i,R in enumerate(sec_ratios):
    sec_selrate.append(np.mean([sec_lgbm[str(s)][i]==2 for s in range(20)])*100)

out = {"seeds":SEEDS,"ratios":RATIOS,"aps_select_winrate":[round(v,1) for v in aps_selrate],
       "n_pos":N_pos,"note":"strategy 0=무검사 1=전수검사 2=선별검사; winrate=선별(2) 비율"}
json.dump(out, open(os.path.join(ROOT,"results","aps_multiseed.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)

# ── 비교 그림 ──
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(sec_ratios, sec_selrate, "-o", ms=3, color="#d94f4f", label="SECOM (양성 104, AUC~0.75)")
ax.plot(RATIOS, aps_selrate, "-o", ms=3, color="#2e7d32", label="APS (양성 1,000, AUC~0.99)")
ax.axvline(50, ls="--", color="gray", lw=1); ax.text(50.5, 8, "50:1\n(APS 공식)", fontsize=8, color="gray")
ax.set_xlabel("미검:과검 비용비 (R:1)"); ax.set_ylabel("선별검사가 최선인 seed 비율 (%)")
ax.set_title("의사결정 신뢰도: 고비용 구간 안정성 — APS vs SECOM\n"
             "양성 많고 신호 강하면 고비용에서도 선별검사가 안정적으로 최적", fontsize=10)
ax.set_ylim(-3,105); ax.grid(alpha=.3); ax.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures","fig_aps_confidence.png")); plt.close()

# 핵심 비교 출력
def at(ratios, rates, R):
    return rates[ratios.index(R)] if R in ratios else None
print("\n"+"="*56)
print(f"{'비용비':>6} | {'SECOM 선별승률':>14} | {'APS 선별승률':>12}")
print("-"*56)
for R in [10,20,30,40,50,60]:
    print(f"{R:>4}:1 | {at(sec_ratios,sec_selrate,R):>12.0f}% | {at(RATIOS,aps_selrate,R):>10.0f}%")
print("="*56)
print(f"→ results/aps_multiseed.json, figures/fig_aps_confidence.png  (총 {time.time()-t0:.0f}s)")
