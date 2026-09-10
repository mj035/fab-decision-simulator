"""
공정최적화 마무리 (리뷰어 11차):
 1. 손익분기 = CI 기준 통일(승률컷 폐기) → 대등 cl≤15, 모델우위 cl≥16
 2. 표/그림 통계량 통일 → seed별 '차이'가 정본, 차이곡선 ± 95%CI (대응정보 보존)
 3. AUC 격차 대응 차이 + 센서 역전 횟수
 5. K-곡선 — K개 센서로 재학습, K별 OOF AUC·절감률 → "센서 몇 개면 충분한가"
"""
import os, sys, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False; plt.rcParams["figure.dpi"]=140
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP=os.path.dirname(ROOT)
B,A=0.95,0.02
X,y,info=adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                       "Pass/Fail",1,drop_cols=["Time"])
y=np.asarray(y); N=len(y); npos=int(y.sum()); nneg=N-npos
valid=[c for c in X.columns if np.nanstd(X[c].to_numpy())>0]
Xi=X[valid].fillna(X[valid].median()).to_numpy(); j59=valid.index('59')
THS=np.linspace(0.005,0.995,300)
def auc_cols(idx):
    sub=Xi[idx]; yy=y[idx]; p=yy==1; np_=p.sum(); nn_=len(idx)-np_
    return (((sub.argsort(0).argsort(0)+1)[p].sum(0))-np_*(np_+1)/2)/(np_*nn_)

# ── 3. AUC 대응 차이 ──
dauc=[]
for seed in range(50):
    m_auc=float(np.load(glob.glob(os.path.join(ROOT,'results','cms_seeds',f'SECOM_s{seed}.npz'))[0])["auc"])
    saucs=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xi,y):
        a=auc_cols(tr); j=int(np.argmax(np.maximum(a,1-a)))
        saucs.append(max(roc_auc_score(y[te],Xi[te,j]),1-roc_auc_score(y[te],Xi[te,j])))
    dauc.append(m_auc-np.mean(saucs))
dauc=np.array(dauc)
print(f"[3] AUC 대응차(모델−센서) 50seed: 중앙 {np.median(dauc):.3f} · 95%CI [{np.mean(dauc)-1.96*dauc.std(ddof=1)/np.sqrt(50):.3f},{np.mean(dauc)+1.96*dauc.std(ddof=1)/np.sqrt(50):.3f}] · 센서 역전 {int((dauc<0).sum())}/50")

# ── 5. K-곡선 ──
def lgbm(spw): return LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=7,min_child_samples=25,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.3,reg_lambda=10.0,scale_pos_weight=spw,random_state=0,n_jobs=-1,verbose=-1)
def kmodel_folds(K,seed):
    spw=nneg/max(npos,1); folds=[]; oof=np.zeros(N)
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xi,y)):
        a=auc_cols(tr); top=np.argsort(-np.maximum(a,1-a))[:K]
        # in_cm = inner CV OOF(train) 점수 (in-sample 아님 → 과적합 임계값 방지, 캐시 프로토콜 일치)
        s_tr=cross_val_predict(lgbm(spw),Xi[tr][:,top],y[tr],cv=StratifiedKFold(4,shuffle=True,random_state=k),
                               method="predict_proba",n_jobs=-1)[:,1]
        s_te=lgbm(spw).fit(Xi[tr][:,top],y[tr]).predict_proba(Xi[te][:,top])[:,1]; oof[te]=s_te
        cm=lambda s,idx:np.array([[int(((s<t)&(y[idx]==0)).sum()),int(((s>=t)&(y[idx]==0)).sum()),
                                   int(((s<t)&(y[idx]==1)).sum()),int(((s>=t)&(y[idx]==1)).sum())] for t in THS])
        folds.append((cm(s_tr,tr),cm(s_te,te)))
    return folds,roc_auc_score(y,oof)
def vsinsp(folds,cl,cs=0):
    lam,mu,rho=CM.to_effective(cl,cs,B,A); sel,agg,_,_=CM._accumulate(folds,lam,mu,rho,cl)
    ins=N+npos*cl-npos*lam+nneg*mu-npos*rho; pa=npos*cl; return (ins-min(pa,ins,sel))/ins*100

KS=[1,5,10,20,50,100,474]; NS=5; CLS=[15,20,30]
print("\n[5] K-곡선 (5 seed): K센서 재학습 → OOF AUC · best_vs_inspect")
print(f"  {'K':>4} | {'OOF AUC':>8} | " + " ".join(f"cl={c}%".rjust(7) for c in CLS))
krep={}
for K in KS:
    aus=[]; sv={c:[] for c in CLS}
    for seed in range(NS):
        f,au=kmodel_folds(K,seed); aus.append(au)
        for c in CLS: sv[c].append(vsinsp(f,c))
    krep[K]={"auc":round(float(np.mean(aus)),3),**{f"cl{c}":round(float(np.median(sv[c])),1) for c in CLS}}
    print(f"  {K:>4} | {np.mean(aus):>8.3f} | " + " ".join(f"{np.median(sv[c]):>6.1f}%" for c in CLS))

# ── K-곡선 그림 ──
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.2))
ks=list(krep); ax1.plot(ks,[krep[k]["auc"] for k in ks],"o-",color="#1f4e8c")
ax1.set_xscale("log"); ax1.set_xlabel("센서 수 K"); ax1.set_ylabel("OOF AUC"); ax1.set_title("K개 센서 → AUC",fontsize=9); ax1.grid(alpha=.3)
for c,col in zip(CLS,["#2e7d32","#d94f4f","#b8860b"]):
    ax2.plot(ks,[krep[k][f"cl{c}"] for k in ks],"o-",color=col,label=f"cl={c}")
ax2.set_xscale("log"); ax2.set_xlabel("센서 수 K"); ax2.set_ylabel("best_vs_inspect %"); ax2.set_title("K개 센서 → 절감률",fontsize=9); ax2.legend(fontsize=8); ax2.grid(alpha=.3)
fig.suptitle("K-곡선 — 센서 몇 개면 충분한가 (K=10 부근서 K=474에 근접하면 나머지 464개 불필요)",fontsize=10)
plt.tight_layout(rect=[0,0,1,.94]); plt.savefig(os.path.join(ROOT,"figures","fig_K_curve.png")); plt.close()

# ── 1·2. 차이곡선 재작성 (process_opt5.json 재사용) ──
p5=json.load(open(os.path.join(ROOT,"results","process_opt5.json"),encoding="utf-8"))["cost_cl"]
cc=sorted(int(k) for k in p5); be=next((c for c in cc if p5[str(c)]["ci"][0]>0), None)
fig,ax=plt.subplots(figsize=(8,4.6))
dm=[p5[str(c)]["abs_diff_mean"] for c in cc]; lo=[p5[str(c)]["ci"][0] for c in cc]; hi=[p5[str(c)]["ci"][1] for c in cc]
ax.axhline(0,color="#888",lw=.8); ax.fill_between(cc,lo,hi,color="#1f4e8c",alpha=.15,label="95% CI")
ax.plot(cc,dm,"o-",color="#1f4e8c",lw=1.6,label="절대비용 차이 (센서−모델)")
ax.axvline(be,color="#333",ls="--",lw=1); ax.text(be+.3,max(dm)*.85,f"손익분기 cl={be}\n(CI 하단>0)",fontsize=8)
ax.set_xlabel("cl = 유출:검사 비용비"); ax.set_ylabel("절대비용 차이 (>0=모델 우위)")
ax.set_title(f"대응설계 정본 — seed별 차이의 평균 ± 95%CI · 손익분기 cl={be}\n(승률컷 폐기, CI 기준 통일)",fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures","fig_sensor_vs_model_diff.png")); plt.close()
print(f"\n[1·2] 손익분기 CI기준 = cl {be} (승률컷 폐기). 차이곡선 그림 재작성.")

json.dump({"auc_paired":{"median":round(float(np.median(dauc)),3),"sensor_wins":int((dauc<0).sum())},
           "breakeven_ci":be,"K_curve":krep},
          open(os.path.join(ROOT,"results","process_opt6.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("→ results/process_opt6.json, figures/fig_K_curve.png, fig_sensor_vs_model_diff.png")
