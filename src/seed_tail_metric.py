"""
⭐ 리뷰어 실험 3(a) — 경계를 결정하는 건 AUC인가 ROC 고재현율 꼬리인가.
경계 공식(μ=0): λ_max=(tn+fn)/fn 는 고재현율 임계값에서 평가됨 → AUC(전체면적) 아니라 TPR→1 국소모양이 결정.
가설: corr(고재현율특이도, 경계) ≫ corr(AUC, 경계)=0.24  이면 "경계=ROC꼬리" SECOM 내부 실증.

부수: per-seed nested CM 을 results/cms_seeds/ 에 캐싱 → 이후 regret·brute-force·DEGEN 재학습 불필요.
(cache_cms import 시 모듈 본문이 SECOM+APS 재계산을 트리거하므로, 모델 코드는 여기 복제.)
"""
import os, sys, time, json
import numpy as np, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False; plt.rcParams["figure.dpi"]=140
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
from preprocess_adapter import adapt_from_csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
B, A = 0.95, 0.02
THS = np.linspace(0.005, 0.995, 300)
N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 50
CACHE = os.path.join(ROOT, "results", "cms_seeds"); os.makedirs(CACHE, exist_ok=True)

def mkpipe(spw):
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0, scale_pos_weight=spw,
              random_state=0, n_jobs=-1, verbose=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median")),("var",VarianceThreshold(0.0)),
                     ("sc",StandardScaler()),("m",LGBMClassifier(**kw))])

def nested_cms(X, y, seed):
    spw=(y==0).sum()/max(y.sum(),1); ins=[]; tes=[]; oof=np.zeros(len(y))
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
        Xtr,ytr,Xte,yte=X.iloc[tr],y[tr],X.iloc[te],y[te]
        inner=cross_val_predict(mkpipe(spw),Xtr,ytr,cv=StratifiedKFold(4,shuffle=True,random_state=k),
                                method="predict_proba",n_jobs=-1)[:,1]
        te_score=mkpipe(spw).fit(Xtr,ytr).predict_proba(Xte)[:,1]; oof[te]=te_score
        ins.append(np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS]))
        tes.append(np.array([confusion_matrix(yte,(te_score>=t),labels=[0,1]).ravel() for t in THS]))
    return np.array(ins), np.array(tes), roc_auc_score(y,oof)

X, y, info = adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                           "Pass/Fail", 1, drop_cols=["Time"])
N=len(y); npos=int(y.sum()); nneg=N-npos
print(f"SECOM N={N} 양성={npos} 유병률 {npos/N*100:.2f}%  seed {N_SEED}개 (모델 rs=0 고정)")

def tail_metrics(te_cms):
    """fold합 CM에서 고재현율 꼬리 지표."""
    agg = te_cms.sum(axis=0)                       # (300,4)=(tn,fp,fn,tp)
    tn,fp,fn,tp = agg[:,0],agg[:,1],agg[:,2],agg[:,3]
    tpr = tp/npos; tnr = tn/nneg
    m95 = tpr>=0.95
    tnr95 = float(tnr[m95].max()) if m95.any() else 0.0     # 재현율95% 유지하며 최대 특이도
    m90 = tpr>=0.90
    tnr90 = float(tnr[m90].max()) if m90.any() else 0.0
    # pAUC over TPR∈[0.9,1]: 특이도(1-fpr)를 tpr에 대해 평균 (고재현율 부분면적, 0~1 정규화)
    band = (tpr>=0.90)
    if band.sum()>=2:
        o=np.argsort(tpr[band]); t_=tpr[band][o]; s_=tnr[band][o]
        pauc = float(np.trapezoid(s_, t_)/(t_.max()-t_.min())) if t_.max()>t_.min() else s_.mean()
    else:
        pauc = tnr90
    return tnr95, tnr90, pauc

def econ_boundary(folds):
    """경제 교차 = 닫힌해 고정점 g(cl)=β·cl−upper 첫 부호전환(neg→pos, fn0 이전)."""
    cls=np.arange(2,1500); g=np.empty(len(cls)); fn_=np.empty(len(cls))
    for i,cl in enumerate(cls):
        lam,mu,rho=CM.to_effective(cl,0,B,A)
        _,agg,_,_=CM._accumulate(folds,lam,mu,rho,cl); tn,fp,fn,tp=agg
        _,up=CM.band_closed_form(tn,fp,fn,tp,mu,rho); g[i]=B*cl-up; fn_[i]=fn
    f0=np.argmax(fn_==0) if (fn_==0).any() else len(cls)
    below=np.arange(len(cls))<f0
    gp=np.where((g[:-1]<0)&(g[1:]>0)&below[1:])[0]
    return int(cls[gp[0]+1]) if len(gp) else None

rows=[]; t0=time.time()
for s in range(N_SEED):
    fpath=os.path.join(CACHE,f"SECOM_s{s}.npz")
    ins,tes,auc = nested_cms(X,y,s)
    np.savez(fpath, in_cms=ins, te_cms=tes, auc=auc)     # 캐시
    folds=list(zip(ins,tes))
    tnr95,tnr90,pauc = tail_metrics(tes)
    econ = econ_boundary(folds)
    rows.append({"seed":s,"auc":round(float(auc),4),"tnr95":round(tnr95,4),"tnr90":round(tnr90,4),
                 "pauc90":round(pauc,4),"econ":econ})
    print(f"  s{s:>2} AUC {auc:.3f} | TNR@95 {tnr95:.3f} TNR@90 {tnr90:.3f} pAUC90 {pauc:.3f} | econ {econ}  [{time.time()-t0:.0f}s]")

def pear(a,b):
    a=np.array(a,float); b=np.array(b,float); m=~np.isnan(a)&~np.isnan(b)
    return float(np.corrcoef(a[m],b[m])[0,1])
def spear(a,b):
    a=np.array(a,float); b=np.array(b,float); m=~(np.isnan(a)|np.isnan(b))
    ra=np.argsort(np.argsort(a[m])); rb=np.argsort(np.argsort(b[m])); return float(np.corrcoef(ra,rb)[0,1])

econ=[r["econ"] for r in rows]; auc=[r["auc"] for r in rows]
keep=[i for i,e in enumerate(econ) if e is not None]
def col(k): return [rows[i][k] for i in keep]
E=col("econ")
print(f"\n{'='*66}\n경계(econ) 대비 예측력 (교차없음 {len(rows)-len(keep)}개 제외, n={len(keep)}):")
res={}
for k,label in [("auc","AUC(전체)"),("tnr95","TNR@재현율95"),("tnr90","TNR@재현율90"),("pauc90","pAUC[0.9,1]")]:
    p,sp=pear(col(k),E),spear(col(k),E)
    res[k]={"pearson":round(p,3),"spearman":round(sp,3)}
    print(f"  {label:>16} ↔ 경계 : Pearson {p:+.3f} | Spearman {sp:+.3f}")

# 산점도: 최강 예측변수 vs 경계
best_key=max([("tnr95",),("tnr90",),("pauc90",)], key=lambda kk: abs(res[kk[0]]["pearson"]))[0]
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
ax[0].scatter(col("auc"),E,s=30,color="#bbb"); ax[0].set_xlabel("AUC(전체)"); ax[0].set_ylabel("경계 cl")
ax[0].set_title(f"AUC ↔ 경계  (r={res['auc']['pearson']:+.2f}, 노이즈-노이즈)",fontsize=9)
ax[1].scatter(col(best_key),E,s=30,color="#d94f4f")
ax[1].set_xlabel(f"{best_key} (고재현율 특이도)"); ax[1].set_ylabel("경계 cl")
ax[1].set_title(f"{best_key} ↔ 경계  (r={res[best_key]['pearson']:+.2f}, 꼬리모양-경계)",fontsize=9)
fig.suptitle("경계를 결정하는 건 AUC가 아니라 ROC 고재현율 꼬리 — SECOM 50seed 내부 검정",fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures","fig_tail_vs_boundary.png")); plt.close()

json.dump({"n_seed":N_SEED,"n_used":len(keep),"prevalence":round(npos/N,4),
           "corr_to_boundary":res,"rows":rows},
          open(os.path.join(ROOT,"results","seed_tail_metric.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
verdict = ("꼬리지표가 AUC보다 확실히 강함 → '경계=ROC꼬리' 실증" if abs(res[best_key]["pearson"])>abs(res["auc"]["pearson"])+0.2
           else "꼬리지표도 약함 → 내부 변동폭 부족이 원인(미검정 유지)")
print(f"\n판정({best_key} 최강): {verdict}")
print(f"per-seed CM 캐시 → {CACHE}  (regret/brute-force 재사용)")
print("→ results/seed_tail_metric.json, figures/fig_tail_vs_boundary.png")
