"""nested CV 혼동행렬을 한 번 계산해 디스크 캐시 (이후 비용모델 반복실험은 이걸 로드)."""
import os, sys, time
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_adapter import adapt_from_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
THS = np.linspace(0.005, 0.995, 300)

def mkpipe(spw):
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=spw, random_state=0, n_jobs=-1, verbose=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median")),("var",VarianceThreshold(0.0)),
                     ("sc",StandardScaler()),("m",LGBMClassifier(**kw))])

def nested_cms(X,y,seed=7):
    spw=(y==0).sum()/max(y.sum(),1); ins=[]; tes=[]; oof=np.zeros(len(y))
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
        Xtr,ytr,Xte,yte=X.iloc[tr],y[tr],X.iloc[te],y[te]
        inner=cross_val_predict(mkpipe(spw),Xtr,ytr,cv=StratifiedKFold(4,shuffle=True,random_state=k),
                                method="predict_proba",n_jobs=-1)[:,1]
        te_score=mkpipe(spw).fit(Xtr,ytr).predict_proba(Xte)[:,1]; oof[te]=te_score
        ins.append(np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS]))
        tes.append(np.array([confusion_matrix(yte,(te_score>=t),labels=[0,1]).ravel() for t in THS]))
    return np.array(ins), np.array(tes), roc_auc_score(y,oof)

def load(name):
    if name=="SECOM":
        return adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                              "Pass/Fail",1,drop_cols=["Time"])
    return adapt_from_csv(os.path.join(ROOT,"data_aps","aps_failure_training_set.csv"),
                          "class","pos",skip_comment_prefix="class")

t0=time.time()
os.makedirs(os.path.join(ROOT,"results","cms"), exist_ok=True)
for name in ["SECOM","APS"]:
    X,y,info=load(name); N=len(y); npos=int(y.sum())
    ins,tes,auc=nested_cms(X,y)
    np.savez(os.path.join(ROOT,"results","cms",f"{name}.npz"),
             in_cms=ins, te_cms=tes, N=N, npos=npos, nneg=N-npos, auc=auc)
    print(f"{name}: N={N} 양성={npos} AUC={auc:.3f} 저장 [{time.time()-t0:.0f}s]")
print(f"→ results/cms/*.npz  (총 {time.time()-t0:.0f}s)  THS={len(THS)}")
