"""
B-1 실행: 새 비용모델을 SECOM·APS에 적용.
 (1) 구모델(2-param) vs 신모델(β<1) 실익구간 비교 — 상단이 어떻게 움직이나
 (2) 2D 실익맵 (C_leak/C_test × C_scrap/C_test) — 실제 산업 좌표가 어디 떨어지나
 (3) β 민감도 — 검사 검출률이 실익구간 상단을 어떻게 미나
 (4) APS 공식 50:1 재확인 (신모델서도 선별검사인가)
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False; plt.rcParams["figure.dpi"]=140

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_adapter import adapt_from_csv
import cost_model as CM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = os.path.dirname(ROOT)
THS = np.linspace(0.005, 0.995, 199)

def mkpipe(spw):
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=spw, random_state=0, n_jobs=-1, verbose=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median")),("var",VarianceThreshold(0.0)),
                     ("sc",StandardScaler()),("m",LGBMClassifier(**kw))])

def nested_cms(X, y, seed=7):
    """5 outer fold. 각 fold의 (inner OOF CM 배열, outer test CM 배열) 캐시."""
    spw = (y==0).sum()/max(y.sum(),1)
    folds=[]; oof=np.zeros(len(y))
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
        Xtr,ytr,Xte,yte = X.iloc[tr],y[tr],X.iloc[te],y[te]
        inner = cross_val_predict(mkpipe(spw),Xtr,ytr,cv=StratifiedKFold(4,shuffle=True,random_state=k),
                                  method="predict_proba",n_jobs=-1)[:,1]
        te_score = mkpipe(spw).fit(Xtr,ytr).predict_proba(Xte)[:,1]; oof[te]=te_score
        in_cm = np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS])
        te_cm = np.array([confusion_matrix(yte,(te_score>=t),labels=[0,1]).ravel() for t in THS])
        folds.append((in_cm, te_cm))
    return folds, roc_auc_score(y,oof)

# ── 구모델(2-param): cost=fp*C_FP+fn*R*C_FP, 임계값도 안쪽fold서 ──
def old_model(folds, N, npos, nneg, R):
    C_FP=1.0; sel=0.0
    for in_cm,te_cm in folds:
        c = in_cm[:,1]*C_FP + in_cm[:,2]*R*C_FP
        j=int(np.argmin(c)); tn,fp,fn,tp=te_cm[j]; sel += fp*C_FP + fn*R*C_FP
    pass_all=npos*R*C_FP; insp=nneg*C_FP
    costs=[pass_all,insp,sel]; s=int(np.argmin(costs)); floor=min(pass_all,insp)
    return s, (floor-costs[s])/floor*100

def load(name):
    if name=="SECOM":
        return adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                              label_col="Pass/Fail", positive_class=1, drop_cols=["Time"])
    return adapt_from_csv(os.path.join(ROOT,"data_aps","aps_failure_training_set.csv"),
                          label_col="class", positive_class="pos", skip_comment_prefix="class")

# 신모델 기본 엔지니어링 상수
CS_DEF, BETA_DEF, ALPHA_DEF = 20.0, 0.95, 0.02
report={}
t0=time.time()

for name in ["SECOM","APS"]:
    X,y,info = load(name)
    N=len(y); npos=int(y.sum()); nneg=N-npos
    folds, auc = nested_cms(X,y)
    print(f"\n{'='*76}\n{name}  N={N} 양성={npos} AUC={auc:.3f}   [{time.time()-t0:.0f}s]\n{'='*76}")

    # (1) 구/신 실익구간 비교 (cl 스윕, cs=20 β=0.95 α=0.02)
    CLS = list(range(2,201))
    old_strat=[]; new_strat=[]; new_sav=[]
    for cl in CLS:
        so,_ = old_model(folds,N,npos,nneg,cl)
        rn = CM.evaluate(folds,N,npos,nneg,cl,CS_DEF,BETA_DEF,ALPHA_DEF)
        old_strat.append(so); new_strat.append(rn["strategy"]); new_sav.append(rn["savings"])
    def band(strats):
        idx=[CLS[i] for i,s in enumerate(strats) if s==2]
        return (min(idx),max(idx)) if idx else None
    ob,nb = band(old_strat), band(new_strat)
    print(f"  선별검사 실익구간(C_leak:C_test):  구모델 {ob}   신모델(β=.95) {nb}")

    # 공식 비용비 지점
    for cl_mark,lbl in ([(50,"APS공식 50:1")] if name=="APS" else [(20,"예시 20:1"),(100,"고유출 100:1")]):
        rn = CM.evaluate(folds,N,npos,nneg,cl_mark,CS_DEF,BETA_DEF,ALPHA_DEF)
        print(f"    C_leak:C_test={cl_mark:>4} ({lbl}) → {CM.STRAT_NAME[rn['strategy']]}  절감 {rn['savings']:.1f}%")

    # (3) β 민감도: 상단이 어떻게 미나
    print("  β 민감도 (선별 실익구간 상단):")
    beta_band={}
    for b in [1.0,0.95,0.9,0.8]:
        ss=[CM.evaluate(folds,N,npos,nneg,cl,CS_DEF,b,ALPHA_DEF)["strategy"] for cl in CLS]
        bb=band(ss); beta_band[b]=bb
        print(f"    β={b}: 실익구간 {bb}")

    # (2) 2D 실익맵 데이터
    CL_G = np.unique(np.round(np.logspace(0,3,40)).astype(int))   # 1~1000
    CS_G = np.unique(np.round(np.logspace(0,2,30)).astype(int))   # 1~100
    grid = np.zeros((len(CS_G),len(CL_G)),dtype=int)
    for i,cs in enumerate(CS_G):
        for j,cl in enumerate(CL_G):
            grid[i,j]=CM.evaluate(folds,N,npos,nneg,cl,cs,BETA_DEF,ALPHA_DEF)["strategy"]
    # 그림
    fig,ax=plt.subplots(figsize=(7.5,5.2))
    cmap=ListedColormap(["#bdbdbd","#d94f4f","#2e7d32"]); norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
    im=ax.pcolormesh(CL_G,CS_G,grid,cmap=cmap,norm=norm,shading="nearest")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("유출:검사  C_leak / C_test"); ax.set_ylabel("폐기:검사  C_scrap / C_test")
    ax.set_title(f"{name} 실익맵 (β={BETA_DEF}) — 회색 전수통과 / 빨강 전수검사 / 초록 선별검사\n"
                 f"반도체·디스플레이 실제 좌표(고유출)는 오른쪽 영역", fontsize=9.5)
    ax.axvspan(200,1000,color="k",alpha=0.05); ax.text(430,1.3,"실제\n산업구간",fontsize=8,ha="center")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#bdbdbd",label="전수통과"),Patch(color="#d94f4f",label="전수검사"),
                       Patch(color="#2e7d32",label="선별검사")],fontsize=8,loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures",f"fig_costmap_{name.lower()}.png")); plt.close()

    report[name]={"N":N,"npos":npos,"auc":round(float(auc),4),
                  "실익구간_구모델":ob,"실익구간_신모델_b95":nb,
                  "beta_band":{str(k):v for k,v in beta_band.items()},
                  "params":{"cs":CS_DEF,"beta":BETA_DEF,"alpha":ALPHA_DEF}}

json.dump(report, open(os.path.join(ROOT,"results","cost_model.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n→ results/cost_model.json, figures/fig_costmap_secom.png, fig_costmap_aps.png  (총 {time.time()-t0:.0f}s)")
