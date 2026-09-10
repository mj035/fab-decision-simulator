"""
B-1 v2 검증 — 재설계 모델의 닫힌해·매핑·민감도·리페어 검증 (멘토 4차 요청 출력 #9).
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
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
import cost_model as CM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); COMP = os.path.dirname(ROOT)
THS = np.linspace(0.005, 0.995, 300)   # 그리드 300 (리뷰어 권장 200~500)
BETA, ALPHA = 0.95, 0.02

def mkpipe(spw):
    kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=10.0,
              scale_pos_weight=spw, random_state=0, n_jobs=-1, verbose=-1)
    return Pipeline([("imp",SimpleImputer(strategy="median")),("var",VarianceThreshold(0.0)),
                     ("sc",StandardScaler()),("m",LGBMClassifier(**kw))])

def nested_cms(X,y,seed=7):
    spw=(y==0).sum()/max(y.sum(),1); folds=[]; oof=np.zeros(len(y))
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
        Xtr,ytr,Xte,yte=X.iloc[tr],y[tr],X.iloc[te],y[te]
        inner=cross_val_predict(mkpipe(spw),Xtr,ytr,cv=StratifiedKFold(4,shuffle=True,random_state=k),
                                method="predict_proba",n_jobs=-1)[:,1]
        te_score=mkpipe(spw).fit(Xtr,ytr).predict_proba(Xte)[:,1]; oof[te]=te_score
        in_cm=np.array([confusion_matrix(ytr,(inner>=t),labels=[0,1]).ravel() for t in THS])
        te_cm=np.array([confusion_matrix(yte,(te_score>=t),labels=[0,1]).ravel() for t in THS])
        folds.append((in_cm,te_cm))
    return folds, roc_auc_score(y,oof)

def load(name):
    if name=="SECOM":
        return adapt_from_csv(os.path.join(ROOT, "data", "fab_process_yield.csv"),
                              "Pass/Fail",1,drop_cols=["Time"])
    return adapt_from_csv(os.path.join(ROOT,"data_aps","aps_failure_training_set.csv"),
                          "class","pos",skip_comment_prefix="class")

def genuine_band(folds,N,npos,nneg,cs,gamma=0.0,V=0.0,cl_max=1200,max_rate=0.9):
    """첫 연속 '진짜 선별' 구간 [lo,hi]. 퇴화(검사율≥90%=사실상 전수검사) 제외 — 리뷰어 5(b) 가드."""
    lo=hi=None; started=False
    for cl in range(2,cl_max):
        r=CM.evaluate(folds,N,npos,nneg,cl,cs,BETA,ALPHA,gamma,V)
        tn,fp,fn,tp=r["counts"]; rate=(tp+fp)/N
        genuine=(r["strategy"]==2 and rate<max_rate)
        if genuine:
            if not started: lo=cl; started=True
            hi=cl
        elif started:
            break
    return lo,hi
def upper_cl_empirical(folds,N,npos,nneg,cs,gamma=0.0,V=0.0,cl_max=1200):
    return genuine_band(folds,N,npos,nneg,cs,gamma,V,cl_max)[1]

rep={}; t0=time.time()
for name in ["SECOM","APS"]:
    X,y,info=load(name); N=len(y); npos=int(y.sum()); nneg=N-npos
    folds,auc=nested_cms(X,y)
    print(f"\n{'#'*72}\n# {name}  N={N} 양성={npos} AUC={auc:.3f}   [{time.time()-t0:.0f}s]\n{'#'*72}")
    d={"N":N,"npos":npos,"auc":round(float(auc),4)}

    # (A) 회귀 sanity + savings 음수 노출
    print("[A] 지점별 판정 (cs=20, β=.95, ρ=0):")
    for cl in ([50,200] if name=="APS" else [20,100,300]):
        r=CM.evaluate(folds,N,npos,nneg,cl,20,BETA,ALPHA)
        print(f"    cl={cl:>4} → {CM.STRAT_NAME[r['strategy']]:6} | R_eff={r['R_eff']:5.1f} | "
              f"선별 vs전수검사 {r['sel_vs_inspect']:+6.1f}% | 선별 vs최적baseline {r['sel_vs_floor']:+6.1f}%")

    # (B) 닫힌해 vs 그리드 대조검증 (cs=20, 퇴화 가드 적용)
    lo,hi=genuine_band(folds,N,npos,nneg,20)
    r_hi=CM.evaluate(folds,N,npos,nneg,hi,20,BETA,ALPHA)
    tn,fp,fn,tp=r_hi["counts"]; mu=ALPHA*20
    lo_cf,up_cf=CM.band_closed_form(tn,fp,fn,tp,mu,0.0)
    cl_up_cf=up_cf/BETA
    ok = np.isfinite(cl_up_cf) and abs(cl_up_cf-hi)<max(5,0.15*hi)
    print(f"[B] 진짜 선별구간(cs=20): 그리드 [lo≈{lo}, hi≈{hi}] | 닫힌해 상한 cl≈{cl_up_cf:.0f}  "
          f"({'일치 ✓' if ok else '차이(암묵경계)'})")
    d["closed_vs_grid"]={"grid_lo":lo,"grid_upper":hi,"closed_upper":round(float(cl_up_cf),1)}

    # (C) cs 민감도 — 상한이 폐기비에 따라 밀리나 (B-2 해법)
    print("[C] cs(폐기:검사)별 선별 실익 상한 cl:")
    csrow={}
    for cs in [0,10,50,500]:
        u=upper_cl_empirical(folds,N,npos,nneg,cs); csrow[cs]=u
        print(f"    cs={cs:>4} → 상한 cl ≈ {u}")
    d["cs_sensitivity"]=csrow

    # (D) 예선 14:1 매핑 (analytic, μ=0.4)
    mu=ALPHA*20; lam=14*(1+mu)+1; cl14=lam/BETA
    print(f"[D] 예선 R=14 매핑: μ={mu} → λ={lam:.1f} → cl(유출:검사)≈{cl14:.1f}")
    d["preselection_R14_to_cl"]=round(float(cl14),1)

    # (E) 산업 좌표(고유출) — B-2 해결 여부
    print("[E] 실제 산업좌표(고유출) 판정 (cs=20):")
    for cl in [200,500,1000]:
        r=CM.evaluate(folds,N,npos,nneg,cl,20,BETA,ALPHA)
        print(f"    cl={cl:>5} → {CM.STRAT_NAME[r['strategy']]}")

    # (F) 리페어 항 효과 (디스플레이): γ=0.5, V=10 → 상한 얼마나 밀리나
    hi0=upper_cl_empirical(folds,N,npos,nneg,20,0.0,0.0)
    hiR=upper_cl_empirical(folds,N,npos,nneg,20,0.5,10.0)
    print(f"[F] 리페어(γ=.5,V=10) 실익상한: {hi0} → {hiR}  (Δ{ (hiR-hi0) if (hiR and hi0) else '—'})  ρ={BETA*0.5*10:.2f}")
    d["repair_upper"]={"no_repair":hi0,"repair":hiR}

    # (G) λ vs μ 민감도 — 어느 쪽이 결정을 더 흔드나 ("5배" 후속)
    base=CM.evaluate(folds,N,npos,nneg,20,20,BETA,ALPHA)["sel_vs_floor"]
    up_l=CM.evaluate(folds,N,npos,nneg,24,20,BETA,ALPHA)["sel_vs_floor"]   # cl +20% → λ +20%
    up_m=CM.evaluate(folds,N,npos,nneg,20,24,BETA,ALPHA)["sel_vs_floor"]   # cs +20% → μ +20%
    print(f"[G] 민감도(±20%): base {base:+.1f}% | λ+20%→{up_l:+.1f}%(Δ{up_l-base:+.1f}) | "
          f"μ+20%→{up_m:+.1f}%(Δ{up_m-base:+.1f})  → {'λ가 더 지배' if abs(up_l-base)>abs(up_m-base) else 'μ가 더 지배'}")
    d["sensitivity_lam_vs_mu"]={"base":round(base,2),"lam+20":round(up_l,2),"mu+20":round(up_m,2)}
    rep[name]=d

json.dump(rep,open(os.path.join(ROOT,"results","cost_model_v2.json"),"w",encoding="utf-8"),
          ensure_ascii=False,indent=2)
print(f"\n→ results/cost_model_v2.json  (총 {time.time()-t0:.0f}s)")
