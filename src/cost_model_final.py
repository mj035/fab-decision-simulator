"""
B-1 최종 — 리뷰어 4차 요청 완결.
 #4 닫힌해 vs 그리드 대조 (cs 트렌드)  #5 구/신 매핑표  #6 (λ,μ) 실익맵+R_eff 등고선  #8 퇴화 가드
 + savings음수/스케일assert/λμ리팩터/리페어γ(이미 cost_model.py) + λμ민감도·R14매핑·산업좌표.
"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
plt.rcParams["font.family"]="Malgun Gothic"; plt.rcParams["axes.unicode_minus"]=False; plt.rcParams["figure.dpi"]=140
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BETA, ALPHA = 0.95, 0.02

def load(name):
    d=np.load(os.path.join(ROOT,"results","cms",f"{name}.npz"))
    return list(zip(d["in_cms"],d["te_cms"])), int(d["N"]), int(d["npos"]), int(d["nneg"]), float(d["auc"])

# ── 구/신 매핑표 (#5) — 데이터 무관 ──
def mapping_table():
    print("\n[#5] 구모델 ↔ 신모델 매핑표 (C_test=1 정규화)")
    print("  C_FP = 1+μ = 1+α·cs   |   C_FN = λ−1 = β·cl−1   |   R_eff = (λ−1)/(1+μ)")
    print(f"  {'cl(유출:검사)':>12} {'cs(폐기:검사)':>12} {'λ=β·cl':>8} {'μ=α·cs':>8} {'C_FP':>6} {'C_FN':>7} {'R_eff':>7}")
    rows=[]
    for cl,cs in [(14,0),(14,20),(20,20),(50,20),(100,20),(200,50)]:
        lam,mu,rho=CM.to_effective(cl,cs,BETA,ALPHA)
        cfp,cfn,reff=1+mu, lam-1, (lam-1)/(1+mu)
        print(f"  {cl:>12} {cs:>12} {lam:>8.2f} {mu:>8.2f} {cfp:>6.2f} {cfn:>7.2f} {reff:>7.1f}")
        rows.append({"cl":cl,"cs":cs,"lam":round(lam,3),"mu":round(mu,3),
                     "C_FP":round(cfp,3),"C_FN":round(cfn,3),"R_eff":round(reff,2)})
    return rows

rep={"mapping_table": mapping_table()}

for name in ["SECOM","APS"]:
    folds,N,npos,nneg,auc=load(name)
    print(f"\n{'#'*78}\n# {name}  N={N} 양성={npos} AUC={auc:.3f}\n{'#'*78}")
    d={"N":N,"npos":npos,"auc":round(auc,4)}

    # [A] 지점 판정 (검사율·유효전략 포함)
    print("[A] 지점 판정 (cs=20):")
    for cl in ([50,200] if name=="APS" else [20,50,100,300]):
        r=CM.evaluate(folds,N,npos,nneg,cl,20,BETA,ALPHA)
        print(f"    cl={cl:>4} | 원시 {CM.STRAT_NAME[r['strategy']]} → 유효 {CM.STRAT_NAME[r['eff_strategy']]}"
              f" | 검사율 {r['inspect_rate']*100:4.1f}% | R_eff {r['R_eff']:5.1f}"
              f" | 선별vs전수검사 {r['sel_vs_inspect']:+5.1f}% | vs최적 {r['sel_vs_floor']:+5.1f}%")

    # [#4/#8] 닫힌해 vs 그리드 — cs 트렌드 (퇴화 가드 적용 genuine band)
    print("[#4] 닫힌해 vs 그리드 진짜선별 상한 cl (cs별 트렌드):")
    cs_row={}
    for cs in [0,10,50,500]:
        band=CM.genuine_selective_band(folds,N,npos,nneg,cs,BETA,ALPHA)
        if band:
            lo,hi=band
            r=CM.evaluate(folds,N,npos,nneg,hi,cs,BETA,ALPHA); tn,fp,fn,tp=r["counts"]
            _,up=CM.band_closed_form(tn,fp,fn,tp,ALPHA*cs,0.0); cl_cf=up/BETA
        else:
            lo=hi=cl_cf=None
        cs_row[cs]={"grid_band":band,"closed_upper":None if cl_cf is None else round(float(cl_cf),0)}
        print(f"    cs={cs:>4} → 그리드 진짜선별 [{lo},{hi}] | 닫힌해 상한≈{None if cl_cf is None else round(cl_cf)}")
    d["closed_vs_grid_cs"]=cs_row

    # [D] 예선 R=14 매핑
    mu=ALPHA*20; cl14=(14*(1+mu)+1)/BETA
    print(f"[D] 예선 R=14 → μ={mu} → cl(유출:검사)≈{cl14:.1f}")
    d["R14_to_cl"]=round(cl14,1)

    # [E] 산업 좌표(고유출) 유효전략
    print("[E] 산업좌표(고유출) 유효전략 (cs=20):")
    for cl in [200,500,1000]:
        r=CM.evaluate(folds,N,npos,nneg,cl,20,BETA,ALPHA)
        print(f"    cl={cl:>5} → {CM.STRAT_NAME[r['eff_strategy']]} (검사율 {r['inspect_rate']*100:.1f}%)")

    # [F] 리페어 효과 (γ=.5,V=10)
    b0=CM.genuine_selective_band(folds,N,npos,nneg,20,BETA,ALPHA,0.0,0.0)
    bR=CM.genuine_selective_band(folds,N,npos,nneg,20,BETA,ALPHA,0.5,10.0)
    print(f"[F] 리페어(γ=.5,V=10,ρ={BETA*0.5*10:.2f}) 진짜선별 상한: {b0[1] if b0 else None} → {bR[1] if bR else None}")
    d["repair_band"]={"no":b0,"yes":bR}

    # [G] λ vs μ 민감도
    base=CM.evaluate(folds,N,npos,nneg,20,20,BETA,ALPHA)["sel_vs_floor"]
    upl=CM.evaluate(folds,N,npos,nneg,24,20,BETA,ALPHA)["sel_vs_floor"]
    upm=CM.evaluate(folds,N,npos,nneg,20,24,BETA,ALPHA)["sel_vs_floor"]
    print(f"[G] 민감도±20%: base {base:+.1f} | λ+20%→{upl:+.1f}(Δ{upl-base:+.1f}) | μ+20%→{upm:+.1f}(Δ{upm-base:+.1f})"
          f" → {'λ 지배' if abs(upl-base)>abs(upm-base) else 'μ 지배'}")
    d["sens"]={"base":round(base,2),"lam":round(upl,2),"mu":round(upm,2)}

    # [#6] (λ,μ) 실익맵 + R_eff 등고선
    LAM=np.logspace(np.log10(0.5),np.log10(1000),60)
    MU =np.logspace(np.log10(0.01),np.log10(3),45)
    grid=np.zeros((len(MU),len(LAM)),dtype=int)
    for i,mu_ in enumerate(MU):
        for j,lam_ in enumerate(LAM):
            grid[i,j],_=CM.decide_lmr(folds,N,npos,nneg,lam_,mu_,0.0)
    fig,ax=plt.subplots(figsize=(8,5.4))
    cmap=ListedColormap(["#bdbdbd","#d94f4f","#2e7d32"]); norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
    ax.pcolormesh(LAM,MU,grid,cmap=cmap,norm=norm,shading="nearest")
    # R_eff 등고선 (임계값 iso): λ = R_eff*(1+μ)+1
    for reff,c in [(10,"#1f4e8c"),(50,"#1f4e8c"),(100,"#1f4e8c")]:
        ax.plot(reff*(1+MU)+1, MU, "--", color=c, lw=1)
        ax.text(reff*(1+MU[len(MU)//2])+1, MU[len(MU)//2], f"R_eff={reff}", fontsize=7, color=c, rotation=90, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("λ = β·cl  (유효 유출:검사)"); ax.set_ylabel("μ = α·cs  (유효 오폐기:검사)")
    ax.set_title(f"{name} (λ,μ) 실익맵 — 회색 전수통과 / 빨강 전수검사 / 초록 진짜선별\n"
                 f"(퇴화선별→전수검사 병합, β·α 스윕 불필요) · 점선=R_eff 등고선(임계값 iso)", fontsize=9)
    ax.legend(handles=[Patch(color="#bdbdbd",label="전수통과"),Patch(color="#d94f4f",label="전수검사(+퇴화선별)"),
                       Patch(color="#2e7d32",label="진짜 선별검사")],fontsize=8,loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(ROOT,"figures",f"fig_costmap_lm_{name.lower()}.png")); plt.close()
    rep[name]=d

json.dump(rep,open(os.path.join(ROOT,"results","cost_model_final.json"),"w",encoding="utf-8"),
          ensure_ascii=False,indent=2)
print("\n→ results/cost_model_final.json, figures/fig_costmap_lm_secom.png, fig_costmap_lm_aps.png")
