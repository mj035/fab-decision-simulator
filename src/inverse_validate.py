"""
역산 검증·정제 (리뷰어 6차):
 1. 🔴 왕복 검증 — R → 순방향 argmin(fn·R+fp)=j* → 역산 implied_R_interval(j*) → 원래 R이 구간 복귀?
 3. 🟡 tol 제거 — 검사율을 ROC 볼록껍질 정점으로 스냅 → 그 정점의 R 구간 (+ 스냅거리)
 4. ⭐ tol-무관 사실 — 볼록껍질 정점 수(효율정책 개수) SECOM vs APS
 5. 2%→단측 R≤ / 지지비율≥50% 구간
"""
import os, sys, json, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_model as CM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_aggs():
    fs=sorted(glob.glob(os.path.join(ROOT,"results","cms_seeds","SECOM_s*.npz")),
              key=lambda p:int(p.split("_s")[-1].split(".")[0]))
    return [np.load(f)["te_cms"].sum(axis=0) for f in fs]
aggs=load_aggs(); N=int(aggs[0][0].sum()); npos=int(aggs[0][0][2]+aggs[0][0][3])
print(f"SECOM N={N} 양성={npos} · seed {len(aggs)}\n{'='*66}")

def hull_vertices(cm):
    """ROC 상단 볼록껍질 정점의 임계값 인덱스 (효율적 정책)."""
    fpr=cm[:,1]/(cm[:,0]+cm[:,1]); tpr=cm[:,3]/(cm[:,2]+cm[:,3])
    idx=sorted(range(len(cm)), key=lambda j:(fpr[j],tpr[j]))
    hull=[]
    for j in idx:
        while len(hull)>=2:
            x1,y1=fpr[hull[-2]],tpr[hull[-2]]; x2,y2=fpr[hull[-1]],tpr[hull[-1]]; x3,y3=fpr[j],tpr[j]
            if (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1) >= 0: hull.pop()   # 좌회전(오목 아님) 제거 = 상단껍질
            else: break
        hull.append(j)
    return hull

# ═══ 1. 왕복 검증 ═══
print("[1] 왕복 검증: R → j*=argmin(fn·R+fp) → implied_R_interval(j*) ∋ R ?")
R_TEST=[3,8,15,30,80]; fail=0; tot=0; margins=[]
for agg in aggs:
    fn=agg[:,2].astype(float); fp=agg[:,1].astype(float)
    for R in R_TEST:
        jstar=int(np.argmin(fn*R+fp))
        lo,hi=CM.implied_R_interval(agg,jstar)
        tot+=1
        if lo is None: fail+=1; continue
        ok = (lo-1e-9)<=R<=(hi+1e-9)
        if not ok: fail+=1
        else: margins.append(min(R-lo, (hi-R) if np.isfinite(hi) else np.inf))
print(f"    {tot}회(50seed×{len(R_TEST)}R) 중 구간이탈 {fail} → {'✅ 왕복 정합(역산 구현 검증)' if fail==0 else '❌ 매핑 오류'}")

# ═══ 3. 정점 스냅 (tol 제거) ═══
print("\n[3] 검사율 → 최근접 볼록껍질 정점 스냅 → R 구간 (tol 없음):")
for r_obs in [0.12,0.30,0.50]:
    snapped=[]; snapdist=[]; los=[]; his=[]
    for agg in aggs:
        hv=hull_vertices(agg); rates=[(agg[j,3]+agg[j,1])/N for j in hv]
        k=int(np.argmin([abs(rv-r_obs) for rv in rates])); j=hv[k]
        snapdist.append(abs(rates[k]-r_obs))
        lo,hi=CM.implied_R_interval(agg,j)
        if lo is not None: los.append(lo); his.append(hi); snapped.append(rates[k])
    his_f=np.array([h for h in his if np.isfinite(h)])
    print(f"  관측 {r_obs*100:4.0f}% → 스냅정점 중앙 {np.median(snapped)*100:.1f}% (스냅거리 중앙 {np.median(snapdist)*100:.1f}%p) "
          f"| R⁻ 중앙 {np.median(los):.1f} · R⁺ 중앙 {(np.median(his_f) if len(his_f) else np.inf):.1f} | 정의됨 {len(los)}/{len(aggs)}")

# ═══ 4. 효율정책 개수 (tol 무관) ═══
nv_secom=[len(hull_vertices(a)) for a in aggs]
aps=np.load(os.path.join(ROOT,"results","cms","APS.npz"))["te_cms"].sum(axis=0)
nv_aps=len(hull_vertices(aps)); Na=int(aps[0].sum())
print(f"\n[4] 효율정책 개수 = ROC 볼록껍질 정점 수 (tol 무관 기하 사실):")
print(f"    SECOM(AUC .75): 중앙 {int(np.median(nv_secom))}개/300 (범위 {min(nv_secom)}~{max(nv_secom)}) → 검사율 간격 ~{100/np.median(nv_secom):.1f}%p")
print(f"    APS  (AUC .99): {nv_aps}개/300 → 볼록ROC라 정점 더 많음(더 촘촘한 정책)" )

# ═══ 5. 지지비율 ≥50% 구간 + 극단 단측 ═══
print("\n[5] 지지비율(정점스냅) ≥50% R 구간:")
Rg=np.logspace(np.log10(1),np.log10(300),300)
def snap_interval(agg,r_obs):
    hv=hull_vertices(agg); rates=[(agg[j,3]+agg[j,1])/N for j in hv]
    k=int(np.argmin([abs(rv-r_obs) for rv in rates])); return CM.implied_R_interval(agg,hv[k])
rep={}
for r_obs in [0.12,0.30,0.50]:
    ivs=[snap_interval(a,r_obs) for a in aggs]
    sup=np.array([np.mean([(lo is not None and lo<=R<=(hi if np.isfinite(hi) else np.inf)) for lo,hi in ivs]) for R in Rg])
    hit=Rg[sup>=0.5]
    band=f"[{hit.min():.1f}, {hit.max():.1f}] (peak 지지 {sup.max():.0%})" if len(hit) else f"지지 50% 미달(peak {sup.max():.0%})"
    print(f"  검사율 {r_obs*100:4.0f}% : {band}")
    rep[f"{r_obs:.2f}"]={"support50_lo":float(hit.min()) if len(hit) else None,
                         "support50_hi":float(hit.max()) if len(hit) else None,"peak_support":round(float(sup.max()),2)}

json.dump({"roundtrip_fail":fail,"roundtrip_total":tot,
           "hull_vertices":{"secom_median":int(np.median(nv_secom)),"secom_range":[int(min(nv_secom)),int(max(nv_secom))],
                            "aps":nv_aps,"secom_rate_spacing_pp":round(100/np.median(nv_secom),1)},
           "support50":rep},
          open(os.path.join(ROOT,"results","inverse_validate.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("\n→ results/inverse_validate.json")
