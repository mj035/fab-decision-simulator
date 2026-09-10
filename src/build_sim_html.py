# -*- coding: utf-8 -*-
"""web/sim_final.json → web/simulator.html (서버 불필요 단일 HTML, 데이터·③그림 인라인).
리뷰어 18차 반영: FWHM폴백 · 두 경계(가드/교차, rec철회) · 무료기준선 병기 · run · Λ라벨 · β강등 · ③정적."""
import os, json, base64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
data = json.load(open(os.path.join(WEB, "sim_final.json"), encoding="utf-8"))
xai = json.load(open(os.path.join(ROOT, "results", "xai_tab.json"), encoding="utf-8"))
b64 = lambda f: base64.b64encode(open(os.path.join(ROOT, "figures", f), "rb").read()).decode()
mdl_png = b64("fig_sensor_vs_model_diff.png")   # 25차: v2(승률컷 cl=17)는 폐기 — 정본 교체
# OOF SHAP v2 (render-only 정본 그림) — 구 in-bag 그림(fig_xai_stability)은 화면에서 제외(legacy 혼입 방지)
oof1_png = b64("fig_oof_shap_summary.png"); oof2_png = b64("fig_oof_group_summary.png")
cond_png = b64("fig_process_cond_defect_v2.png"); local_png = b64("fig_shap_local.png")

# ── OOF SHAP v2 정본 수치 주입 (aggregate/groups JSON만 읽음 — 하드코딩 금지) ──
agg = json.load(open(os.path.join(ROOT, "results", "oof_shap_v2", "aggregate.json"), encoding="utf-8"))
grp = json.load(open(os.path.join(ROOT, "results", "oof_shap_v2", "groups.json"), encoding="utf-8"))
_s59 = agg["sensor_stats"]["59"]; TOP1_N = int(round(_s59["top1_rate"] * agg["n_seed"]))
N_CFG = len(grp["configs"])
S59_ALONE_ALL = all(len(c["S59_members"]) == 1 for c in grp["configs"].values())
_cfg9 = grp["configs"]["complete_linkage_combined_0.9"]
_byid = {c["cluster_id"]: c for c in _cfg9["clusters"]}
_rows = []
for _rank, _cid in enumerate(_cfg9["contrib"]["display_top30_cluster_ids"][:5], 1):
    _c = _byid[_cid]
    _nm = ("S" + _c["member_names"][0]) if _c["size"] == 1 else \
          "·".join("S" + m for m in _c["member_names"][:3]) + (f" 외 {_c['size']-3}" if _c["size"] > 3 else "")
    _typ = "단독 피처" if _c["size"] == 1 else f"센서군({_c['size']}개)"
    _rows.append(f"<tr><td><b>{_nm}</b></td><td>{_typ}</td><td>군집 기여도 {_rank}위</td>"
                 f"<td>우선 모니터링·DOE 후보</td></tr>")
CAND5 = "".join(_rows)
_j10 = agg["jaccard_seed_분포"]["10"]; J10_MED = sorted(_j10)[len(_j10)//2]

HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>비용 최적 검사정책 시뮬레이터 (본선)</title>
<style>
  :root{--bg:#f4f6f9;--card:#fff;--ink:#16202c;--ink2:#5b6875;--ink3:#8b97a4;--line:#e2e7ee;
    --blue:#4f86d9;--blue-d:#2a5da8;--red:#d94f4f;--gray:#8c8c8c;--green:#3fa06b;--amber:#c98a1b;--purple:#8a63c9;
    --shadow:0 1px 2px rgba(16,32,52,.05),0 8px 24px rgba(16,32,52,.06);}
  @media (prefers-color-scheme:dark){:root{--bg:#0f141a;--card:#171e26;--ink:#e8edf3;--ink2:#a2aebb;--ink3:#6f7d8c;
    --line:#26313d;--blue:#6fa0e8;--blue-d:#9dc0f2;--red:#e8736f;--gray:#93a0ad;--green:#5cbb87;--amber:#e0a63c;--purple:#b394e0;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1060px;margin:0 auto;padding:28px 20px 64px}
  h1{font-size:21px;font-weight:700;margin:0 0 6px;letter-spacing:-.02em}
  .sub{color:var(--ink2);font-size:14px;margin:0}
  .tag{display:inline-block;margin-top:10px;padding:4px 10px;border-radius:999px;background:color-mix(in srgb,var(--blue) 12%,transparent);color:var(--blue-d);font-size:12px;font-weight:600}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
  .tabs{display:flex;gap:6px;margin:18px 0 20px;flex-wrap:wrap}
  .tabs button{padding:10px 16px;font-size:14px;font-weight:700;cursor:pointer;font-family:inherit;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink2)}
  .tabs button[aria-selected=true]{background:color-mix(in srgb,var(--blue) 14%,transparent);border-color:var(--blue);color:var(--blue-d)}
  .tabs .num{font-size:12px;opacity:.7;margin-right:5px}
  section[role=tabpanel]{display:none} section[role=tabpanel].on{display:block}
  .grid{display:grid;grid-template-columns:340px 1fr;gap:18px;align-items:start}
  @media (max-width:840px){.grid{grid-template-columns:1fr}}
  .panel{padding:20px}
  .panel h2{font-size:12.5px;font-weight:700;color:var(--ink2);margin:0 0 16px;text-transform:uppercase;letter-spacing:.06em}
  .field{margin-bottom:15px}
  .field label{display:block;font-size:13px;font-weight:600;margin-bottom:6px}
  .field .hint{font-size:12px;color:var(--ink3);font-weight:400}
  .inputrow{display:flex;align-items:center;gap:8px}
  input[type=number]{flex:1;min-width:0;padding:9px 11px;font-size:15px;font-variant-numeric:tabular-nums;border:1px solid var(--line);border-radius:9px;background:var(--bg);color:var(--ink)}
  input[type=number]:focus{outline:2px solid var(--blue);outline-offset:-1px}
  .unit{font-size:13px;color:var(--ink3);white-space:nowrap}
  input[type=range]{width:100%;accent-color:var(--blue);margin-top:6px}
  details{margin-top:14px;border-top:1px dashed var(--line);padding-top:10px}
  details summary{font-size:12.5px;color:var(--ink2);cursor:pointer;font-weight:600}
  details .note{font-size:11.5px;color:var(--ink3);margin:8px 0 0}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .big{margin-top:16px;padding:18px;border-radius:12px;text-align:center;background:color-mix(in srgb,var(--purple) 8%,transparent);border:1px solid color-mix(in srgb,var(--purple) 24%,transparent)}
  .big.blue{background:color-mix(in srgb,var(--blue) 8%,transparent);border-color:color-mix(in srgb,var(--blue) 22%,transparent)}
  .big .lab{font-size:12px;color:var(--ink2)}
  .big .val{font-size:30px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;color:var(--purple)}
  .big.blue .val{color:var(--blue-d)}
  .verdict{padding:20px;display:flex;gap:16px;align-items:center}
  .verdict .icon{width:44px;height:44px;flex:0 0 44px;border-radius:11px;display:grid;place-items:center;font-size:20px;font-weight:800;color:#fff}
  .verdict .t1{font-size:11.5px;font-weight:700;color:var(--ink2);letter-spacing:.06em;text-transform:uppercase}
  .verdict .t2{font-size:19px;font-weight:800;letter-spacing:-.02em;margin:1px 0 3px}
  .verdict .t3{font-size:13px;color:var(--ink2)}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border-top:1px solid var(--line)}
  @media (max-width:640px){.kpis{grid-template-columns:repeat(2,1fr)}}
  .kpi{background:var(--card);padding:13px 15px}
  .kpi .k{font-size:11px;color:var(--ink3);font-weight:600}
  .kpi .v{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
  .bars{padding:18px 20px}
  .bars h3,.chart h3{font-size:12.5px;font-weight:700;color:var(--ink2);margin:0 0 14px;text-transform:uppercase;letter-spacing:.06em}
  .bar{margin-bottom:12px}
  .bar .top{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}
  .bar .nm{font-weight:600}.bar .hint{font-weight:400;color:var(--ink3);font-size:12px;margin-left:2px}
  .bar .amt{font-variant-numeric:tabular-nums;color:var(--ink2)}
  .track{height:10px;background:var(--bg);border-radius:99px;overflow:hidden}
  .fill{height:100%;border-radius:99px;transition:width .25s ease}
  .best{font-size:11px;font-weight:700;padding:1px 7px;border-radius:99px;margin-left:6px;background:color-mix(in srgb,var(--green) 16%,transparent);color:var(--green)}
  .chart{padding:18px 20px;margin-top:16px}
  svg{width:100%;height:auto;display:block;overflow:visible}
  .axis{font-size:10px;fill:var(--ink3)}
  .steps{padding:18px 20px;margin-top:16px}
  .steps table{width:100%;border-collapse:collapse}
  .steps td{padding:8px 6px;border-bottom:1px solid var(--line);font-size:13px}
  .steps td.a{color:var(--ink3);width:120px}
  .steps td.b{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;width:130px}
  .steps td.c{color:var(--ink3);font-size:12px}
  .steps tr:last-child td{border-bottom:none}
  .warn{margin-top:16px;padding:13px 16px;border-radius:11px;font-size:13.5px;background:color-mix(in srgb,var(--amber) 12%,transparent);border:1px solid color-mix(in srgb,var(--amber) 30%,transparent)}
  .warn.ok{background:color-mix(in srgb,var(--green) 10%,transparent);border-color:color-mix(in srgb,var(--green) 26%,transparent)}
  .warn.bad{background:color-mix(in srgb,var(--red) 10%,transparent);border-color:color-mix(in srgb,var(--red) 28%,transparent)}
  .warn b{color:var(--amber)}.warn.ok b{color:var(--green)}.warn.bad b{color:var(--red)}
  .bnd{margin-top:16px;padding:16px 18px}
  .bnd h3{font-size:12.5px;font-weight:700;color:var(--ink2);margin:0 0 10px;text-transform:uppercase;letter-spacing:.06em}
  .bnd .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .bnd .bx{padding:12px;border-radius:10px;background:var(--bg);text-align:center}
  .bnd .bx .n{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}
  .bnd .bx .l{font-size:12px;color:var(--ink2)}.bnd .bx .r{font-size:11px;color:var(--ink3)}
  .bnd .msg{margin-top:12px;font-size:13px;color:var(--ink);padding:10px 12px;border-radius:9px;background:color-mix(in srgb,var(--amber) 10%,transparent);border:1px solid color-mix(in srgb,var(--amber) 26%,transparent)}
  footer{margin-top:22px;font-size:12px;color:var(--ink3);line-height:1.8}
  footer code{font-size:11.5px;background:var(--card);padding:1px 5px;border-radius:4px;border:1px solid var(--line)}
  .mdlimg{padding:18px 20px}.mdlimg img{width:100%;border-radius:10px;border:1px solid var(--line)}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>비용 최적 검사정책 시뮬레이터</h1>
  <p class="sub">모델은 점수까지만 준다. 재검사 여부·비용비는 <b>비용 구조</b>가 정한다.</p>
  <span class="tag">SECOM 실측 1,567 생산 단위 · 50-seed Nested CV · 낙관편향 제거 · 합성데이터 없음</span>
  <span class="tag" style="margin-left:6px">고정된 SECOM/LightGBM 산출물과 Cost Layer 결과를 탐색하는 <b>정적 프론트엔드 프로토타입</b> — 재학습·실시간 계산 없음</span>
</header>
<div class="tabs" role="tablist">
  <button role="tab" aria-selected="true"  data-tab="xai"><span class="num">1</span>무엇을 점검할까 (XAI)</button>
  <button role="tab" aria-selected="false" data-tab="pol"><span class="num">2</span>검사정책 판단</button>
  <button role="tab" aria-selected="false" data-tab="inv"><span class="num">3</span>역산 — 비용비를 되묻다</button>
  <button role="tab" aria-selected="false" data-tab="mdl"><span class="num">4</span>모델이 필요한가</button>
</div>

<!-- ④ XAI -->
<section role="tabpanel" id="xai" class="on">
  <div class="card verdict">
    <div class="icon" style="background:var(--purple)">1</div>
    <div><div class="t1">XAI 기반 의사결정 · 운영 모델 LightGBM (champion)</div><div class="t2">SHAP이 지목한 센서 <b>하나</b>가 곧 점검 우선순위</div>
    <div class="t3">그 센서 하나가 <b>점검(안정·단독 피처 + 시간대리)</b>을 거쳐 비용 결정(모델필요 탭)까지 이어진다.</div></div>
  </div>
  <div class="card" style="padding:16px 20px;margin-top:14px">
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">
      <div class="kpi"><div class="k">운영 모델</div><div class="v" style="font-size:16px">LightGBM <span style="font-size:11px;color:var(--ink3)">champion</span></div></div>
      <div class="kpi"><div class="k">추천 전략 (live)</div><div class="v" id="kStrat" style="font-size:14px">–</div></div>
      <div class="kpi"><div class="k">예상 절감률 <span style="opacity:.7">(전수검사 대비 · 보간 조건)</span></div><div class="v" id="kSave">–</div></div>
      <div class="kpi"><div class="k">최적 임계값</div><div class="v" id="kTh">–</div></div>
    </div>
    <div style="margin-top:6px;font-size:11.5px;color:var(--ink3)">※ 절감률은 <b>랜덤 CV OOF(보간) 조건</b>값 — 전방(walk-forward) 조건에서는 구간별로 다름(검사정책 탭의 병기 참조).</div>
    <div style="margin-top:10px;font-size:13.5px;line-height:1.8">
      <b style="font-size:16px">S59</b> — OOF 예측 기여도 <b>__TOP1N__/20 seed 모두 1위</b> · 전체·양성·음성 집계 <b>모두 1위</b> ·
      시험한 <b>__NCFG__개 상관 군집 설정</b>에서 고상관 짝 미확인(단독 피처) · complete-linkage 0.90 <b>군집 기여도 1위</b>
    </div>
    <div style="margin-top:6px;font-size:12px;color:#8a4a4a">예측 기여도 결과이며 인과관계를 의미하지 않습니다. 시간·배치 대리 가능성에 대한 추가 공정 검증이 필요합니다.</div>
  </div>
  <div class="card" style="padding:16px 20px;margin-top:14px">
    <h3 style="font-size:12.5px;font-weight:700;color:var(--ink2);margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em">우선 점검 후보 Top 5 — OOF 군집 기여도 (정본 JSON에서 주입)</h3>
    <table style="width:100%;font-size:12.5px;line-height:2">__CAND5__</table>
  </div>
  <details style="margin-top:14px">
    <summary style="cursor:pointer;font-size:13px;font-weight:700;padding:10px 6px;color:var(--ink2)">분석 근거 상세 보기 (펼치기)</summary>
    <div class="card" style="padding:14px 20px;margin-top:10px">
      <div style="font-size:13px;color:var(--ink2);text-align:center;line-height:2">
        <b>SHAP 상위 (S59)</b> &nbsp;→&nbsp; seed안정·단독 피처·시간대리 점검 &nbsp;→&nbsp; 조건부 불량률 &nbsp;→&nbsp; 관리한계선 &nbsp;→&nbsp; <b>단일센서 vs 474모델</b>
      </div>
    </div>
    <div class="card mdlimg" style="margin-top:10px"><img src="data:image/png;base64,__XAIIMG__" alt="OOF SHAP 요약">
      <p class="hint" style="margin:10px 2px 0">OOF(미관측 validation 전용) SHAP — 상위 10과 Top-k 집합 안정성. 정본 <code>results/oof_shap_v2</code>에서 render-only.</p>
    </div>
    <div class="card" style="padding:14px 20px;margin-top:10px">
      <ul style="font-size:12.5px;line-height:1.9;margin:0;padding-left:18px">
        <li>20 seeds × 5-fold StratifiedKFold — <b>validation-only SHAP</b>(train 행 미포함), 전처리는 fold 내부 fit</li>
        <li>Top-1 Jaccard <b>1.000</b>(전 쌍 = S59 고정) · Top-10 Jaccard 중앙 <b>약 __J10__</b> — 개별 S59만 고정, 상위 집합은 후보군</li>
        <li>준상수 6개 피처(S74·S206·S209·S342·S347·S478)는 각 seed에서 1/5 fold 비활성(SHAP=0 처리)</li>
        <li>OOF AUC는 <b>QC 전용</b> — 성능 성과·모델 선정에 사용하지 않음</li>
      </ul>
    </div>
    <div class="card mdlimg" style="margin-top:10px"><img src="data:image/png;base64,__GRPIMG__" alt="고상관 센서군 기여도">
      <p class="hint" style="margin:10px 2px 0">고상관 센서군 기여도(complete-linkage 0.90, 상위 5) — A(순 기여)·B(총 배분)·C(내부 상쇄 지표) 구분.</p>
    </div>
    <div class="card" style="padding:14px 20px;margin-top:10px">
      <p class="hint" style="margin:0"><b>참고(학습 내 bootstrap — 구 분석, OOF 아님)</b>: 순위 중앙 1위, top-3 <b id="xT3"></b>·top-10 <b id="xT10"></b>,
      중복센서 제거(474→<b id="xNC"></b>) 후에도 1위, 클러스터 크기 <b id="xCS"></b>(고상관 짝이 확인되지 않은 단독 피처). ※ 상위 <i>집합</i>은 흔들려(Jaccard <b id="xJ"></b>) 개별 S59만 반복 관측 — <b>OOF 결과와 방향 일치</b>.</p>
    </div>
  <div class="card" style="padding:16px 20px;margin-top:14px">
    <h3 style="font-size:12.5px;font-weight:700;color:var(--ink2);margin:0 0 8px;text-transform:uppercase;letter-spacing:.06em">S59 신호의 안정성 — 3중 점검 (시간대리 가능성 병기)</h3>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;font-size:12.5px">
      <div style="padding:10px;border-radius:9px;background:var(--bg)"><b>① 시간대리 점검</b><br>행 순서 상관 ρ≈−0.44·후반(Q4) 단독 AUC 0.514 — 공정 신호와 시간·배치 대리를 <b>분리할 수 없음</b> → 원인·제어 대상 아님, 모니터링 후보로만</div>
      <div style="padding:10px;border-radius:9px;background:var(--bg)"><b>② seed 안정</b><br>top-3 85%, 중복제거 후 1위 (검증 안 된 순위 아님)</div>
      <div style="padding:10px;border-radius:9px;background:var(--bg)"><b>③ 단독 피처</b><br>고상관 짝 미확인(클러스터 크기 1) — 중복센서가 순위 나눠먹은 착시 아님. 통계·인과적 독립 의미 아님</div>
    </div>
  </div>
  <div class="card mdlimg" style="margin-top:14px"><img src="data:image/png;base64,__CONDIMG__" alt="S59 조건부 불량률">
    <p class="hint" style="margin:10px 2px 0">S59 십분위 <b>8~9에서 불량률 ~17%</b>(전체 6.6%의 2.5배). <b>관리한계선(커널 최적)</b> → cl=15에서 S59 <b>상위 <span id="xCtrl">–</span></b>를 관리 대상으로(<code>best_threshold_idx</code> 계산, 눈대중 아님 — 십분위 8~9와 일치) → 이 관리한계선이 474모델과 대등한지 <b>모델필요 탭</b>에서 확인.
    <br>※ 그림 제목의 "결측은 신호 아님"은 <b>전역 집계 기준(철회)</b> — 센서별 지시자로 재검정 시 6개 클러스터가 BH-FDR q&lt;0.10 통과, <b>결측은 신호임</b>(KEY_NUMBERS).</p>
  </div>
  <div class="card mdlimg" style="margin-top:14px"><img src="data:image/png;base64,__LOCALIMG__" alt="국소 SHAP 워터폴">
    <p class="hint" style="margin:10px 2px 0"><b>국소 설명</b> — "이 run이 왜 재검사 대상인가?"에 답. 고위험 run은 S59가 불량 쪽으로 밀고(+0.6), 저위험은 전 센서가 정상 쪽. 사전계산 정적.</p>
  </div>
  <div class="card verdict" style="margin-top:14px">
    <div class="icon" style="background:var(--green)">✓</div>
    <div><div class="t1">클로징</div><div class="t2" style="font-size:16px">XAI가 알려준 것이 곧 의사결정</div>
    <div class="t3">SHAP이 S59를 지목했고, 그 센서 하나의 관리한계선만으로 <b>474센서 모델과 대등(cl 9~15, 랜덤 CV 평균 명제)</b>. 단 S59는 시간·배치 대리 가능성을 분리할 수 없어 <b>추가 공정 검증이 필요한 모니터링 후보</b>다. → 모델필요 탭에서 확인.</div></div>
  </div>
  </details>
</section>

<!-- ② 역산 -->
<section role="tabpanel" id="inv">
  <div class="grid">
    <div class="card panel">
      <h2>검사율만 입력 → 비용비 역산</h2>
      <div class="field">
        <label>현재(또는 목표) 검사율 <span class="hint">— 재검사로 돌리는 run 비율</span></label>
        <div class="inputrow"><input type="number" id="iRate" value="12" min="3" max="65" step="1"><span class="unit">%</span></div>
        <input type="range" id="iRateS" min="3" max="65" step="1" value="12">
      </div>
      <p class="hint" id="iSnap" style="margin:6px 0 0"></p>
      <details>
        <summary>환산용 선택 입력 (λ=β·cl, μ=α·cs — 붕괴)</summary>
        <div class="row2" style="margin-top:10px">
          <div><label style="font-size:12px">β 유출스케일</label><input type="number" id="iBeta" value="0.95" min="0.1" max="1" step="0.05"></div>
          <div><label style="font-size:12px">μ=α·cs 과검</label><input type="number" id="iMu" value="0" min="0" max="2" step="0.1"></div>
        </div>
        <p class="note">β는 λ=β·cl 형태로만 등장 → cl과 구분 불가, <b>별도 추정 불필요</b>(cl→λ 환산 스케일일 뿐). 그래서 R 산출엔 β·μ가 필요 없다.</p>
      </details>
    </div>
    <div>
      <div class="card big" style="margin-top:0">
        <div class="lab">암묵 비용비 R = 유출 : 검사  <span id="iBandTag" style="font-size:11px"></span></div>
        <div class="val" id="iR">–</div>
        <div class="lab" id="iVdesc" style="margin-top:6px"></div>
      </div>
      <div class="card steps">
        <h3>3단 환산 — 위로 갈수록 가정이 적다</h3>
        <table>
          <tr><td class="a">R (암묵 비용비)</td><td class="b" id="s_R">–</td><td class="c">입력 불필요 — ROC만으로</td></tr>
          <tr><td class="a">Λ = 유효 유출비용</td><td class="b" id="s_L">–</td><td class="c">μ 필요 · Λ=R(1+μ)+1 (리페어 시 ρ 포함)</td></tr>
          <tr><td class="a">cl = 원 비용비</td><td class="b" id="s_cl">–</td><td class="c">β 필요 · cl=Λ/β (ρ=0)</td></tr>
        </table>
      </div>
      <div class="card chart">
        <h3>지지비율 곡선 — R이 최적일 확률 (50 seed)</h3>
        <svg id="iChart" viewBox="0 0 640 240" role="img" aria-label="지지비율 곡선"></svg>
        <p class="hint" style="margin:8px 0 0">가로 R(로그)·세로 이 검사율서 R이 최적정책이 되는 seed 비율. 음영=함의 R 구간(다수지지≥50% 또는 반치폭).<br>
        ※ 검사정책↔역산 왕복: R구간은 <b>seed별 지지</b> 기준, 검사정책 탭 검사율은 <b>50seed 평균</b> — 집계 방식이 달라 점추정 R=(λ−1)이 이 구간과 정확히 겹치지 않을 수 있음(방향은 일관, 역산 왕복검증 250회 통과).</p>
      </div>
    </div>
  </div>
</section>

<!-- ① 검사정책 -->
<section role="tabpanel" id="pol">
  <div class="grid">
    <div class="card panel">
      <h2>비용 구조 입력 (3-param)</h2>
      <div class="field">
        <label>유출:검사 비용비 cl <span class="hint">— 미검 1건 / 검사 1건</span></label>
        <div class="inputrow"><input type="number" id="pCl" value="15" min="2" max="100" step="1"><span class="unit">: 1</span></div>
        <input type="range" id="pClS" min="2" max="100" step="1" value="15">
      </div>
      <div class="row2">
        <div class="field"><label style="font-size:12px">β 유출스케일</label><input type="number" id="pBeta" value="0.95" min="0.1" max="1" step="0.05"></div>
        <div class="field"><label style="font-size:12px">μ=α·cs 과검</label><input type="number" id="pMu" value="0" min="0" max="2" step="0.1"></div>
      </div>
      <div class="big blue"><div class="lab">유효 파라미터 (의사결정은 이 둘로 충분)</div><div class="val" id="pLM" style="font-size:20px">–</div></div>
      <p class="hint" style="margin:10px 0 0">β는 λ=β·cl 환산 스케일 — 별도 추정 불필요.</p>
    </div>
    <div>
      <div class="card verdict">
        <div class="icon" id="pIcon">–</div>
        <div><div class="t1">권고 검사 전략 · 운영 모델 LightGBM (champion)</div><div class="t2" id="pTitle">–</div><div class="t3" id="pDesc">–</div></div>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="k">판정 임계값</div><div class="v" id="pTh">–</div></div>
        <div class="kpi"><div class="k">전수검사 대비 <span style="opacity:.75">(보간 조건)</span></div><div class="v" id="pSave">–</div></div>
        <div class="kpi"><div class="k">무료기준선 대비</div><div class="v" id="pSaveF">–</div></div>
        <div class="kpi"><div class="k">검사율</div><div class="v" id="pRate">–</div></div>
      </div>
      <div class="bars" style="border-top:1px solid var(--line)">
        <h3>세 전략의 총 기대비용 (검사단위)</h3>
        <div class="bar"><div class="top"><span class="nm">① 전수통과 <span class="hint">검사 안 함</span><span id="pb1" class="best" hidden>최선</span></span><span class="amt" id="pa1">–</span></div><div class="track"><div class="fill" id="pf1" style="background:var(--gray)"></div></div></div>
        <div class="bar"><div class="top"><span class="nm">② 전수검사 <span class="hint">전부 재검사</span><span id="pb2" class="best" hidden>최선</span></span><span class="amt" id="pa2">–</span></div><div class="track"><div class="fill" id="pf2" style="background:var(--red)"></div></div></div>
        <div class="bar"><div class="top"><span class="nm">③ 선별검사 <span class="hint">474모델</span><span id="pb3" class="best" hidden>최선</span></span><span class="amt" id="pa3">–</span></div><div class="track"><div class="fill" id="pf3" style="background:var(--blue)"></div></div></div>
      </div>
      <p class="hint" style="margin:6px 2px 0">※ 판정 임계값 = <b id="pThHint">–</b> — 모델 점수가 이 값 이상인 run을 재검사 대상으로 분류(안쪽 fold서 선택).</p>
      <div id="pWarn" class="warn"></div>
      <p class="hint" style="margin:10px 2px 0">※ 위 절감률은 <b>랜덤 CV OOF(보간) 조건</b>의 값이다. 전방(walk-forward) 조건에서는 구간별로 크게 달라진다 — 관측 <b>3.9~53.1%</b>(Q2 구간은 커널이 검사 자체를 권고하지 않음). 특정 구간의 전방 성과를 상·하한 어느 쪽으로도 보장하지 않는다.</p>
      <div class="card bnd">
        <h3>실익 상한 — 단일 숫자로 말할 수 없다</h3>
        <div class="two">
          <div class="bx"><div class="l">경제 교차</div><div class="n" id="bEcon">–</div><div class="r" id="bEconR"></div></div>
          <div class="bx"><div class="l">운영 가드(검사율90%)</div><div class="n" id="bGuard">–</div><div class="r" id="bGuardR"></div></div>
        </div>
        <div class="msg" id="bMsg"></div>
      </div>
    </div>
  </div>
</section>

<!-- ③ 모델 필요? -->
<section role="tabpanel" id="mdl">
  <div class="card verdict">
    <div class="icon" style="background:var(--green)">4</div>
    <div><div class="t1">서사의 결론</div><div class="t2">모델이 필요한 구간은 좁다 — cl≥16부터</div>
    <div class="t3">cl 9~15면 474모델 없이 <b>단일 센서 S59 관리한계선</b>으로 대등. 검사 투자가 정당화되지 않는 (유병률, 비용비) 조건을 커널이 명시한다(<b>cl=15 가정 하</b>).</div></div>
  </div>
  <div class="card mdlimg" style="margin-top:16px">
    <img src="data:image/png;base64,__MDLIMG__" alt="단일센서 vs 474모델 절감곡선">
    <p class="hint" style="margin:10px 2px 0">지표 = <b>절대비용 차이(센서 − 모델)</b>, 50 seed 대응설계 ± 95%CI. 검사정책 탭의 sel_vs_inspect와는 다른 축이므로 값 직접 비교 금지.
    AUC 격차 0.052(모델 0.731 vs 센서 0.679, 둘 다 outer). <b>손익분기 cl=16</b> — "이후 단조유지" 규칙, boot CI [14,16], 정의역 cl∈[2,64].
    <b>cl 9~15는 대등</b>이고 모델 우위는 cl≥16 구간뿐.
    <br>※ cl=8의 아래쪽 표식은 <b>센서가 유의 우세한 고립점</b>(cl 7·9는 대등 — 구간 아님, 다중비교 미보정). 동일 검사량에서 센서가 tp를 2.1건 더 적발하는 <b>국소 순위 역전</b>으로, 전역 AUC 우위가 운영점 우위를 보장하지 않음을 같은 데이터에서 보여준다.
    <br>※ 종전 그림의 세로선 <b>17</b>은 <b>철회된 승률컷(75%) 규칙</b>의 산물이었다 — 해당 그림(<code>fig_sensor_vs_model_v2</code>)은 폐기하고 본 정본으로 교체했다(25차).</p>
  </div>
</section>

<footer>
  수치는 SECOM 실측 1,567 생산 단위(불량 104, 6.64%)에 대해 <b>50-seed Nested CV</b>로 산출. 임계값은 안쪽 fold에서만 선택(낙관편향 제거).
  역산은 ROC 볼록껍질 접선(정점 스냅) 기반. 실익 상한 두 경계는 <b>seed_boundaries</b>(50 seed). 합성·증강 없음.
  단일 출처 <code>web/sim_final.json</code>.
</footer>
</div>

<script id="simdata" type="application/json">__SIMDATA__</script>
<script>
const D=JSON.parse(document.getElementById('simdata').textContent);
const M=D.meta,INV=D.inverse,POL=D.policy,RG=D.Rg;
const $=id=>document.getElementById(id);
const cvar=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
document.querySelectorAll('.tabs button').forEach(b=>{b.addEventListener('click',()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-selected',x===b));
  document.querySelectorAll('section[role=tabpanel]').forEach(s=>s.classList.toggle('on',s.id===b.dataset.tab));
  if(b.dataset.tab==='inv')renderInv(); else if(b.dataset.tab==='pol')renderPol();});});

/* ② 역산 */
function nearestInv(rate){let best=INV[0],bd=1e9;for(const it of INV){const d=Math.abs(it.rate*100-rate);if(d<bd){bd=d;best=it;}}return best;}
function renderInv(){
  const rate=+$('iRate').value,beta=Math.max(.1,+$('iBeta').value||.95),mu=Math.max(0,+$('iMu').value||0);
  const it=nearestInv(rate),maj=it.majority;
  const lo=maj?it.R50_lo:it.Rf_lo, hi=maj?it.R50_hi:it.Rf_hi;
  $('iSnap').textContent=`가장 가까운 효율 정책(볼록껍질 정점): ${(it.snap_rate*100).toFixed(1)}% (스냅 거리 ${(it.snap_dist*100).toFixed(1)}%p)`;
  if(lo==null){ $('iR').textContent='—'; $('iBandTag').textContent=''; $('iVdesc').textContent='어떤 R에서도 이 검사율이 최적이 아님(볼록껍질 내부).'; }
  else{
    $('iR').textContent=`${lo} ~ ${hi}`;
    $('iBandTag').textContent = maj ? `· 다수지지≥50% (peak ${(it.peak*100).toFixed(0)}%)` : `· ⚠ 다수 미달 (peak ${(it.peak*100).toFixed(0)}%) · 반치폭`;
    $('iVdesc').innerHTML = maj
      ? `귀사가 실제로 평가하는 유출 비용이 <b>이 범위 밖</b>이라면, 현재 검사율 ${rate}%는 최적이 아닙니다.`
      : `이 검사율은 어떤 비용비에서도 <b>다수 지지를 못 받습니다</b>(효율적이지 않음). 반치폭 구간만 표시.`;
  }
  const conv=r=>({L:r*(1+mu)+1, cl:(r*(1+mu)+1)/beta});
  if(lo!=null){const a=conv(lo),b=conv(hi);
    $('s_R').textContent=`${lo} ~ ${hi}`; $('s_L').textContent=`${a.L.toFixed(1)} ~ ${b.L.toFixed(1)}`; $('s_cl').textContent=`${a.cl.toFixed(1)} ~ ${b.cl.toFixed(1)}`;}
  else{$('s_R').textContent=$('s_L').textContent=$('s_cl').textContent='—';}
  drawSupport(it,lo,hi,maj);
}
function drawSupport(it,lo,hi,maj){
  const w=640,h=240,L=42,Rp=14,T=14,Bt=28;
  const lx=r=>L+(Math.log10(r)-Math.log10(RG[0]))/(Math.log10(RG[RG.length-1])-Math.log10(RG[0]))*(w-L-Rp);
  const ly=v=>T+(1-v)*(h-T-Bt);
  const blue=cvar('--blue'),line=cvar('--line'),col=maj?cvar('--purple'):cvar('--amber');
  let g=`<line x1="${L}" y1="${ly(.5)}" x2="${w-Rp}" y2="${ly(.5)}" stroke="${line}" stroke-dasharray="4 3"/><text class="axis" x="${w-Rp}" y="${ly(.5)-4}" text-anchor="end">지지 50%</text>`;
  [0,0.5,1].forEach(v=>{ g+=`<text class="axis" x="${L-6}" y="${ly(v)+3}" text-anchor="end">${(v*100)|0}%</text>`; });
  if(lo!=null){ g+=`<rect x="${lx(lo)}" y="${T}" width="${lx(hi)-lx(lo)}" height="${h-T-Bt}" fill="${col}" opacity=".12"/>`;
    [[lo,'end',-3],[hi,'start',3]].forEach(([R,anc,dx])=>{ g+=`<line x1="${lx(R)}" y1="${T}" x2="${lx(R)}" y2="${h-Bt}" stroke="${col}" stroke-width="1.4"/><text class="axis" x="${lx(R)+dx}" y="${T+8}" text-anchor="${anc}" fill="${col}" font-weight="700">${R}</text>`; }); }
  const pts=RG.map((r,i)=>`${lx(r).toFixed(1)},${ly(it.sup[i]).toFixed(1)}`).join(' ');
  g+=`<polyline points="${pts}" fill="none" stroke="${blue}" stroke-width="2.2" stroke-linejoin="round"/>`;
  [1,2,5,10,20,50,100].forEach(r=>{if(r<RG[0]||r>RG[RG.length-1])return; g+=`<text class="axis" x="${lx(r)}" y="${h-9}" text-anchor="middle">${r}</text>`;});
  g+=`<text class="axis" x="${w-Rp}" y="${h-9}" text-anchor="end">R</text>`;
  $('iChart').innerHTML=g;
}

/* ① 검사정책 — nested 격자 조회(낙관편향 제거) */
function gridLookup(lam,mu){const lamR=Math.min(100,Math.max(2,Math.round(lam)));let muR=POL.MUS[0],bd=1e9;for(const m of POL.MUS){const d=Math.abs(m-mu);if(d<bd){bd=d;muR=m;}}return POL.grid[`${lamR}_${muR}`];}
function renderPol(){
  const cl=+$('pCl').value,beta=Math.max(.1,+$('pBeta').value||.95),mu=Math.max(0,+$('pMu').value||0);
  const lam=beta*cl, g=gridLookup(lam,mu),[tn,fp,fn,tp]=g.c;
  $('pLM').textContent=`λ=${lam.toFixed(1)},  μ=${mu.toFixed(2)}`;
  const passAll=M.npos*cl, inspAll=M.N+M.npos*cl-M.npos*lam+M.nneg*mu, sel=(tp+fp)+M.npos*cl-tp*lam+fp*mu;
  const floor=Math.min(passAll,inspAll), rate=(tp+fp)/M.N, degen=rate>=0.9, selEff=degen?inspAll:sel;
  const costs=[passAll,inspAll,selEff], winner=costs.indexOf(Math.min(...costs));
  const svInsp=(inspAll-selEff)/inspAll*100, svFloor=(floor-selEff)/floor*100, det=tp/(tp+fn)*100;
  $('pTh').textContent=g.th.toFixed(3); $('pThHint').textContent=g.th.toFixed(3);
  $('pSave').textContent=(svInsp>=0?'+':'')+svInsp.toFixed(1)+'%'; $('pSave').style.color=svInsp>0?'var(--green)':'var(--red)';
  $('pSaveF').textContent=(svFloor>=0?'+':'')+svFloor.toFixed(1)+'%'; $('pSaveF').style.color=svFloor>0?'var(--green)':'var(--red)';
  $('pRate').textContent=(rate*100).toFixed(1)+'%'+(degen?' ⚠퇴화':'');
  const mx=Math.max(passAll,inspAll,sel);
  [[passAll,'pa1','pf1','pb1'],[inspAll,'pa2','pf2','pb2'],[sel,'pa3','pf3','pb3']].forEach(([c,a,f,b],k)=>{
    $(a).textContent=Math.round(c).toLocaleString(); $(f).style.width=(c/mx*100)+'%';
    $(b).hidden=(winner!==k)&&!(k===2&&winner===1&&degen);});
  const V=[{t:'전수통과 (검사 안 함)',d:'유출 손실이 검사비보다 싸다. 검사 자체가 불필요.',c:'var(--gray)',i:'①'},
    {t:'전수검사 (전부 재검사)',d:'유출이 너무 비싸다. 선별은 전수검사로 폴백(퇴화/열위).',c:'var(--red)',i:'②'},
    {t:'선별검사 (474모델)',d:'점수가 임계값 넘는 run만 재검사. 두 대안을 모두 이긴다.',c:'var(--blue)',i:'③'}][winner];
  $('pIcon').textContent=V.i; $('pIcon').style.background=V.c; $('pTitle').textContent=V.t; $('pDesc').textContent=V.d;
  if($('kStrat')){$('kStrat').textContent=V.t; $('kSave').textContent=(svInsp>=0?'+':'')+svInsp.toFixed(1)+'%';
    $('kSave').style.color=svInsp>0?'var(--green)':'var(--red)'; $('kTh').textContent=g.th.toFixed(3);}  // XAI 요약 KPI 미러(live)
  const wn=$('pWarn');
  if(winner===2){wn.className='warn ok'; wn.innerHTML=`<b>선별검사 권고.</b> 전수검사 대비 <b>${svInsp.toFixed(1)}%</b>, 무료기준선(더 싼 쪽) 대비 <b>${svFloor.toFixed(1)}%</b> 절감 <span style="opacity:.75">(보간 조건 기준)</span>.`;}
  else if(winner===1){wn.className='warn'; wn.innerHTML=`<b>선별=전수검사(폴백).</b> ${degen?'검사율 90%↑ 퇴화':'선별이 전수검사보다 열위'} → 전수검사로 운영.`;}
  else{wn.className='warn bad'; wn.innerHTML=`<b>검사 불필요.</b> 유출 비용이 낮아(cl=${cl}) 불량을 그대로 내보내는 편이 쌈(전수통과가 무료기준선).`;}
  // 두 경계
  $('bEcon').textContent=POL.econ_median; $('bEconR').textContent=`IQR [${POL.econ_iqr[0]}, ${POL.econ_iqr[1]}]`;
  $('bGuard').textContent=POL.guard_median; $('bGuardR').textContent=`IQR [${POL.guard_iqr[0]}, ${POL.guard_iqr[1]}]`;
  // ⚠️ 25차: r 수치 인용 철회 (분할변동 설계 n=50, CI [-0.16,+0.37] — '무상관'이 아니라 '검출 못함')
  $('bMsg').innerHTML=`두 경계는 <b>서로 다른 통계량이 결정합니다</b> — 운영 가드는 ROC 꼬리(fn 카운트), 경제 교차는 중심 영역(검사 건수). `
    +`seed 재분할 ${POL.n_seed}회 중 <b>${POL.order_flip}회에서 대소가 뒤집히므로</b>(CI [0.38, 0.66]) `
    +`선별 실익 상한은 <b>단일 숫자로 보고하지 않습니다</b>. (min을 쓰는 rec 통계는 순열대조로 기각·철회)`;
}
$('iRate').addEventListener('input',e=>{$('iRateS').value=e.target.value;renderInv();});
$('iRateS').addEventListener('input',e=>{$('iRate').value=e.target.value;renderInv();});
['iBeta','iMu'].forEach(id=>$(id).addEventListener('input',renderInv));
$('pCl').addEventListener('input',e=>{$('pClS').value=e.target.value;renderPol();});
$('pClS').addEventListener('input',e=>{$('pCl').value=e.target.value;renderPol();});
['pBeta','pMu'].forEach(id=>$(id).addEventListener('input',renderPol));
function renderXai(){const x=D.xai; $('xT3').textContent=(x.s59_top3*100).toFixed(0)+'%'; $('xT10').textContent=(x.s59_top10*100).toFixed(0)+'%';
  $('xNC').textContent=x.n_clusters; $('xCS').textContent=x.s59_cluster_size; $('xJ').textContent=x.jaccard.toFixed(2);
  if(x.ctrl&&x.ctrl['15']!=null) $('xCtrl').textContent=x.ctrl['15']+'%';}
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>{renderInv();renderPol();});
renderInv(); renderPol(); renderXai();
</script>
</body>
</html>"""

data["xai"] = xai
html = (HTML.replace("__SIMDATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__MDLIMG__", mdl_png).replace("__XAIIMG__", oof1_png).replace("__GRPIMG__", oof2_png)
            .replace("__CONDIMG__", cond_png).replace("__LOCALIMG__", local_png)
            .replace("__CAND5__", CAND5).replace("__TOP1N__", str(TOP1_N))
            .replace("__NCFG__", str(N_CFG)).replace("__J10__", f"{J10_MED:.2f}"))
open(os.path.join(WEB, "simulator.html"), "w", encoding="utf-8").write(html)
print(f"→ web/simulator.html ({len(html)/1024:.0f} KB)")
