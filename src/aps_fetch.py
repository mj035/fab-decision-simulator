# APS Failure at Scania Trucks (UCI id=421) 정적 CSV 로드 + 진단
# 공식 비용비: FP(불필요 점검)=10, FN(고장 놓침)=500  ->  50:1
# 파일은 data_aps/ 에 이미 압축해제됨 (aps_fetch 이전 단계에서 zip 다운로드)
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # 콘솔 한글 깨짐 방지

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data_aps")

def load_aps(fname):
    """앞쪽 주석 줄을 건너뛰고 'class,...' 헤더부터 읽는다. 결측='na'."""
    path = os.path.join(DATA, fname)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 헤더 행 = 'class'로 시작하는 첫 줄
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("class"))
    df = pd.read_csv(io.StringIO("".join(lines[hdr:])), na_values="na")
    return df

print("[1] APS train/test CSV 로드...")
train = load_aps("aps_failure_training_set.csv")
test = load_aps("aps_failure_test_set.csv")

# 라벨 분리: class 컬럼 'pos'/'neg' -> 1/0
def split_xy(df):
    y = (df["class"].astype(str).str.lower().str.strip() == "pos").astype(int)
    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
    return X, y

Xtr, ytr = split_xy(train)
Xte, yte = split_xy(test)

print("[2] 정제본 저장(train)...")
Xtr.to_csv(os.path.join(DATA, "aps_X.csv"), index=False)
ytr.to_frame("fail").to_csv(os.path.join(DATA, "aps_y.csv"), index=False)

# ---- 진단 (train 기준) ----
n, p = Xtr.shape
pos = int(ytr.sum())
miss = Xtr.isna().mean().mean()
const_cols = int((Xtr.nunique(dropna=True) <= 1).sum())

print("\n" + "=" * 52)
print("APS 데이터 진단 (training set)")
print("=" * 52)
print(f"행(row)         : {n:,}")
print(f"열(feature)     : {p}")
print(f"양성(fail=1)    : {pos:,}  ({pos/n*100:.2f}%)")
print(f"음성(fail=0)    : {n-pos:,}")
print(f"결측률(평균)    : {miss*100:.2f}%")
print(f"상수/분산0 센서 : {const_cols}")
print(f"test set        : {Xte.shape[0]:,}행 / 양성 {int(yte.sum()):,}")
print("=" * 52)
print("\n[비교] SECOM : 1,567행 / 590피처 / 양성 104(6.64%) / 결측 4.54% / 상수 116")
print("[APS 공식 비용비] FN=500, FP=10  ->  50:1")
print(f"\n저장: {os.path.abspath(DATA)}  (aps_X.csv, aps_y.csv)")
