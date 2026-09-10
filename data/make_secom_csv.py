# -*- coding: utf-8 -*-
"""UCI SECOM 원본(secom.data, secom_labels.data) → data/fab_process_yield.csv

사용법: python data/make_secom_csv.py secom.data secom_labels.data
출력 열: Time, 0..589, Pass/Fail  (불량 +1, 정상 -1)
"""
import os, sys
import numpy as np, pandas as pd

def main(data_path, label_path):
    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
    lab = pd.read_csv(label_path, sep=r"\s+", header=None, names=["Pass/Fail", "date", "time"])
    if len(X) != len(lab):
        raise SystemExit(f"행 수 불일치: data {len(X)} vs labels {len(lab)}")
    ts = pd.to_datetime(lab["date"] + " " + lab["time"], format="%d/%m/%Y %H:%M:%S")
    X.columns = [str(i) for i in range(X.shape[1])]
    out = pd.concat([ts.dt.strftime("%Y-%m-%d %H:%M:%S").rename("Time"), X, lab["Pass/Fail"].astype(int)], axis=1)
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "fab_process_yield.csv")
    out.to_csv(dst, index=False)
    print(f"저장: {dst}  ({len(out)} 행, 센서 {X.shape[1]}개, 불량 {(out['Pass/Fail']==1).sum()}건)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
