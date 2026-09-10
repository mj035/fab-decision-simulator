"""
입력 어댑터 — 임의 정형 CSV를 규격독립 파이프라인의 입력(수치 X + 이진 y)으로 변환.
'형식'은 자동 흡수하되, 도메인 지식 3가지(라벨열·양성클래스·비용비)는 사용자가 명세한다(정직한 경계).

adapt(df, label_col, positive_class, drop_cols=None, min_numeric_frac=0.5) -> (X, y, info)
  · 라벨 정규화 : positive_class == 1, 그 외 == 0  (-1/+1, neg/pos, 0/1 다 흡수)
  · 비수치/ID/시간열 자동 제외 : object열 중 수치 파싱률 < min_numeric_frac 이면 drop
  · 결측 마커 흡수 : 'na', 빈칸, 'NaN', '?' 등 → NaN (to_numeric coerce)
  · 결측/상수 처리 자체는 다음 단계(SimpleImputer+VarianceThreshold)가 담당 — 규격독립 유지
"""
import numpy as np, pandas as pd

NA_TOKENS = ["na", "n/a", "nan", "null", "none", "?", "", "-", "missing"]

def adapt(df, label_col, positive_class, drop_cols=None, min_numeric_frac=0.5):
    df = df.copy()
    info = {"n_row_in": int(len(df)), "n_col_in": int(df.shape[1])}

    if label_col not in df.columns:
        raise ValueError(f"라벨열 '{label_col}' 없음. 실제 열: {list(df.columns)[:10]}...")

    # 1) 라벨 분리 + 이진 정규화
    raw_y = df[label_col]
    def _match(v):
        return str(v).strip().lower() == str(positive_class).strip().lower()
    y = raw_y.map(_match).astype(int).to_numpy()
    df = df.drop(columns=[label_col])
    info["pos_rate"] = round(float(y.mean()), 4)
    info["n_pos"] = int(y.sum())

    # 2) 사용자 지정 제외열
    dropped_user = []
    if drop_cols:
        exist = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=exist); dropped_user = exist

    # 3) 결측 토큰 → NaN, 수치 강제 변환
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().replace(NA_TOKENS, np.nan)
    X_num = df.apply(pd.to_numeric, errors="coerce")

    # 4) 비수치열 자동 제외 : object였고 수치 파싱률 낮은 열
    dropped_nonnum = []
    for c in obj_cols:
        parse_frac = X_num[c].notna().mean()
        orig_notna = df[c].notna().mean()
        # 원래 값이 있었는데(결측 아닌데) 수치로 못 바꾼 비율이 높으면 = 문자/ID/시간열
        if orig_notna > 0 and (parse_frac / max(orig_notna, 1e-9)) < min_numeric_frac:
            dropped_nonnum.append(c)
    X = X_num.drop(columns=dropped_nonnum)

    info.update({
        "dropped_user": dropped_user,
        "dropped_nonnumeric": dropped_nonnum,
        "n_feat_out": int(X.shape[1]),
        "miss_rate_out": round(float(X.isna().mean().mean()), 4),
    })
    return X, y, info


def adapt_from_csv(path, label_col, positive_class, drop_cols=None, skip_comment_prefix=None, **kw):
    """앞쪽 주석줄이 있는 파일(APS 등)도 처리."""
    if skip_comment_prefix:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith(skip_comment_prefix))
        import io
        df = pd.read_csv(io.StringIO("".join(lines[hdr:])))
    else:
        df = pd.read_csv(path)
    return adapt(df, label_col, positive_class, drop_cols=drop_cols, **kw)
