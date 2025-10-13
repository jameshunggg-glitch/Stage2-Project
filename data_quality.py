"""
data_quality.py
---------------
航程資料品質檢查模組
提供對航程訊號密度、缺口與覆蓋率的量化指標
"""

import numpy as np
import pandas as pd


def voyage_integrity_metrics(
    df: pd.DataFrame,
    voyages: pd.DataFrame,
    expected_interval_sec: float | None = None,
) -> pd.DataFrame:
    """
    為每個航程計算資料完整性指標：
    --------------------------------------------------------
    - max_gap_hr : 航程中最大的訊號間隔（小時）
    - avg_signal_interval_sec : 平均訊號間隔（秒）
    - coverage_ratio : 訊號密度（實際筆數 / 預期筆數）
    - signal_quality_flag : 根據訊號稀疏程度與缺口判斷資料品質

    參數
    ----
    df : 包含 Timestamp、voyage_id 的 DataFrame
    voyages : 航程摘要表，至少包含 voyage_id
    expected_interval_sec : 預期訊號間隔秒數，若 None 則使用各航程中位數作為基準

    回傳
    ----
    voyages (DataFrame) : 加上上述欄位的品質檢查結果
    """

    out = voyages.copy()
    max_gaps, avg_intervals, coverages, flags = [], [], [], []

    for _, row in out.iterrows():
        vid = int(row["voyage_id"])
        seg = df[df["voyage_id"] == vid].sort_values("Timestamp")

        if len(seg) < 2:
            max_gaps.append(np.nan)
            avg_intervals.append(np.nan)
            coverages.append(np.nan)
            flags.append("NoData")
            continue

        # 計算訊號時間間隔
        deltas = seg["Timestamp"].diff().dt.total_seconds().iloc[1:]
        max_gap_hr = deltas.max() / 3600.0
        avg_interval = deltas.mean()

        # 訊號覆蓋率（coverage_ratio）
        med = np.median(deltas)
        baseline = expected_interval_sec if expected_interval_sec else med
        total_time = (seg["Timestamp"].iloc[-1] - seg["Timestamp"].iloc[0]).total_seconds()
        expected_points = max(total_time / baseline, 1)
        coverage = len(seg) / expected_points

        # 品質旗標
        if max_gap_hr > 12:
            flag = "Low"
        elif avg_interval > 600:  # 平均間隔超過10分鐘
            flag = "Medium"
        else:
            flag = "High"

        max_gaps.append(max_gap_hr)
        avg_intervals.append(avg_interval)
        coverages.append(coverage)
        flags.append(flag)

    out["max_gap_hr"] = max_gaps
    out["avg_signal_interval_sec"] = avg_intervals
    out["coverage_ratio"] = coverages
    out["signal_quality_flag"] = flags

    return out
