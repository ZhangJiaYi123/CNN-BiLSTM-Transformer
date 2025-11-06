#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_windows.py (cleaning + annotated)
修正版：增加对 inf/极端值的清洗与填充，避免 MinMaxScaler.fit 时出现
"Input X contains infinity or a value too large for dtype('float64')" 错误。
保持之前功能：按时间或按文件划分 -> 滑窗 -> Any-Attack 标签 -> 保存 npz。
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import json

# --------------------------
# LABEL MAP / CATEGORIES
# --------------------------
LABEL_MAP = {
    'BENIGN': 'Benign',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DDoS',
    'DoS GoldenEye': 'DDoS',
    'DoS slowloris': 'DDoS',
    'DoS Slowhttptest': 'DDoS',
    'PortScan': 'PortScan',
    'Brute Force': 'BruteForce',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'Web Attack – Brute Force': 'WebAttack',
    'Web Attack – XSS': 'WebAttack',
    'Web Attack – Sql Injection': 'WebAttack',
    'Bot': 'Botnet',
    'Infiltration': 'Infiltration',
}

CATEGORIES = ['Benign', 'DoS', 'DDoS', 'PortScan',
              'BruteForce', 'WebAttack', 'Botnet', 'Infiltration']
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}


def map_label(raw_label):
    """Map raw label to our target categories. Default to 'Benign' if not matched."""
    s = str(raw_label)
    for k, v in LABEL_MAP.items():
        if k.lower() in s.lower():
            return v
    return 'Benign'

# --------------------------
# 主函数
# --------------------------


def build_windows_from_csvs(csv_dir, out_dir, T=50, stride=25,
                            time_train_frac=0.7, val_frac=0.15, time_col=None,
                            extreme_threshold=1e12):
    """
    extreme_threshold: 绝对值超过该阈值的数将被视为异常并作为 NaN 处理（以防止过大数字导致 sklearn 报错）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 收集 CSV 文件
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    print(f"Found {len(csv_paths)} csv files. Loading...")

    # 读取每个文件并记录来源文件名（用于 file-based split）
    df_list = []
    for p in tqdm(csv_paths, desc="loading csvs"):
        sub = pd.read_csv(p, low_memory=False)
        sub["_source_file"] = os.path.basename(p)
        df_list.append(sub)

    # 合并
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows after concat: {len(df)}")

    # 2) 检测时间列（或回退到 file-based split）
    use_file_based_split = False
    time_column = None
    if time_col and time_col in df.columns:
        time_column = time_col
        print(f"Using user-specified time column: {time_column}")
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df = df.sort_values(time_column).reset_index(drop=True)
    else:
        time_cols = [c for c in df.columns if "time" in c.lower()
                     or "timestamp" in c.lower()]
        if len(time_cols) > 0:
            time_column = time_cols[0]
            print(f"Auto-detected time column: {time_column}")
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            parsed_frac = df[time_column].notna().mean()
            if parsed_frac < 0.2:
                print(
                    f"Detected time column '{time_column}' but parse success only {parsed_frac:.2%} -> fallback to file-based split.")
                time_column = None
                use_file_based_split = True
            else:
                df = df.sort_values(time_column).reset_index(drop=True)
        else:
            print("No timestamp-like column found -> fallback to file-based split.")
            use_file_based_split = True

    # 3) label 列检测和映射
    label_cols = [c for c in df.columns if "label" in c.lower()
                  or "attack" in c.lower()]
    if not label_cols:
        # 打印可用列名帮助定位问题
        raise ValueError(
            "Cannot find label column. Available columns: " + ",".join(df.columns))
    label_col = label_cols[0]
    print("Using label column:", label_col)
    df["mapped_label"] = df[label_col].apply(map_label)

    # 4) 选择 numeric 列
    drop_like = ["Flow ID", "Src IP", "Dst IP",
                 "Protocol", "Timestamp", "Label", "_source_file"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in drop_like]
    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns detected. Columns: " + ",".join(df.columns))
    print(f"Using {len(numeric_cols)} numeric features.")

    # 5) 划分 train/val/test（file-based 或 time-based）
    if use_file_based_split:
        unique_files = sorted(df["_source_file"].unique())
        n_files = len(unique_files)
        n_train = max(1, int(n_files * time_train_frac))
        n_val = max(1, int(n_files * val_frac))
        train_files = unique_files[:n_train]
        val_files = unique_files[n_train:n_train + n_val]
        test_files = unique_files[n_train + n_val:]
        print(
            f"File-based split -> train {len(train_files)} files, val {len(val_files)} files, test {len(test_files)} files")

        def assign_split_by_file(x):
            if x in train_files:
                return "train"
            elif x in val_files:
                return "val"
            else:
                return "test"
        df["split"] = df["_source_file"].apply(assign_split_by_file)
    else:
        times = df[time_column]
        t0, t1 = times.min(), times.max()
        total_seconds = (t1 - t0).total_seconds()
        train_cutoff = t0 + \
            pd.Timedelta(seconds=total_seconds * time_train_frac)
        val_cutoff = t0 + \
            pd.Timedelta(seconds=total_seconds * (time_train_frac + val_frac))
        print("Time range:", t0, "->", t1)
        print("Train cutoff:", train_cutoff, "Val cutoff:", val_cutoff)

        def assign_split_by_time(ts):
            if ts <= train_cutoff:
                return "train"
            elif ts <= val_cutoff:
                return "val"
            else:
                return "test"
        df["split"] = df[time_column].apply(assign_split_by_time)

    # 6) 数据清洗：处理 inf / 极端值，并用训练集中位数填充
    # NOTE: 这是关键修正，避免 MinMaxScaler.fit 报错
    # 6.1 先将 inf 替换为 NaN（inplace）
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 6.2 将绝对值过大的数也当作异常 (threshold 可调整)
    # 将过大值设为 NaN，避免 sklearn 在 fit 时报错或造成数值不稳定
    abs_vals = df[numeric_cols].abs()
    mask_extreme = abs_vals > extreme_threshold
    if mask_extreme.values.any():
        print(f"Found {mask_extreme.values.sum()} extreme entries (abs > {extreme_threshold}). These will be treated as NaN and imputed.")
        # 把极端位置置为 NaN
        df.loc[:, numeric_cols] = df[numeric_cols].mask(
            mask_extreme, other=np.nan)

    # 6.3 计算训练集每列中位数并用它填充所有 NaN（训练集优先）
    train_df = df[df["split"] == "train"]
    if len(train_df) == 0:
        raise ValueError(
            "No rows in training split. Check split strategy (time_col / file order).")
    medians = train_df[numeric_cols].median(skipna=True)
    # 如果某列在训练集中全部为 NaN，会导致 medians 为 NaN -> 抛出明确错误提示以便用户检查
    cols_all_nan = medians[medians.isna()].index.tolist()
    if len(cols_all_nan) > 0:
        raise ValueError(f"The following numeric columns are entirely NaN in the training set (cannot impute): {cols_all_nan}. "
                         f"Please check CSVs or choose a different set of numeric features.")
    # 用训练集中位数填充全表
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    # 最终再替换掉任何仍然存在的 inf（理论上应无）
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if df[numeric_cols].isna().sum().sum() > 0:
        # 此时若还有 NaN，说明填充未生效或 medians 有 NaN（已在上面阻断），提示用户
        raise ValueError(
            "After imputation, numeric features still contain NaN. Aborting. Check data and imputation logic.")

    # 7) 归一化：fit 在训练行级（训练集行），使用 MinMaxScaler
    scaler = MinMaxScaler()
    train_vals = df[df["split"] == "train"][numeric_cols].values
    # 作为保险：再检查训练值是否包含非有限数
    if not np.all(np.isfinite(train_vals)):
        # 说明清洗/填充未完全生效
        n_nonfinite = np.count_nonzero(~np.isfinite(train_vals))
        raise ValueError(
            f"Training values contain non-finite entries ({n_nonfinite}). Check earlier cleaning steps.")
    # fit scaler
    scaler.fit(train_vals)
    # transform all rows
    df[numeric_cols] = scaler.transform(df[numeric_cols].values)

    # 8) 构造滑窗（按 split 分别执行，避免跨 split 窗口）
    def make_windows_for_df(sub_df):
        X_ws = []
        y_ws = []
        feats = sub_df[numeric_cols].values
        labels = sub_df["mapped_label"].values
        if len(feats) < T:
            return np.zeros((0, T, len(numeric_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        for start in range(0, len(feats) - T + 1, stride):
            end = start + T
            window_feats = feats[start:end]
            window_labels = labels[start:end]
            if np.all(window_labels == "Benign"):
                lbl = "Benign"
            else:
                attacks = [l for l in window_labels if l != "Benign"]
                if len(attacks) == 0:
                    lbl = "Benign"
                else:
                    lbl = max(set(attacks), key=attacks.count)
            if lbl not in CATEGORY_TO_ID:
                raise ValueError(
                    f"Window label '{lbl}' not found in CATEGORY_TO_ID mapping.")
            X_ws.append(window_feats.astype(np.float32))
            y_ws.append(CATEGORY_TO_ID[lbl])
        if len(X_ws) == 0:
            return np.zeros((0, T, len(numeric_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.stack(X_ws, axis=0), np.array(y_ws, dtype=np.int64)

    outputs = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp].reset_index(drop=True)
        X_sp, y_sp = make_windows_for_df(sub)
        print(f"{sp}: windows={X_sp.shape}, labels={y_sp.shape}")
        outputs[sp] = (X_sp, y_sp)

    # 9) 保存为单个 npz 文件，包含 feature_names
    out_path = os.path.join(out_dir, "cicids2017_windows.npz")
    np.savez_compressed(
        out_path,
        train_X=outputs["train"][0],
        train_y=outputs["train"][1],
        val_X=outputs["val"][0],
        val_y=outputs["val"][1],
        test_X=outputs["test"][0],
        test_y=outputs["test"][1],
        feature_names=np.array(numeric_cols, dtype=object)
    )
    print("✅ Saved windows npz ->", out_path)

    # 10) 同时把 scaler（joblib）与 class mapping 保存，便于推理（可选）
    try:
        import joblib
        joblib.dump(scaler, os.path.join(out_dir, "minmax_scaler.save"))
        print("Saved scaler ->", os.path.join(out_dir, "minmax_scaler.save"))
    except Exception as e:
        print("Warning: failed to save scaler via joblib:", e)

    mapping_path = os.path.join(out_dir, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"CATEGORY_TO_ID": CATEGORY_TO_ID, "CATEGORIES": CATEGORIES,
                  "numeric_features": numeric_cols}, f, ensure_ascii=False, indent=2)
    print("Saved class mapping ->", mapping_path)


# --------------------------
# CLI 参数
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make sliding-window sequences from CICIDS2017 CSVs (with cleaning for inf/extreme values).")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing CICIDS2017 CSV files")
    parser.add_argument("--out_dir", type=str,
                        required=True, help="Output directory")
    parser.add_argument("--T", type=int, default=50, help="Window length")
    parser.add_argument("--stride", type=int, default=25, help="Window stride")
    parser.add_argument("--time_train_frac", type=float, default=0.7,
                        help="Train fraction on time axis (if using time-based split)")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Validation fraction on time axis")
    parser.add_argument("--time_col", type=str, default=None,
                        help="Optional name of timestamp column in CSVs")
    parser.add_argument("--extreme_threshold", type=float, default=1e12,
                        help="Absolute-value threshold to treat as extreme and set to NaN before imputation")
    args = parser.parse_args()

    build_windows_from_csvs(
        csv_dir=args.csv_dir,
        out_dir=args.out_dir,
        T=args.T,
        stride=args.stride,
        time_train_frac=args.time_train_frac,
        val_frac=args.val_frac,
        time_col=args.time_col,
        extreme_threshold=args.extreme_threshold
    )
