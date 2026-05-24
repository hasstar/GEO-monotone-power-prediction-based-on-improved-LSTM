from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import IterableDataset
import os
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
import csv
import gc
import random
from scipy.stats import ks_2samp


# 加载函数load_era_dict：导入测试集、验证集、训练集的大气文件。返回era_dict, feature_names
def load_era_dict(split):
    # print(f"\n==================== 加载函数load_era_dict：导入{split}的大气文件。返回era_dict, feature_names ====================")

    dfs = []
    feature_names = []  # 👈 新增：按最终拼接顺序记录名字
    global_idx = 0
    for name in ERA_FILES:
        df = pd.read_csv(
            f"{SPLIT_DIR}/{name}_{split}.csv",
            parse_dates=["valid_time"]
        )
        df = df.set_index(df["valid_time"].dt.floor("h"))
        df = df.drop(columns=["valid_time"])
        for col in df.columns:
            feature_names.append(f"{global_idx:03d}|{name}|{col}")
            global_idx += 1

        dfs.append(df)

    era = pd.concat(dfs, axis=1)
    era_dict = {
        hour: row.to_numpy(dtype=np.float32)
        for hour, row in era.iterrows()
    }
    return era_dict, feature_names


# 自动获取大气参数每层的维度
def get_layer_dims(split_dir, era_files, split="train"):
    # 大气参数已经拼接了，为了方便分开，获取拼接前每个文件的参数个数
    dims = []
    for name in era_files:
        df = pd.read_csv(f"{split_dir}/{name}_{split}.csv", nrows=1)
        # 排除 valid_time 列
        cols = [c for c in df.columns if c != 'valid_time']
        dims.append(len(cols))
    return dims


# 数据集划分类
class PowerAtmSlidingDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, Y, P):
        super().__init__()
        self.split = split
        self.Y = Y
        self.P = P

        self.power_path = f"{SPLIT_DIR}/power_{split}.csv"
        self.era_dict, self.feature_names = load_era_dict(split)

    def __iter__(self):
        df = pd.read_csv(
            self.power_path,
            parse_dates=["Timestamp"]
        ).sort_values("Timestamp")

        df["hour"] = df["Timestamp"].dt.floor("h")

        for hour, g in df.groupby("hour"):
            if hour not in self.era_dict:
                continue

            times = g["Timestamp"].to_numpy()
            power = g["TimePower"].to_numpy(dtype=np.float32)
            atm = self.era_dict[hour]

            if len(power) < self.Y + self.P:
                continue

            for i in range(len(power) - self.Y - self.P + 1):
                x = torch.from_numpy(power[i:i + self.Y]).unsqueeze(-1)
                y = torch.from_numpy(power[i + self.Y:i + self.Y + self.P])
                a = torch.from_numpy(atm)

                # ✅ 转成 string（关键）
                t_x_start = str(times[i])
                t_x_end = str(times[i + self.Y - 1])
                t_y_end = str(times[i + self.Y + self.P - 1])
                hour_str = str(hour)

                yield x, y, a, t_x_start, t_x_end, t_y_end, hour_str


# 数据集划分类-可均值
class PowerAtmSlidingDatasetRate(torch.utils.data.IterableDataset):
    def __init__(self, split, Y, P, rate=10):
        super().__init__()
        self.split = split
        self.rate = rate  # 采样率（秒）

        # 将物理时长转换为对应的点数
        self.Y = Y // rate
        self.P = P // rate

        self.power_path = f"{SPLIT_DIR}/power_{split}.csv"
        self.era_dict, self.feature_names = load_era_dict(split)

    def __iter__(self):
        print(f"\n Loading {self.power_path}...")
        df = pd.read_csv(self.power_path, parse_dates=["Timestamp"]).sort_values("Timestamp")
        df["hour"] = df["Timestamp"].dt.floor("h")

        for hour, g in df.groupby("hour"):
            if hour not in self.era_dict: continue

            # --- 执行聚合平均 ---
            power_raw = g["TimePower"].to_numpy(dtype=np.float32)
            times_raw = g["Timestamp"].to_numpy()

            n_points = len(power_raw) // self.rate
            if n_points < (self.Y + self.P): continue

            # 聚合计算：均值处理
            power = power_raw[:n_points * self.rate].reshape(-1, self.rate).mean(axis=1)
            # print(f"len(power):{len(power)}")
            # 时间取中点或起点
            times = times_raw[:n_points * self.rate:self.rate]

            atm = self.era_dict[hour]

            for i in range(len(power) - self.Y - self.P + 1):
                x = torch.from_numpy(power[i: i + self.Y]).unsqueeze(-1)
                y = torch.from_numpy(power[i + self.Y: i + self.Y + self.P])
                a = torch.from_numpy(atm)

                yield x, y, a, str(times[i]), str(times[i + self.Y - 1]), str(times[i + self.Y + self.P - 1]), str(hour)


# 数据集划分使用例程
def exampleUsingDataset():
    print(f"\n==================== 数据集划分的使用示例 ====================")

    ds = PowerAtmSlidingDataset(split="train", Y=1798, P=1798)
    loader = DataLoader(ds, batch_size=5, shuffle=False)

    for i, batch in enumerate(loader):
        x, y, a, t_x_start, t_x_end, t_y_end, hour = batch
        print(x.shape)
        print(y.shape)
        print(a.shape)
        print(f"\n========== Batch {i} ==========")

        batch_size = x.shape[0]

        for b in range(batch_size):
            print(
                f"""
    【Batch {i} | 样本 {b}】

    x  : 秒级功率历史输入
         shape = {tuple(x[b].shape)}   # (Y, 1)
         含义  = 从 {t_x_start[b]} 到 {t_x_end[b]} 的连续 {x[b].shape[0]} 个秒级功率点
         示例  = {x[b, :5, 0].numpy()} ... {x[b, -5:, 0].numpy()}

    y  : 秒级功率预测标签
         shape = {tuple(y[b].shape)}   # (P,)
         含义  = 从 {t_x_end[b]} 之后开始的连续 {y[b].shape[0]} 个秒级功率点
         示例  = {y[b, :5].numpy()} ... {y[b, -5:].numpy()}

    a  : 小时级大气参数（模型输入，不是标签）
         shape = {tuple(a[b].shape)}   # (A,)
         含义  = hour = {hour[b]} 对应的 ERA5 大气特征
         示例  = {a[b, :5].numpy()} ... {a[b, -5:].numpy()}

    时间标注（仅用于人工验证，不参与训练）:
         hour        = {hour[b]}
         X 起点时间 = {t_x_start[b]}
         X 终点时间 = {t_x_end[b]}
         Y 终点时间 = {t_y_end[b]}
    """
            )

        if i == 2:
            break


# 数据集划分使用例程
def exampleUsingDatasetRate(Y, P, rate):
    print(f"\n==================== 数据集划分的使用示例 ====================")

    ds = PowerAtmSlidingDatasetRate(split="train", Y=Y, P=P, rate=rate)
    loader = DataLoader(ds, batch_size=5, shuffle=False)

    for i, batch in enumerate(loader):
        x, y, a, t_x_start, t_x_end, t_y_end, hour = batch
        print(x.shape)
        print(y.shape)
        print(a.shape)
        print(f"\n========== Batch {i} ==========")

        batch_size = x.shape[0]

        for b in range(batch_size):
            print(
                f"""
    【Batch {i} | 样本 {b}】

    x  : 秒级功率历史输入
         shape = {tuple(x[b].shape)}   # (Y, 1)
         含义  = 从 {t_x_start[b]} 到 {t_x_end[b]} 的连续 {x[b].shape[0]} 个{rate}秒级功率点
         示例  = {x[b, :5, 0].numpy()} ... {x[b, -5:, 0].numpy()}

    y  : 秒级功率预测标签
         shape = {tuple(y[b].shape)}   # (P,)
         含义  = 从 {t_x_end[b]} 之后开始的连续 {y[b].shape[0]} 个{rate}秒级功率点
         示例  = {y[b, :5].numpy()} ... {y[b, -5:].numpy()}

    a  : 小时级大气参数（模型输入，不是标签）
         shape = {tuple(a[b].shape)}   # (A,)
         含义  = hour = {hour[b]} 对应的 ERA5 大气特征
         示例  = {a[b, :5].numpy()} ... {a[b, -5:].numpy()}

    时间标注（仅用于人工验证，不参与训练）:
         hour        = {hour[b]}
         X 起点时间 = {t_x_start[b]}
         X 终点时间 = {t_x_end[b]}
         Y 终点时间 = {t_y_end[b]}
    """
            )

        if i == 2:
            break


# 模型评估
@torch.no_grad()
def estimate_metrics_pro(
        model,
        eval_iters,
        data_loader,
        device,
        fade_ratio=0.2,
        pruner=None,
):
    model.eval()

    metrics_sum = {
        "RMSE": 0.0,
        "NRMSE": 0.0,
        "MAE": 0.0,
        "MBE": 0.0,
        "R2": 0.0,
        "Corr": 0.0,
        "W_Dist": 0.0,
        "Tail_W_Dist": 0.0,
        "KS_Dist": 0.0,
        "Fade_RMSE": 0.0,
        "Fade_Recall": 0.0,
        "Fade_Precision": 0.0,
        "Fade_F1": 0.0,
        "Slope_RMSE": 0.0,
    }

    n_batches = 0
    n_fade_batches = 0
    n_slope_batches = 0

    count = 0
    pbar = tqdm(data_loader, desc="Evaluating", leave=False)

    for batch in pbar:

        if eval_iters is not None and count >= eval_iters:
            break
        count += 1

        x, y, a, *_ = batch
        x, y, a = x.to(device), y.to(device), a.to(device)

        if pruner is not None:
            a = a * pruner.mask

        # ---------- 1️⃣ 归一化 ----------
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        preds_norm, _ = model(x_norm, a)
        preds = preds_norm * x_std + x_mean
        preds = preds.squeeze(-1)  # [B, P]

        y_true = y.cpu().numpy()
        y_pred = preds.cpu().numpy()

        B, P = y_true.shape
        diff = y_pred - y_true

        # =========================================================
        # 2️⃣ 基础误差 —— per-sample
        # =========================================================

        mse_sample = np.mean(diff ** 2, axis=1)
        rmse_sample = np.sqrt(mse_sample)
        mae_sample = np.mean(np.abs(diff), axis=1)
        mbe_sample = np.mean(diff, axis=1)

        rmse = np.mean(rmse_sample)
        mae = np.mean(mae_sample)
        mbe = np.mean(mbe_sample)

        std_true_sample = np.std(y_true, axis=1)
        nrmse_sample = rmse_sample / (std_true_sample + 1e-8)
        nrmse = np.mean(nrmse_sample)

        ss_res = np.sum(diff ** 2, axis=1)
        ss_tot = np.sum(
            (y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2,
            axis=1
        )
        r2_sample = 1 - ss_res / (ss_tot + 1e-8)
        r2 = np.mean(r2_sample)

        # Corr per-sample
        corr_list = []
        for i in range(B):
            yt = y_true[i]
            yp = y_pred[i]

            std_yt = np.std(yt)
            std_yp = np.std(yp)

            if std_yt < 1e-8 or std_yp < 1e-8:
                corr_list.append(0.0)
            else:
                cov = np.mean((yt - yt.mean()) * (yp - yp.mean()))
                corr_list.append(cov / (std_yt * std_yp))

        corr = np.mean(corr_list)

        # =========================================================
        # 3️⃣ Wasserstein —— per-sample
        # =========================================================

        preds_sorted, _ = torch.sort(preds, dim=1)
        y_sorted, _ = torch.sort(y, dim=1)

        w_sample = torch.mean(
            torch.abs(preds_sorted - y_sorted),
            dim=1
        )
        w_dist = torch.mean(w_sample).item()

        # Tail Wasserstein
        k = max(1, int(fade_ratio * P))

        tail_sample = torch.mean(
            torch.abs(preds_sorted[:, :k] - y_sorted[:, :k]),
            dim=1
        )
        tail_w = torch.mean(tail_sample).item()

        # =========================================================
        # 4️⃣ KS —— per-sample
        # =========================================================

        ks_list = [
            ks_2samp(y_true[i], y_pred[i]).statistic
            for i in range(B)
        ]
        ks_stat = np.mean(ks_list)

        # =========================================================
        # 5️⃣ 动态深衰落
        # =========================================================

        thresholds_true = np.percentile(
            y_true, fade_ratio * 100, axis=1, keepdims=True
        )
        thresholds_pred = np.percentile(
            y_pred, fade_ratio * 100, axis=1, keepdims=True
        )

        fade_mask_true = y_true <= thresholds_true
        fade_mask_pred = y_pred <= thresholds_pred

        if np.any(fade_mask_true):
            fade_mse = np.mean(
                (y_pred[fade_mask_true] - y_true[fade_mask_true]) ** 2
            )
            fade_rmse = np.sqrt(fade_mse)

            TP = np.sum(fade_mask_true & fade_mask_pred)
            FP = np.sum(~fade_mask_true & fade_mask_pred)
            FN = np.sum(fade_mask_true & ~fade_mask_pred)

            recall = TP / (TP + FN + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics_sum["Fade_RMSE"] += fade_rmse
            metrics_sum["Fade_Recall"] += recall
            metrics_sum["Fade_Precision"] += precision
            metrics_sum["Fade_F1"] += f1

            n_fade_batches += 1

        # =========================================================
        # 6️⃣ Slope —— per-sample
        # =========================================================

        if P > 1:
            slope_true = np.diff(y_true, axis=1)
            slope_pred = np.diff(y_pred, axis=1)

            slope_rmse_sample = np.sqrt(
                np.mean((slope_true - slope_pred) ** 2, axis=1)
            )

            slope_rmse = np.mean(slope_rmse_sample)

            metrics_sum["Slope_RMSE"] += slope_rmse
            n_slope_batches += 1

        # =========================================================
        # 7️⃣ 累加
        # =========================================================

        metrics_sum["RMSE"] += rmse
        metrics_sum["NRMSE"] += nrmse
        metrics_sum["MAE"] += mae
        metrics_sum["MBE"] += mbe
        metrics_sum["R2"] += r2
        metrics_sum["Corr"] += corr
        metrics_sum["W_Dist"] += w_dist
        metrics_sum["Tail_W_Dist"] += tail_w
        metrics_sum["KS_Dist"] += ks_stat

        n_batches += 1

    if n_batches == 0:
        return {}

    results = {
        "RMSE": metrics_sum["RMSE"] / n_batches,
        "NRMSE": metrics_sum["NRMSE"] / n_batches,
        "MAE": metrics_sum["MAE"] / n_batches,
        "MBE": metrics_sum["MBE"] / n_batches,
        "R2": metrics_sum["R2"] / n_batches,
        "Corr": metrics_sum["Corr"] / n_batches,
        "W_Dist": metrics_sum["W_Dist"] / n_batches,
        "Tail_W_Dist": metrics_sum["Tail_W_Dist"] / n_batches,
        "KS_Dist": metrics_sum["KS_Dist"] / n_batches,
        "Fade_RMSE": metrics_sum["Fade_RMSE"] / max(1, n_fade_batches),
        "Fade_Recall": metrics_sum["Fade_Recall"] / max(1, n_fade_batches),
        "Fade_Precision": metrics_sum["Fade_Precision"] / max(1, n_fade_batches),
        "Fade_F1": metrics_sum["Fade_F1"] / max(1, n_fade_batches),
        "Slope_RMSE": metrics_sum["Slope_RMSE"] / max(1, n_slope_batches),
    }

    model.train()
    return results


def rmse(pred, target):
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)

    rmse = torch.sqrt(torch.mean((pred - target) ** 2) + 1e-8)

    # dynamic_range = target.max(dim=1).values - target.min(dim=1).values
    # nrmse_sample = rmse / (dynamic_range + 1e-8)

    return rmse


def wasserstein_1d(pred, target):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)
    pred_sorted, _ = torch.sort(pred, dim=1)
    target_sorted, _ = torch.sort(target, dim=1)
    return torch.mean(torch.abs(pred_sorted - target_sorted))


def tail_loss(pred, target, q=0.1):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)

    threshold = torch.quantile(target, q, dim=1, keepdim=True)

    mask = target <= threshold

    loss = ((pred - target) ** 2) * mask

    return loss.sum() / (mask.sum() + 1e-8)


def fading_distribution_loss(pred, target,
                             alpha=0.4,
                             beta=0.4,
                             gamma=0.2):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)
    loss_point = rmse(pred, target)
    loss_dist = wasserstein_1d(pred, target)
    loss_tail = tail_loss(pred, target)
    total_loss = alpha * loss_point \
                 + beta * loss_dist \
                 + gamma * loss_tail
    return total_loss


def tail_wasserstein_loss(pred, target, q=0.1):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)

    threshold = torch.quantile(target, q, dim=1, keepdim=True)
    mask = target <= threshold

    losses = []

    for b in range(pred.size(0)):
        pred_tail = pred[b][mask[b]]
        target_tail = target[b][mask[b]]

        if pred_tail.numel() < 2:
            continue

        pred_sorted, _ = torch.sort(pred_tail)
        target_sorted, _ = torch.sort(target_tail)

        losses.append(torch.mean(torch.abs(pred_sorted - target_sorted)))

    # if len(losses) == 0:
    #     return torch.tensor(0.0, device=pred.device)
    if len(losses) == 0:
        return pred.mean() * 0.0  # 保留计算图

    return torch.stack(losses).mean()


def fading_tail_distribution_loss(pred, target,
                                  alpha=0.4,
                                  beta=0.6):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)
    loss_point = rmse(pred, target)
    loss_tail_w = tail_wasserstein_loss(pred, target, q=0.1)
    return alpha * loss_point + beta * loss_tail_w


def wasserstein_2d(pred, target):
    # 1. 维度规整 [Batch, P]
    if pred.dim() == 3: pred = pred.squeeze(-1)
    if target.dim() == 3: target = target.squeeze(-1)
    pred_sorted, _ = torch.sort(pred, dim=1)
    target_sorted, _ = torch.sort(target, dim=1)
    return torch.mean((pred_sorted - target_sorted) ** 2)


# 模型训练--大气参数剪枝操作
def train_satellite_model_pruner(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        epochs=50,
        save_dir="checkpoints",
        eval_freq=1,
        patience=10,
        pruner=None,
        round_idx=0
):
    """
    通用的卫星模型训练引擎

    参数:
        model: 定义好的 PyTorch 模型
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        optimizer: 优化器 (如 Adam)
        criterion: 损失函数 (TrendAwareSatelliteLoss)
        device: 'cuda' or 'cpu'
        epochs: 训练轮数
        save_dir: 模型保存路径
        eval_freq: 每多少个 epoch 验证一次
        patience: 早停机制 (Early Stopping) 的忍耐轮数
    """
    os.makedirs(save_dir, exist_ok=True)

    # 记录历史数据
    history = {
        'train_loss': [],

        "val_RMSE": [],
        "val_NRMSE": [],
        "val_MAE": [],
        "val_MBE": [],
        "val_R2": [],
        "val_Corr": [],
        "val_W_Dist": [],
        "val_Tail_W_Dist": [],
        "val_KS_Dist": [],
        "val_Fade_RMSE": [],
        "val_Fade_Recall": [],
        "val_Fade_Precision": [],
        "val_Fade_F1": [],
        "val_Slope_RMSE": [],

        "tes_RMSE": [],
        "tes_NRMSE": [],
        "tes_MAE": [],
        "tes_MBE": [],
        "tes_R2": [],
        "tes_Corr": [],
        "tes_W_Dist": [],
        "tes_Tail_W_Dist": [],
        "tes_KS_Dist": [],
        "tes_Fade_RMSE": [],
        "tes_Fade_Recall": [],
        "tes_Fade_Precision": [],
        "tes_Fade_F1": [],
        "tes_Slope_RMSE": []}

    best_val_rmse = float('inf')
    early_stop_counter = 0

    print(f"🚀 开始训练... 设备: {device}")
    print(f"📂 模型将保存至: {save_dir}")

    for epoch in range(1, epochs + 1):
        # ==========================
        # 1. 训练阶段 (Training)
        # ==========================
        model.train()
        running_loss = 0.0
        batch_count = 0

        # 使用 tqdm 包装训练加载器
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)

        for batch in pbar:
            # 数据解包 (适配 PowerAtmSlidingDataset 的返回格式)
            # x: 功率历史, y: 功率标签, a: 大气参数, 后面的 *_ 是时间字符串忽略掉
            x, y, a, *_ = batch

            x = x.to(device)
            y = y.to(device)
            a = a.to(device)

            # =========== 👇 新增逻辑开始 👇 ===========
            if pruner is not None:
                # 如果传入了剪枝器，就用 Mask 过滤掉不需要的大气参数
                # 被 Mask 掉的参数变为 0，相当于从网络中“切断”了
                a = a * pruner.mask
            # =========== 👆 新增逻辑结束 👆 ===========

            # --- 实时归一化逻辑 ---
            # 计算 x 在序列维度 (dim=1) 上的均值和标准差
            # keepdim=True 保证维度对齐，方便后续计算
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True) + 1e-8  # 加上极小值防止除以0

            # 归一化 x
            x_norm = (x - x_mean) / x_std

            # 前向传播
            optimizer.zero_grad()

            # 注意：这里假设模型输出是 (predictions, attention_weights)
            # 如果模型只输出 predictions，请改为: preds = model(x, a)
            preds_norm, attn_weights = model(x_norm, a)

            # --- 反归一化逻辑 ---
            # 假设 preds_norm 形状为 [Batch, P]
            # 这里的 x_mean 和 x_std 形状是 [Batch, 1, 1]，需要 squeeze 或适配维度
            # 我们根据历史序列的统计特性，将预测值还原回原始功率量级
            preds = preds_norm * x_std + x_mean
            preds = preds.squeeze(-1)
            # 计算损失 (TrendAwareSatelliteLoss)
            loss = criterion(preds, y)

            # 反向传播
            loss.backward()

            # 梯度裁剪 (防止 LSTM 梯度爆炸，非常重要！)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录损失
            current_loss = loss.item()
            running_loss += current_loss
            batch_count += 1

            # 实时更新进度条上的 Loss
            pbar.set_postfix({'Loss': f"{current_loss:.4f}"})

        epoch_loss = running_loss / batch_count if batch_count > 0 else 0
        history['train_loss'].append(epoch_loss)

        # ==========================
        # 2. 验证阶段 (Validation)
        # ==========================
        if epoch % eval_freq == 0:
            # 调用你之前定义的评估函数 (estimate_metrics_pro)
            # 注意：eval_iters 可以设大一点以覆盖更多验证数据
            val_metrics = estimate_metrics_pro(
                model=model,
                eval_iters=None,  # 验证集抽样次数，如果验证集不大可以设为 None 跑全量
                data_loader=val_loader,
                device=device,
                fade_ratio=0.2,  # 根据你的数据调整深衰落阈值
                pruner=pruner
            )
            test_metrics = estimate_metrics_pro(
                model=model,
                eval_iters=None,  # 验证集抽样次数，如果验证集不大可以设为 None 跑全量
                data_loader=test_loader,
                device=device,
                fade_ratio=0.2,  # 根据你的数据调整深衰落阈值
                pruner=pruner
            )

            history['val_RMSE'].append(val_metrics['RMSE'])
            history['val_NRMSE'].append(val_metrics['NRMSE'])
            history['val_MAE'].append(val_metrics['MAE'])
            history['val_MBE'].append(val_metrics['MBE'])
            history['val_R2'].append(val_metrics['R2'])
            history['val_Corr'].append(val_metrics['Corr'])
            history['val_W_Dist'].append(val_metrics['W_Dist'])
            history['val_Tail_W_Dist'].append(val_metrics['Tail_W_Dist'])
            history['val_KS_Dist'].append(val_metrics['KS_Dist'])
            history['val_Fade_RMSE'].append(val_metrics['Fade_RMSE'])
            history['val_Fade_Recall'].append(val_metrics['Fade_Recall'])
            history['val_Fade_Precision'].append(val_metrics['Fade_Precision'])
            history['val_Fade_F1'].append(val_metrics['Fade_F1'])
            history['val_Slope_RMSE'].append(val_metrics['Slope_RMSE'])

            history['tes_RMSE'].append(test_metrics['RMSE'])
            history['tes_NRMSE'].append(test_metrics['NRMSE'])
            history['tes_MAE'].append(test_metrics['MAE'])
            history['tes_MBE'].append(test_metrics['MBE'])
            history['tes_R2'].append(test_metrics['R2'])
            history['tes_Corr'].append(test_metrics['Corr'])
            history['tes_W_Dist'].append(test_metrics['W_Dist'])
            history['tes_Tail_W_Dist'].append(test_metrics['Tail_W_Dist'])
            history['tes_KS_Dist'].append(test_metrics['KS_Dist'])
            history['tes_Fade_RMSE'].append(test_metrics['Fade_RMSE'])
            history['tes_Fade_Recall'].append(test_metrics['Fade_Recall'])
            history['tes_Fade_Precision'].append(test_metrics['Fade_Precision'])
            history['tes_Fade_F1'].append(test_metrics['Fade_F1'])
            history['tes_Slope_RMSE'].append(test_metrics['Slope_RMSE'])

            # 打印本轮总结
            print(f"✅ Epoch {epoch} | Loss: {epoch_loss:.4f} | \n"
                  f" valu_metrics['RMSE']: {val_metrics['RMSE']:.4f}\n"
                  f" valu_metrics['NRMSE']: {val_metrics['NRMSE']:.4f}\n"
                  f" valu_metrics['MAE']: {val_metrics['MAE']:.4f}\n"
                  f" valu_metrics['MBE']: {val_metrics['MBE']:.4f}\n"
                  f" valu_metrics['R2']: {val_metrics['R2']:.4f}\n"
                  f" valu_metrics['Corr']: {val_metrics['Corr']:.4f}\n"
                  f" valu_metrics['W_Dist']: {val_metrics['W_Dist']:.4f}\n"
                  f" valu_metrics['Tail_W_Dist']: {val_metrics['Tail_W_Dist']:.4f}\n"
                  f" valu_metrics['KS_Dist']: {val_metrics['KS_Dist']:.4f}\n"
                  f" valu_metrics['Fade_RMSE']: {val_metrics['Fade_RMSE']:.4f}\n"
                  f" valu_metrics['Fade_Recall']: {val_metrics['Fade_Recall']:.4f}\n"
                  f" valu_metrics['Fade_Precision']: {val_metrics['Fade_Precision']:.4f}\n"
                  f" valu_metrics['Fade_F1']: {val_metrics['Fade_F1']:.4f}\n"
                  f" valu_metrics['Slope_RMSE']: {val_metrics['Slope_RMSE']:.4f}\n"
                  f" test_metrics['RMSE']: {test_metrics['RMSE']:.4f}\n"
                  f" test_metrics['NRMSE']: {test_metrics['NRMSE']:.4f}\n"
                  f" test_metrics['MAE']: {test_metrics['MAE']:.4f}\n"
                  f" test_metrics['MBE']: {test_metrics['MBE']:.4f}\n"
                  f" test_metrics['R2']: {test_metrics['R2']:.4f}\n"
                  f" test_metrics['Corr']: {test_metrics['Corr']:.4f}\n"
                  f" test_metrics['W_Dist ']: {test_metrics['W_Dist']:.4f}\n"
                  f" test_metrics['Tail_W_Dist']: {test_metrics['Tail_W_Dist']:.4f}\n"
                  f" test_metrics['KS_Dist']: {test_metrics['KS_Dist']:.4f}\n"
                  f" test_metrics['Fade_RMSE']: {test_metrics['Fade_RMSE']:.4f}\n"
                  f" test_metrics['Fade_Recall']: {test_metrics['Fade_Recall']:.4f}\n"
                  f" test_metrics['Fade_Precision']: {test_metrics['Fade_Precision']:.4f}\n"
                  f" test_metrics['Fade_F1']: {test_metrics['Fade_F1']:.4f}\n"
                  f" test_metrics['Slope_RMSE']: {test_metrics['Slope_RMSE']:.4f}\n")

            # ==========================
            # 3. 模型保存与早停 (Checkpoints)
            # ==========================
            # 保存当前所有状态 (断点续传用)
            # 准备要保存的数据包 (Checkpoint Dict)

            checkpoint_dict = {
                'epoch': epoch,
                'round_idx': round_idx,  # 建议把轮次也存进去
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                # 👇 新增：保存 pruner 的 mask 👇
                'pruner_mask': pruner.mask if pruner is not None else None,
                # 👇 建议：顺便把对应的特征名也存了，方便推理时直接看名字 👇
                'active_feature_names': pruner.get_active_features() if pruner is not None else None
            }

            torch.save(checkpoint_dict, os.path.join(save_dir, "last_checkpoint.pth"))

            # 每个 epoch 都保存
            ckpt_name = f"model_round_{round_idx:02d}_epoch_{epoch:02d}.pth"
            save_path = os.path.join(save_dir, ckpt_name)

            torch.save(checkpoint_dict, save_path)

            print(f"💾 已保存模型: {ckpt_name}")

            # 保存最佳模型
            # if val_RMSE < best_val_rmse:
            #     best_val_rmse = val_rmse
            #     early_stop_counter = 0

            #     # 构造包含轮次、Epoch和RMSE的唯一文件名
            #     ckpt_name = f"best_model_round_{round_idx:02d}_epoch_{epoch:02d}.pth"
            #     save_path = os.path.join(save_dir, ckpt_name)

            #     # 保存完整字典
            #     torch.save(checkpoint_dict, save_path)
            #     print(f"🏆 发现新最佳模型! RMSE: {val_rmse:.4f} -> 已保存至 {ckpt_name}")
            # else:
            #     early_stop_counter += 1

            # 绘图
            # plot_history(history, save_path=os.path.join(save_dir, "training_log.png"))

            # 早停判断
            # if early_stop_counter >= patience:
            #     print(f"🛑 验证集 Loss 连续 {patience} 轮未下降，触发早停。")
            #     break

    print("训练结束。")
    return history


# 抽取1个Batch进行画图
def plot_test_sample_hasOne(model, test_loader, device, pruner=None):
    model.eval()

    # 1. 抽取一个 Batch
    with torch.no_grad():
        batch = next(iter(test_loader))
        x, y, a, *_ = batch

        x, y, a = x.to(device), y.to(device), a.to(device)
        # =========== 👇 新增逻辑开始 👇 ===========
        if pruner is not None:
            # 如果传入了剪枝器，就用 Mask 过滤掉不需要的大气参数
            # 被 Mask 掉的参数变为 0，相当于从网络中“切断”了
            a = a * pruner.mask
        # =========== 👆 新增逻辑结束 👆 ===========
        # 2. 实时归一化 (必须与训练逻辑完全一致)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std
        print(f"Input x_norm mean: {x_norm.mean()}, std: {x_norm.std()}")
        # 如果 x_norm 本身全是一个数，那输出必然是直线

        print(f"x shape: {x.shape}")
        print(f"x_mean shape: {x_mean.shape}")
        print(f"x_std shape: {x_std.shape}")
        print(f"x_norm shape: {x_norm.shape}")
        # 3. 推理
        preds_norm, _ = model(x_norm, a)
        y_norm = (y - x_mean.squeeze(-1)) / x_std.squeeze(-1)
        print(f"preds_norm shape: {preds_norm.shape}")
        print(f"预测值的 std: {preds_norm.std().item():.4f}")
        print(f"真实值的 std: {y_norm.std().item():.4f}")
        sample_preds = preds_norm[0].detach().cpu().numpy().flatten()

        print("归一化预测值（前10个）:", sample_preds[:10])
        print("预测值的标准差:", sample_preds.std())
        if sample_preds.std() < 1e-6:
            print("🚨 结论：模型输出确实是直线。问题出在模型权重或架构上。")
        else:
            print("✅ 结论：模型输出有波动。直线是由于反归一化逻辑导致的。")
        # 4. 反归一化：将预测值还原到原始量级
        # 公式：preds = preds_norm * x_std + x_mean
        # 注意 y 通常是预测未来的 P 秒，其量级通常参考输入序列 x
        preds = preds_norm * x_std + x_mean
        print(f"preds shape: {preds.shape}")

    # 5. 转为 CPU 绘图
    # 1. 强制检查 combined 维度
    # print(f"Combined shape: {combined.shape}") 应该输出 [Batch, 128]

    # 2. 修正后的绘图取样
    print(f"y_lable shape: {y.shape}")  # 应为 (1000,)
    print(f"y_pred shape: {preds.shape}")  # 应为 (1000,)
    y_pred_2d = preds.squeeze(-1)
    print(f"y_pred_2d shape: {y_pred_2d.shape}")  # 应为 (1000,)
    y_true = y[0].cpu().numpy().flatten()  # 强制压扁成 (1000,)
    y_pred = y_pred_2d[0].cpu().numpy().flatten()  # 强制压扁成 (1000,)
    # 确保数据是一维的
    y_true_flat = y_true
    y_pred_flat = y_pred

    # 1. 计算 MSE (均方误差)
    mse = mean_squared_error(y_true_flat, y_pred_flat)

    # 2. 计算 RMSE (均方根误差)
    rmse = np.sqrt(mse)

    # 3. 计算 R2 (决定系数)
    r2 = r2_score(y_true_flat, y_pred_flat)
    corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

    print(f"📊 评估结果:")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"局部相关系数: {corr:.4f}")

    print(f"y_true shape: {y_true.shape}")  # 应为 (1000,)
    print(f"y_pred shape: {y_pred.shape}")  # 应为 (1000,)
    # 6. 开始绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Power', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predicted Power', color='#ff7f0e', linestyle='--', linewidth=2)

    plt.title(f"Model Prediction vs Actual", fontsize=14)
    plt.xlabel("Time Steps (Seconds)", fontsize=12)
    plt.ylabel("Power Value", fontsize=12)
    # plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_test_sample_noOne(model, test_loader, device):
    model.eval()

    # 1. 抽取一个 Batch
    with torch.no_grad():
        batch = next(iter(test_loader))
        x, y, a, *_ = batch

        x, y, a = x.to(device), y.to(device), a.to(device)

        # 2. 实时归一化 (必须与训练逻辑完全一致)
        print(f"x shape: {x.shape}")
        # 3. 推理
        preds_norm, _ = model(x, a)
        sample_preds = preds_norm[0].detach().cpu().numpy().flatten()

        print("归一化预测值（前10个）:", sample_preds[:10])
        print("预测值的标准差:", sample_preds.std())
        if sample_preds.std() < 1e-6:
            print("🚨 结论：模型输出确实是直线。问题出在模型权重或架构上。")
        else:
            print("✅ 结论：模型输出有波动。直线是由于反归一化逻辑导致的。")
        # 4. 反归一化：将预测值还原到原始量级
        # 公式：preds = preds_norm * x_std + x_mean
        # 注意 y 通常是预测未来的 P 秒，其量级通常参考输入序列 x
        preds = preds_norm
        print(f"preds shape: {preds.shape}")

    # 5. 转为 CPU 绘图
    # 1. 强制检查 combined 维度
    # print(f"Combined shape: {combined.shape}") 应该输出 [Batch, 128]

    # 2. 修正后的绘图取样
    print(f"y_lable shape: {y.shape}")  # 应为 (1000,)
    print(f"y_pred shape: {preds.shape}")  # 应为 (1000,)
    y_pred_2d = preds.squeeze(-1)
    print(f"y_pred_2d shape: {y_pred_2d.shape}")  # 应为 (1000,)
    y_true = y[0].cpu().numpy().flatten()  # 强制压扁成 (1000,)
    y_pred = y_pred_2d[0].cpu().numpy().flatten()  # 强制压扁成 (1000,)
    # 确保数据是一维的
    y_true_flat = y_true
    y_pred_flat = y_pred

    # 1. 计算 MSE (均方误差)
    mse = mean_squared_error(y_true_flat, y_pred_flat)

    # 2. 计算 RMSE (均方根误差)
    rmse = np.sqrt(mse)

    # 3. 计算 R2 (决定系数)
    r2 = r2_score(y_true_flat, y_pred_flat)

    print(f"📊 评估结果:")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    print(f"y_true shape: {y_true.shape}")  # 应为 (1000,)
    print(f"y_pred shape: {y_pred.shape}")  # 应为 (1000,)
    # 6. 开始绘图
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Power', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predicted Power', color='#ff7f0e', linestyle='--', linewidth=2)

    plt.title(f"Model Prediction vs Actual", fontsize=14)
    plt.xlabel("Time Steps (Seconds)", fontsize=12)
    plt.ylabel("Power Value", fontsize=12)
    # plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_evaluate_and_compare_all(model, test_loader, device, pruner=None):
    """
    对比 深度学习模型 与 历史均值基准模型 在全量测试集上的表现
    """
    model.eval()

    # 用于存储所有数据的列表
    all_y_true = []
    all_y_pred_model = []
    all_y_pred_baseline = []

    print("🚀 开始全量测试集评估 (Model vs Baseline)...")

    for batch in tqdm(test_loader, desc="Inference"):
        x, y, a, *_ = batch
        x = x.to(device)
        y = y.to(device)
        a = a.to(device)

        # =========== 👇 新增逻辑开始 👇 ===========
        if pruner is not None:
            # 如果传入了剪枝器，就用 Mask 过滤掉不需要的大气参数
            # 被 Mask 掉的参数变为 0，相当于从网络中“切断”了
            a = a * pruner.mask
        # =========== 👆 新增逻辑结束 👆 ===========
        # ==========================
        # 1. 历史均值模型 (Baseline)
        # ==========================
        # 策略：计算历史输入 x 的均值，作为未来 P 个点的预测值
        # x shape: [Batch, Y, 1] -> mean shape: [Batch, 1, 1]
        x_mean_val = x.mean(dim=1, keepdim=True)
        # 扩展到预测长度 P: [Batch, P, 1] (假设 y 是 [Batch, P])
        # 注意：y 的维度可能是 [Batch, P]，需要匹配
        baseline_preds = x_mean_val.repeat(1, y.shape[1], 1).squeeze(-1)

        all_y_pred_baseline.append(baseline_preds.cpu().numpy())

        # ==========================
        # 2. 深度学习模型 (LSTM)
        # ==========================
        # --- 归一化 (必须与训练一致) ---
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        # 推理
        preds_norm, _ = model(x_norm, a)

        # --- 反归一化 ---
        preds_model = preds_norm * x_std + x_mean
        preds_model = preds_model.squeeze(-1)  # [Batch, P]

        all_y_pred_model.append(preds_model.cpu().numpy())

        # ==========================
        # 3. 存储真实标签
        # ==========================
        all_y_true.append(y.cpu().numpy())

    # --- 数据整合 (Concatenate) ---
    # 将 list 转为巨大的 numpy array，形状通常是 [Total_Samples, P]
    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_model_all = np.concatenate(all_y_pred_model, axis=0)
    y_pred_base_all = np.concatenate(all_y_pred_baseline, axis=0)

    # 展平以便计算全局指标 (Flatten)
    y_true_flat = y_true_all.flatten()
    y_pred_model_flat = y_pred_model_all.flatten()
    y_pred_base_flat = y_pred_base_all.flatten()

    print(f"\n📊 测试集数据总量: {len(y_true_flat)} 个点")

    # ==========================
    # 4. 计算指标 (Metrics)
    # ==========================
    # RMSE
    rmse_model = np.sqrt(mean_squared_error(y_true_flat, y_pred_model_flat))
    rmse_base = np.sqrt(mean_squared_error(y_true_flat, y_pred_base_flat))

    # R2
    r2_model = r2_score(y_true_flat, y_pred_model_flat)
    r2_base = r2_score(y_true_flat, y_pred_base_flat)

    print("\n" + "=" * 40)
    print(f"🥊 最终决斗结果 (全量测试集)")
    print("=" * 40)
    print(f"🔴 基准模型 (历史均值): RMSE = {rmse_base:.5f} | R2 = {r2_base:.5f}")
    print(f"🟢 深度学习 (LSTM模型): RMSE = {rmse_model:.5f} | R2 = {r2_model:.5f}")
    print("-" * 40)

    improvement = (rmse_base - rmse_model) / rmse_base * 100
    if rmse_model < rmse_base:
        print(f"✅ 结论: 你的模型有效！比瞎猜均值提升了 {improvement:.2f}%")
    else:
        print(f"❌ 结论: 模型无效。表现不如直接取均值。")
    print("=" * 40)

    # ==========================
    # 5. 全局可视化 (Visualization)
    # ==========================
    plt.figure(figsize=(18, 12))

    # --- 子图 1: 散点图对比 (真实值 vs 预测值) ---
    # 为了防止点太多卡死，随机采样 10000 个点画散点图
    sample_indices = np.random.choice(len(y_true_flat), size=min(200000, len(y_true_flat)), replace=False)

    plt.subplot(2, 2, 1)
    plt.scatter(y_true_flat[sample_indices], y_pred_base_flat[sample_indices], alpha=0.1, color='red',
                label='Baseline (Mean)', s=1)
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'k--', lw=2)
    plt.title(f"Baseline: Predicted vs Actual\nR2: {r2_base:.4f}")
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.scatter(y_true_flat[sample_indices], y_pred_model_flat[sample_indices], alpha=0.1, color='green',
                label='LSTM Model', s=1)
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'k--', lw=2)
    plt.title(f"Your Model: Predicted vs Actual\nR2: {r2_model:.4f}")
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 子图 2: 时序局部放大图 (拼接前 N 个预测序列) ---
    # 我们把前 5 个样本（即 5 个 Batch 中的序列）取出来拼在一起展示
    # 每个样本长度是 P。假设 P=30 (降采样后)，取前 100 个样本拼接就是 3000 个点

    n_display_samples = 20000  # 展示前 20 条序列
    # 只取前 n_display_samples 个样本进行展平，用于画折线图
    display_true = y_true_all[:n_display_samples].flatten()
    display_pred = y_pred_model_all[:n_display_samples].flatten()
    display_base = y_pred_base_all[:n_display_samples].flatten()

    plt.subplot(2, 1, 2)
    plt.plot(display_true, label='Actual Power', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(display_base, label='Baseline (Mean)', color='red', linestyle='--', linewidth=1, alpha=0.6)
    plt.plot(display_pred, label='LSTM Prediction', color='green', linewidth=1.5, alpha=0.9)

    plt.title(f"Time Series Comparison (First {n_display_samples} sequences flattened)", fontsize=14)
    plt.xlabel("Time Steps (Flattened)", fontsize=12)
    plt.ylabel("Power Value", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# 使用方式：
# evaluate_and_compare_all(model, test_loader, DEVICE)
# ===========================================各种模型的定义==========================================
# ============================================================
# Graph Transformer + LSTM模型定义
# ============================================================
class AdvancedSatelliteFusionModel(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        nodes = nodes + self.layer_pos

        # 垂直图建模
        nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]

        # 时序 Attention 聚合
        attn_score = self.temporal_attn(p_out)  # [B, Y, 1]
        attn_weight = torch.softmax(attn_score, dim=1)
        p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_feature)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # ---- Atmosphere → Power ----
        nodes_proj = self.atm_to_power_proj(nodes)
        context_a2p, attn_a2p = self.cross_attn_a2p(
            p_feature, nodes_proj, nodes_proj
        )
        context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [context_a2p, context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        return preds.unsqueeze(-1), {
            "attn_power_to_atm": attn_p2a,
            "attn_atm_to_power": attn_a2p
        }


# ============================================================
# 2️⃣ Graph Transformer + LSTM模型定义 去掉位置编码（No PosEnc）
# ============================================================
class Ablation_NoPosEnc(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        # nodes = nodes + self.layer_pos

        # 垂直图建模
        nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]

        # 时序 Attention 聚合
        attn_score = self.temporal_attn(p_out)  # [B, Y, 1]
        attn_weight = torch.softmax(attn_score, dim=1)
        p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_feature)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # ---- Atmosphere → Power ----
        nodes_proj = self.atm_to_power_proj(nodes)
        context_a2p, attn_a2p = self.cross_attn_a2p(
            p_feature, nodes_proj, nodes_proj
        )
        context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [context_a2p, context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        return preds.unsqueeze(-1), {
            "attn_power_to_atm": attn_p2a,
            "attn_atm_to_power": attn_a2p
        }


# ============================================================
# 3️⃣ Graph Transformer + LSTM模型定义 去掉垂直 Transformer（No Vertical Graph）
# ============================================================
class Ablation_NoVertical(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        nodes = nodes + self.layer_pos

        # 垂直图建模
        # nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]

        # 时序 Attention 聚合
        attn_score = self.temporal_attn(p_out)  # [B, Y, 1]
        attn_weight = torch.softmax(attn_score, dim=1)
        p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_feature)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # ---- Atmosphere → Power ----
        nodes_proj = self.atm_to_power_proj(nodes)
        context_a2p, attn_a2p = self.cross_attn_a2p(
            p_feature, nodes_proj, nodes_proj
        )
        context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [context_a2p, context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        return preds.unsqueeze(-1), {
            "attn_power_to_atm": attn_p2a,
            "attn_atm_to_power": attn_a2p
        }


# ============================================================
# 4️⃣ Graph Transformer + LSTM模型定义 去掉 Temporal Attention（只取最后一步）
# ============================================================
class Ablation_NoTempAttn(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        nodes = nodes + self.layer_pos

        # 垂直图建模
        nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]
        p_last = p_out[:, -1:, :]
        # # 时序 Attention 聚合
        # attn_score = self.temporal_attn(p_out)   # [B, Y, 1]
        # attn_weight = torch.softmax(attn_score, dim=1)
        # p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_last)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # ---- Atmosphere → Power ----
        nodes_proj = self.atm_to_power_proj(nodes)
        context_a2p, attn_a2p = self.cross_attn_a2p(
            p_last, nodes_proj, nodes_proj
        )
        context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [context_a2p, context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        return preds.unsqueeze(-1), {
            "attn_power_to_atm": attn_p2a,
            "attn_atm_to_power": attn_a2p
        }


# ============================================================
# 5️⃣ Graph Transformer + LSTM模型定义 单向 Cross Attention（去掉双向）
# ============================================================
class Ablation_SingleCross(nn.Module):
    def __init__(self,
                 layer_dims,
                 power_hidden=64,
                 gnn_hidden=64,
                 P=300,
                 num_layers_atm=2):
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.P = P

        assert sum(layer_dims) > 0

        # =====================================================
        # 1️⃣ 异构大气层编码
        # =====================================================
        self.atm_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, gnn_hidden),
                nn.LayerNorm(gnn_hidden),
                nn.ReLU()
            ) for dim in layer_dims
        ])

        # =====================================================
        # 2️⃣ 垂直位置编码（物理高度感知）
        # =====================================================
        self.layer_pos = nn.Parameter(
            torch.randn(1, self.num_layers, gnn_hidden)
        )

        # =====================================================
        # 3️⃣ 垂直 Graph Transformer
        # =====================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gnn_hidden,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

        self.vertical_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers_atm
        )

        # =====================================================
        # 4️⃣ 功率时序分支
        # =====================================================
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=power_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(power_hidden, power_hidden),
            nn.Tanh(),
            nn.Linear(power_hidden, 1)
        )

        # =====================================================
        # 5️⃣ 双向跨模态注意力
        # =====================================================
        self.power_to_atm_proj = nn.Linear(power_hidden, gnn_hidden)
        self.atm_to_power_proj = nn.Linear(gnn_hidden, power_hidden)

        self.cross_attn_p2a = nn.MultiheadAttention(
            embed_dim=gnn_hidden,
            num_heads=4,
            batch_first=True
        )

        self.cross_attn_a2p = nn.MultiheadAttention(
            embed_dim=power_hidden,
            num_heads=4,
            batch_first=True
        )

        # =====================================================
        # 6️⃣ 融合与预测头
        # =====================================================
        fusion_dim = power_hidden + gnn_hidden

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x_norm, a_flat):
        """
        x_norm: [B, Y, 1]
        a_flat: [B, sum(layer_dims)]
        """

        B = x_norm.size(0)
        assert sum(self.layer_dims) == a_flat.shape[1]

        # =====================================================
        # 1️⃣ 大气层拆分编码
        # =====================================================
        atm_nodes = []
        idx = 0

        for i, dim in enumerate(self.layer_dims):
            layer_feat = a_flat[:, idx:idx + dim]  # [B, dim]
            atm_nodes.append(self.atm_encoders[i](layer_feat))
            idx += dim

        nodes = torch.stack(atm_nodes, dim=1)  # [B, L, gnn_hidden]

        # 加物理高度位置编码
        nodes = nodes + self.layer_pos

        # 垂直图建模
        nodes = self.vertical_encoder(nodes)  # [B, L, gnn_hidden]

        # =====================================================
        # 2️⃣ 功率时序建模
        # =====================================================
        p_out, _ = self.lstm(x_norm)  # [B, Y, power_hidden]

        # 时序 Attention 聚合
        attn_score = self.temporal_attn(p_out)  # [B, Y, 1]
        attn_weight = torch.softmax(attn_score, dim=1)
        p_feature = torch.sum(attn_weight * p_out, dim=1, keepdim=True)
        # [B, 1, power_hidden]

        # =====================================================
        # 3️⃣ 双向 Cross Attention
        # =====================================================

        # ---- Power → Atmosphere ----
        query_p = self.power_to_atm_proj(p_feature)
        context_p2a, attn_p2a = self.cross_attn_p2a(
            query_p, nodes, nodes
        )
        context_p2a = context_p2a.squeeze(1)  # [B, gnn_hidden]

        # # ---- Atmosphere → Power ----
        # nodes_proj = self.atm_to_power_proj(nodes)
        # context_a2p, attn_a2p = self.cross_attn_a2p(
        #     p_feature, nodes_proj, nodes_proj
        # )
        # context_a2p = context_a2p.squeeze(1)  # [B, power_hidden]

        # =====================================================
        # 4️⃣ 融合
        # =====================================================
        combined = torch.cat(
            [p_feature.squeeze(1), context_p2a],
            dim=-1
        )  # [B, fusion_dim]

        combined = self.fusion_norm(combined)

        preds = self.regressor(combined)  # [B, P]

        return preds.unsqueeze(-1), {
            "attn_power_to_atm": attn_p2a,
            # "attn_atm_to_power": attn_a2p
        }


# ============================================================
# LSTM 模型
# ============================================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, seq_len=300, P=300):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 利用所有时刻的状态：输入维度 = 序列长度 * 隐藏维度
        # 如果 seq_len 很大，建议先做一次 Pooling 或者只取部分，这里按你的要求全用
        self.regressor = nn.Sequential(
            nn.Linear(seq_len * hidden_size, 256),
            # nn.ReLU(),
            nn.Linear(256, P)
        )

    def forward(self, x, a):
        # x: [Batch, seq_len, 1]

        # 1. 经过 LSTM
        # out 形状: [Batch, seq_len, hidden_size] -> 包含了所有时间步的状态
        out, _ = self.lstm(x)

        # 2. 展平所有状态
        # flattened: [Batch, seq_len * hidden_size]
        flattened = out.reshape(out.size(0), -1)

        # 3. 映射到输出 P
        preds = self.regressor(flattened)  # [Batch, P]

        return preds.unsqueeze(-1), None  # 返回 (Batch, P, 1)


# ==============================================================================
# 基于分解与门控的 LSTM
# ==============================================================================
class AtmosphericGatedLSTM(nn.Module):
    def __init__(self, input_size=1, atm_dim=241, hidden_size=64, num_layers=2, P=300):
        """
        参数:
            input_size: 原始功率维度 (1)
            atm_dim: 大气参数总维度 (例如 241)
            hidden_size: LSTM 隐层
            P: 预测步长
        创新点:
            1. Multi-Scale Decomposition: 内部卷积层自动提取 Trend 和 Detail
            2. Atmospheric Gating: 大气参数生成门控系数，调节 LSTM 特征
        """
        super().__init__()

        # --- 创新模块 1: 多尺度分解 (Learnable or Fixed) ---
        # 使用一个固定权重的平滑卷积来提取 Trend (模拟移动平均)
        # kernel_size=21 意味着约 20秒/点 的平滑窗口
        self.k_size = 21
        self.pad = self.k_size // 2
        self.trend_extractor = nn.AvgPool1d(kernel_size=self.k_size, stride=1, padding=self.pad,
                                            count_include_pad=False)

        # 分解后的输入维度 = Raw(1) + Trend(1) + Detail(1) = 3
        decomposed_input_size = 3

        # --- 骨干网络: LSTM ---
        self.lstm = nn.LSTM(
            input_size=decomposed_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # --- 创新模块 2: 大气门控网络 (Gating Network) ---
        # 输入: 大气特征 -> 输出: 门控系数 (0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(atm_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, hidden_size),  # 输出维度与 LSTM 隐层一致，以便逐元素相乘
            nn.Sigmoid()  # 关键: 限制在 0-1 之间作为“阀门”
        )

        # --- 预测头 ---
        # 输入是 Gated 后的特征
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x, a):
        """
        x: [Batch, Seq, 1] 归一化后的功率历史
        a: [Batch, Atm_Dim] 大气参数
        """
        # ==========================
        # 1. 信号分解 (Decomposition)
        # ==========================
        # AvgPool 需要 [Batch, Channel, Seq] 格式
        x_perm = x.permute(0, 2, 1)  # -> [B, 1, Seq]

        # 提取趋势 (Trend)
        x_trend = self.trend_extractor(x_perm)
        # 由于 Padding 问题，长度可能会有微小变化，强制截断对齐
        if x_trend.shape[-1] != x.shape[1]:
            x_trend = x_trend[..., :x.shape[1]]

        # 提取细节 (Detail / Residual)
        x_detail = x_perm - x_trend

        # 拼接: [Raw, Trend, Detail] -> [B, 3, Seq] -> [B, Seq, 3]
        # x_decomposed = torch.cat([x_perm, x_trend, x_detail], dim=1).permute(0, 2, 1)
        x_decomposed = torch.cat([x_perm, x_trend, x_detail], dim=1).permute(0, 2, 1)

        # ==========================
        # 2. LSTM 特征提取
        # ==========================
        # lstm_out: [Batch, Seq, Hidden]
        lstm_out, _ = self.lstm(x_decomposed)

        # 取最后一个时间步的特征: [Batch, Hidden]
        h_t = lstm_out[:, -1, :]

        # ==========================
        # 3. 大气门控 (Atmospheric Gating)
        # ==========================
        # 计算门控系数: [Batch, Hidden]
        gate = self.gate_net(a)

        # 核心创新公式: Feature = LSTM_Feature * Gate
        # 物理含义: 大气环境决定了我们应该"信任"多少历史特征
        # 或者: 大气环境抑制或增强了特征的表达
        gated_feature = h_t * gate

        # ==========================
        # 4. 预测
        # ==========================
        preds = self.regressor(gated_feature)  # [Batch, P]

        # 返回: [Batch, P, 1], Attention权重(无)
        return preds.unsqueeze(-1), None


# ==============================================================================
# 基于分解与FileMA的 LSTM
# ==============================================================================
class AtmosphericFiLMAttentionLSTM(nn.Module):
    def __init__(self,
                 input_size=1,
                 atm_dim=241,
                 hidden_size=64,
                 num_layers=2,
                 P=300,
                 trend_kernel=21):
        super().__init__()

        self.P = P
        self.hidden_size = hidden_size

        # ==================================================
        # 1️⃣ Learnable Trend Decomposition
        # ==================================================
        self.trend_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=trend_kernel,
            padding=trend_kernel // 2,
            bias=False
        )

        # 初始化为均值滤波
        nn.init.constant_(self.trend_conv.weight,
                          1.0 / trend_kernel)

        decomposed_input_size = 3  # Raw + Trend + Detail

        # ==================================================
        # 2️⃣ LSTM Backbone
        # ==================================================
        self.lstm = nn.LSTM(
            input_size=decomposed_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # ==================================================
        # 3️⃣ Temporal Attention
        # ==================================================
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # ==================================================
        # 4️⃣ Atmospheric FiLM Modulation
        # ==================================================
        self.film_net = nn.Sequential(
            nn.Linear(atm_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, hidden_size * 2)
        )

        # ==================================================
        # 5️⃣ Deep Fading State Branch（可选）
        # ==================================================
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # ==================================================
        # 6️⃣ Regression Head
        # ==================================================
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, P)
        )

    def forward(self, x, a):
        """
        x: [Batch, Seq, 1]
        a: [Batch, Atm_Dim]
        """

        B, T, _ = x.shape

        # ==================================================
        # 1️⃣ Decomposition
        # ==================================================
        x_perm = x.permute(0, 2, 1)  # [B, 1, T]

        trend = self.trend_conv(x_perm)

        if trend.shape[-1] != T:
            trend = trend[..., :T]

        detail = x_perm - trend

        x_decomp = torch.cat(
            [x_perm, trend, detail],
            dim=1
        ).permute(0, 2, 1)  # [B, T, 3]

        # ==================================================
        # 2️⃣ LSTM
        # ==================================================
        lstm_out, _ = self.lstm(x_decomp)  # [B, T, H]

        # ==================================================
        # 3️⃣ Temporal Attention
        # ==================================================
        attn_score = self.attn_layer(lstm_out)  # [B, T, 1]
        attn_weight = torch.softmax(attn_score, dim=1)

        h_t = torch.sum(attn_weight * lstm_out, dim=1)  # [B, H]

        # ==================================================
        # 4️⃣ Atmospheric FiLM
        # ==================================================
        gamma_beta = self.film_net(a)  # [B, 2H]
        gamma, beta = gamma_beta.chunk(2, dim=1)

        gamma = torch.sigmoid(gamma)

        h_mod = gamma * h_t + beta

        # ==================================================
        # 5️⃣ Deep Fading Probability
        # ==================================================
        fading_prob = self.state_head(h_mod)

        # ==================================================
        # 6️⃣ Prediction
        # ==================================================
        preds = self.regressor(h_mod)  # [B, P]

        return preds.unsqueeze(-1), fading_prob


# ==============================================================================
# ✨ 核心工具: 掩码梯度剪枝器 (Masked Gradient Pruner)
# ==============================================================================
class MaskedGradientPruner:
    def __init__(self, feature_names, device):
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.device = device

        # 初始化 Mask，全为 1 (代表所有特征一开始都保留)
        self.mask = torch.ones(1, self.num_features).to(device)

    def get_active_features(self):
        """返回当前还活着的特征名字"""
        mask_np = self.mask[0].cpu().numpy()
        active_names = [self.feature_names[i] for i in range(self.num_features) if mask_np[i] == 1]
        return active_names

    def compute_saliency(self, model, dataloader, criterion):
        """
        计算梯度敏感度，决定谁该被置 0
        """
        # 🚨 修正点：必须开启 train 模式，否则 CuDNN LSTM 无法反向传播
        model.train()

        importance_score = torch.zeros(self.num_features).to(self.device)

        # 只遍历少量 Batch 即可，节省时间
        max_batches = 5000
        batch_count = 0

        print("🔍 正在分析大气参数重要性...", end="")
        for batch in dataloader:
            if batch_count >= max_batches: break

            x, y, a_full, *_ = batch
            x = x.to(self.device)
            y = y.to(self.device)
            a_full = a_full.to(self.device)

            # --- 关键 1: 允许求导 ---
            # 必须新建一个叶子节点或者 detach 出来，否则可能会报错
            a_full = a_full.clone().detach()
            a_full.requires_grad = True

            # --- 关键 2: 应用当前的 Mask ---
            a_masked = a_full * self.mask

            # --- 关键 3: 归一化 x ---
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std

            # 前向传播
            preds_norm, _ = model(x_norm, a_masked)

            # 反归一化算 Loss
            # 🚨 修正点：修复之前的广播维度错误
            preds = preds_norm.squeeze(-1) * x_std.squeeze(-1) + x_mean.squeeze(-1)
            loss = criterion(preds, y)

            # 清空模型之前的梯度（虽然不更新权重，但防止累积）
            model.zero_grad()

            # 反向传播 (此时 model.train() 模式下 CuDNN 支持 backward)
            loss.backward()

            # --- 关键 4: 获取梯度 ---
            # 梯度越大，说明该大气参数对 Loss 影响越大
            if a_full.grad is not None:
                grads = torch.abs(a_full.grad).mean(dim=0)
                importance_score += grads

            batch_count += 1

        print("完成")
        return importance_score

    def prune_step(self, importance_scores, num_to_remove):
        """
        将重要性最低的 N 个位置的 Mask 设为 0
        返回: 包含 (特征名, 得分) 的列表
        """
        current_mask = self.mask[0]

        # 将已删除特征的得分设为无限大，防止重复删除
        adjusted_scores = torch.where(
            current_mask == 1,
            importance_scores,
            torch.tensor(float('inf')).to(self.device)
        )

        # 找出最小的 N 个
        values, indices = torch.topk(adjusted_scores, k=num_to_remove, largest=False)

        # 将 Mask 对应位置置 0
        self.mask[0, indices] = 0

        # 获取被移除特征的信息
        removed_idx = indices.cpu().numpy()
        removed_values = values.cpu().numpy()  # 获取得分

        # 组装返回信息：[(name, score), (name, score)...]
        removed_info = []
        for i in range(len(removed_idx)):
            name = self.feature_names[removed_idx[i]]
            score = removed_values[i]
            removed_info.append((name, score))

        return removed_info


# 保存CSV文件
class PaperDataRecorder:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 定义文件名
        self.files = {
            "pruning_curve": os.path.join(save_dir, "fig1_pruning_curve.csv"),
            "predictions": os.path.join(save_dir, "fig2_predictions_sample.csv"),
            "feature_rank": os.path.join(save_dir, "fig4_feature_rank.csv"),
            "metrics_history": os.path.join(save_dir, "tab1_metrics_history.csv")
        }

        # 初始化 CSV头 (👉 修改：增加了 W_Dist 相关列)
        self._init_csv("pruning_curve", [
            "Round", "Num_Features",
            "Val_RMSE", "Val_Fade_RMSE", "Val_W_Dist",
            "Test_RMSE", "Test_Fade_RMSE", "Test_W_Dist"
        ])

        # metrics_history 也可以加上 W_Dist，不过 log_metrics 是直接 dump dataframe，
        # 只要 dataframe 里有(我们在 train 函数里加了)，这里自动会有，不用改 header
        # self._init_csv("metrics_history", ...)

    def _init_csv(self, key, header):
        if not os.path.exists(self.files[key]):
            with open(self.files[key], mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    # 👉 修改：增加了 val_w 和 test_w 参数
    # def log_pruning_step(self, round_idx, num_features, val_rmse, val_fade_rmse, val_w, test_rmse, test_fade_rmse,
    #                      test_w):
    #     """对应 图1：记录每一轮剪枝后的性能变化"""
    #     with open(self.files["pruning_curve"], mode='a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             round_idx, num_features,
    #             val_rmse, val_fade_rmse, val_w,
    #             test_rmse, test_fade_rmse, test_w
    #         ])
    def log_pruning_step(self, round_idx, num_features, val_metrics, test_metrics):
        """
        记录每一轮剪枝后的性能变化
        val_metrics / test_metrics 都是字典
        """

        with open(self.files["pruning_curve"], mode='a', newline='') as f:
            writer = csv.writer(f)

            # 如果是第一次写入，可以自动写表头（推荐）
            if f.tell() == 0:
                header = (
                        ["round_idx", "num_features"]
                        + [f"val_{k}" for k in val_metrics.keys()]
                        + [f"test_{k}" for k in test_metrics.keys()]
                )
                writer.writerow(header)

            row = (
                    [round_idx, num_features]
                    + list(val_metrics.values())
                    + list(test_metrics.values())
            )

            writer.writerow(row)

    def log_metrics(self, history, round_idx):
        """记录详细训练曲线"""
        import pandas as pd
        df = pd.DataFrame(history)
        df['Round'] = round_idx

        # 如果这是第一次写入，包含 header；否则不包含
        header = not os.path.exists(self.files["metrics_history"])
        df.to_csv(self.files["metrics_history"], mode='a', header=header, index=False)

    def save_predictions_for_plot(self, y_true, y_pred, y_pred_baseline=None):
        import pandas as pd
        data = {
            "True_Power": y_true.flatten(),
            "Pred_Power": y_pred.flatten()
        }
        if y_pred_baseline is not None:
            data["Baseline_Power"] = y_pred_baseline.flatten()

        df = pd.DataFrame(data)
        df.to_csv(self.files["predictions"], index=False)
        print(f"📈 预测数据已保存至: {self.files['predictions']}")

    def save_feature_importance(self, feature_names, scores, mask):
        import pandas as pd
        active_indices = torch.nonzero(mask.cpu()).flatten().numpy()

        data = []
        for idx in active_indices:
            data.append({
                "Feature": feature_names[idx],
                "Score": scores[idx].item()
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by="Score", ascending=False)
        df.to_csv(self.files["feature_rank"], index=False)
        print(f"📊 特征重要性已保存至: {self.files['feature_rank']}")


def get_loss_function(loss_name, device):
    if loss_name == "RMSE":
        return rmse

    elif loss_name == "W1":
        return wasserstein_1d

    elif loss_name == "W2":
        return wasserstein_2d

    elif loss_name == "TailMSE":
        return tail_loss

    elif loss_name == "TailW1":
        return tail_wasserstein_loss

    elif loss_name == "FadeDist":
        return fading_distribution_loss

    elif loss_name == "FadeTailDist":
        return fading_tail_distribution_loss

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def build_model(model_name,
                layer_dims,
                seq_len=300,
                P=300,
                atm_dim=None,
                device="cuda"):
    if atm_dim is None:
        atm_dim = sum(layer_dims)

    if model_name == "Fusion":  # Advanced Graph Transformer
        model = AdvancedSatelliteFusionModel(
            layer_dims=layer_dims,
            power_hidden=64,
            gnn_hidden=64,
            P=P,
            num_layers_atm=2
        )

    elif model_name == "NoPosEnc":
        model = Ablation_NoPosEnc(
            layer_dims=layer_dims,
            P=P
        )

    elif model_name == "NoVertical":
        model = Ablation_NoVertical(
            layer_dims=layer_dims,
            P=P
        )

    elif model_name == "NoTempAttn":
        model = Ablation_NoTempAttn(
            layer_dims=layer_dims,
            P=P
        )

    elif model_name == "SingleCross":
        model = Ablation_SingleCross(
            layer_dims=layer_dims,
            P=P
        )

    elif model_name == "SimpleLSTM":
        model = SimpleLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            seq_len=seq_len,
            P=P
        )

    elif model_name == "AGLSTM":
        model = AtmosphericGatedLSTM(
            input_size=1,
            atm_dim=atm_dim,
            hidden_size=64,
            num_layers=2,
            P=P
        )

    elif model_name == "FiLMLSTM":
        model = AtmosphericFiLMAttentionLSTM(
            input_size=1,
            atm_dim=atm_dim,
            hidden_size=64,
            num_layers=2,
            P=P
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)


def reset_environment(seed=42):
    # 1️⃣ 删除缓存
    gc.collect()
    torch.cuda.empty_cache()

    # 2️⃣ 固定随机种子（保证可复现）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def has_files(dir_path):
    if os.path.exists(dir_path):
        # 使用 os.scandir 高效检查是否有文件
        for entry in os.scandir(dir_path):
            if entry.is_file():
                return True
    return False
def count_files(dir_path):
    """
    统计文件夹中的文件数量（不包含子文件夹）
    """
    if not os.path.exists(dir_path):
        return 0
    
    file_count = 0
    for entry in os.scandir(dir_path):
        if entry.is_file():
            file_count += 1
    return file_count
if __name__ == "__main__":
    # ==========================================
    # 0. 初始化环境与参数
    print(f"0. 初始化环境与参数")
    # ==========================================
    # Yhis, Phis = 600, 600  # 点数
    BATCH_SIZE = 32 * 2
    EPOCHS = 5
    LEARNING_RATE = 1e-3 * 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Yhiss = [100, 300,   900 ]
    Yhiss = [100]
    Phiss = [100, 300, 900, 1500, 2100, 2700]
    # Yhiss = [300] #[300, 900, 1500, 2100, 2700]
    # Phiss = [300] #[300, 900, 1500, 2100, 2700]
    # rates = [60, 30, 10, 1]   # 1,2,5,10,15,20,30,60，每rate个点取平均
    rates = [10]
    # models = ["SimpleLSTM","AGLSTM","FiLMLSTM","Fusion","NoPosEnc","NoVertical","NoTempAttn","SingleCross"]
    models = ["Fusion","NoPosEnc","NoVertical","NoTempAttn","SingleCross","SimpleLSTM","AGLSTM","FiLMLSTM"]# 

    losses = ["FadeDist", "RMSE", "W1", "W2", "TailMSE","FadeTailDist", "TailW1",]  #

    ERA_FILES = [
        "era5_pressure_merged_1hPa_filtered_normalized",
        "era5_pressure_merged_5hPa_filtered_normalized",
        "era5_pressure_merged_30hPa_filtered_normalized",
        "era5_pressure_merged_200hPa_filtered_normalized",
        "era5_pressure_merged_650hPa_filtered_normalized",
        "era5_surface_merged_all_filter_normalized",
    ]
    for Yhis in Yhiss:
        for Phis in Phiss:
            if Yhis + Phis <= 3600 and Yhis <= Phis:
                print(f"history input data(s)={Yhis}, predictions data(s)={Phis}")
                for rate in rates:
                    print(f"rate={rate}")
                    Y = Yhis // rate
                    P = Phis // rate
                    if Y >= 10 and P >= 10:
                        for modelmy in models:
                            for lossmy in losses:
                                print(f"清理环境和上一个模型残留")
                                reset_environment(seed=42)
                                print(f"当前模型: {modelmy}, 当前损失: {lossmy}")
                                # ==========================================
                                # 1. 文件夹路径定义
                                print(f"1. 文件夹路径定义")
                                # ==========================================

                                # 获取当前文件的绝对路径
                                # current_file = os.path.abspath(__file__)
                                # print(f"当前文件路径: {current_file}")
                                # 获取当前文件所在文件夹
                                # current_dir = os.path.dirname(current_file)
                                # print(f"当前文件夹: {current_dir}")
                                current_dir = os.getcwd()
                                # current_dir = "/content/drive/MyDrive/Colab Notebooks"
                                BASE_DIR = current_dir  # "/WORKSPACE"
                                print(f"当前工作区文件夹: {BASE_DIR}，之后所有子文件夹都在这个工作区下。")
                                SPLIT_DIR = f"{BASE_DIR}/splits"
                                SAVE_DIR = f"{BASE_DIR}/AWPL_checkpoints_{Yhis}_{Phis}_{rate}_{modelmy}_{lossmy}"
                                LOG_DIR = f"{BASE_DIR}/AWPL_pruning_logs_{Yhis}_{Phis}_{rate}_{modelmy}_{lossmy}"
                                PAPER_DATA_DIR = f"{BASE_DIR}/AWPL_paper_data_{Yhis}_{Phis}_{rate}_{modelmy}_{lossmy}"
                                
                                # 检查两个目录是否有文件
                                save_has_files = has_files(SAVE_DIR)
                                log_has_files = has_files(LOG_DIR)
                                filesNum  =  count_files(SAVE_DIR)
                                if save_has_files:
                                    print(f"⚠️ 发现文件夹 {SAVE_DIR} 存在并有文件")
                                    if filesNum==10:
                                        print(f"⚠️ 跳过此循环，因为存在训练好的10个.pth文件")
                                        continue  # 跳过当前循环，执行下一个
                                    print(f"！！ 继续此循环，因为存在的文件不足10个")
                                # if os.path.exists(SAVE_DIR) and os.path.exists(LOG_DIR):
                                #     print(f"⚠️ 跳过此循环，因为文件已存在: {SAVE_DIR},{LOG_DIR}")
                                #     continue  # 跳过当前循环，执行下一个

                                # 确保保存目录存在
                                os.makedirs(SPLIT_DIR, exist_ok=True)  # 训练使用到的数据集文件保存路径
                                os.makedirs(SAVE_DIR, exist_ok=True)  # 中间训练的pth文件保存路径
                                os.makedirs(LOG_DIR, exist_ok=True)  # 训练的每次循环的日志文件保存路径
                                os.makedirs(PAPER_DATA_DIR, exist_ok=True)  # 训练好的模型测试集结果和真实值的所有功率点的文件保存路径
                                
                                
                                # ==========================================
                                # 2. 实例化数据加载器 (IterableDataset)
                                print(f"2. 实例化数据加载器 (DataLoader)")
                                # ==========================================
                                train_ds = PowerAtmSlidingDatasetRate(split="train", Y=Yhis, P=Phis, rate=rate)
                                val_ds = PowerAtmSlidingDatasetRate(split="val", Y=Yhis, P=Phis, rate=rate)
                                test_ds = PowerAtmSlidingDatasetRate(split="test", Y=Yhis, P=Phis, rate=rate)
                                # 注意：IterableDataset 不要设置 shuffle=True
                                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
                                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
                                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

                                # ==========================================
                                # 3. 初始化剪枝操作
                                print(f"3. 初始化剪枝操作")
                                # ==========================================
                                # 1. 获取所有特征名 (用于打印日志)
                                # 这里的 feature_names 是 load_era_dict 返回的
                                # 你需要从 dataset 中获取，或者重新调用一次 load_era_dict
                                _, all_feature_names = load_era_dict("train")
                                total_atm_dim = len(all_feature_names)

                                # ==========================================
                                # 4. 实例化模型、损失函数与优化器
                                print(f"4. 实例化模型、损失函数与优化器")
                                # ==========================================
                                # 自动探测大气特征维度
                                actual_layer_dims = get_layer_dims(SPLIT_DIR, ERA_FILES)
                                print(f"探测到大气层级维度: {actual_layer_dims}")
                                '''
                                # model = SatelliteFusionModel(
                                #     layer_dims=actual_layer_dims,
                                #     power_hidden=64,
                                #     gnn_hidden=64,
                                #     P=P
                                # ).to(DEVICE)
                                '''

                                print(f"实例化 {modelmy} 模型")
                                # 根据名字实例化模型
                                model = build_model(
                                    model_name=modelmy,
                                    layer_dims=actual_layer_dims,
                                    seq_len=Y,
                                    P=P,
                                    device=DEVICE
                                )
                                print(f"具体模型信息： {model.named_modules}")
                                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                print(f"Trainable parameters: {total_params:,}")
                                print(f"实例化模型 完毕")

                                print(f"选择 {lossmy} 损失函数")
                                # 根据名字选择损失函数
                                criterion = get_loss_function(lossmy, DEVICE)

                                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                                print(f"实例化优化器optimizer")

                                pruner = MaskedGradientPruner(all_feature_names, DEVICE)
                                print(f"实例化剪枝器")

                                # ==========================================
                                # 5. 正式执行训练循环
                                print(f"5. 正式执行训练循环")

                                # ==========================================
                                print(f"\n🚀 正式开始训练 | 设备: {DEVICE} | 批次大小: {BATCH_SIZE}")

                                TARGET_FEATURES = 1  # 期望最后剩余多少大气参数
                                REMOVE_PER_ROUND = 117  # 每轮砍掉 REMOVE_PER_ROUND 个 (前期可以砍快点)
                                # 一共12轮，第一轮5次，剩下1次，一共16次，慢的话2小时，快的话半小时
                                round_idx = 0  # 初始化大气参数剪枝轮次
                                recorder = PaperDataRecorder(PAPER_DATA_DIR)  # 初始化记录训练过程的类
                                while True:
                                    current_active = int(pruner.mask.sum().item())
                                    round_idx += 1

                                    print(f"\n\n========================================================")
                                    print(
                                        f"♻️  大气特征剪枝轮次 Pruning Round {round_idx} | 当前保留特征数: {current_active}")
                                    print(f"========================================================")

                                    # A. 训练 (Fine-tuning)
                                    # 第一轮（round_idx == 1）训练久一点(current_epochs)，后面只需微调(5 Epoch)让权重适应
                                    current_epochs = 1 if round_idx == 1 else 1
                                    # 调用之前定义的通用训练引擎
                                    history = train_satellite_model_pruner(
                                        model=model,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        test_loader=test_loader,
                                        optimizer=optimizer,
                                        criterion=criterion,
                                        device=DEVICE,
                                        epochs=current_epochs,
                                        save_dir=SAVE_DIR,
                                        eval_freq=1,  # 每个 epoch 验证一次
                                        patience=10,  # 10轮不进步则早停
                                        pruner=pruner,
                                        round_idx=round_idx
                                    )
                                    # 记录详细训练日志 ===
                                    recorder.log_metrics(history, round_idx)
                                    # 获取本轮最佳指标 (取最后几个 Epoch 的平均或最佳)

                                    # 确保这里传入的参数数量和顺序与上面定义的 log_pruning_step 一致，增加了当前有效参数个数
                                    print(f"\nSaving... 保存每一轮剪枝操作的模型评估参数结果，随着轮次增加不断写入")

                                    history_df = pd.DataFrame(history)
                                    # 加上轮次和Epoch信息，方便以后合并分析
                                    history_df['Round'] = round_idx
                                    history_df['Epoch'] = range(1, len(history_df) + 1)
                                    # 调整列顺序，把 Round 和 Epoch 放到最前面
                                    cols = ['Round', 'Epoch'] + [c for c in history_df.columns if
                                                                 c not in ['Round', 'Epoch']]
                                    history_df = history_df[cols]
                                    # 保存到 pruning_logs 文件夹
                                    print(
                                        f"\nSaving... 保存第{round_idx:02d}轮中所有子轮次剪枝操作的模型评估参数结果，每个轮次对应一个csv文件")
                                    hist_file = os.path.join(LOG_DIR, f"history_round_{round_idx:02d}.csv")
                                    history_df.to_csv(hist_file, index=False)
                                    # print(f"📈 本轮训练曲线数据已保存至: {hist_file}")

                                    # B. 检查是否达到目标
                                    if current_active <= TARGET_FEATURES:
                                        print("✅ 已达到目标特征数，停止剪枝。")
                                        break

                                    # C. 计算敏感度
                                    importance = pruner.compute_saliency(model, train_loader, criterion)

                                    # D. 执行剪枝
                                    # 动态调整步长：快结束时砍慢点
                                    if current_active < 10: REMOVE_PER_ROUND = 1

                                    if current_active <= 10:  # 比如剩20个的时候存一次
                                        recorder.save_feature_importance(all_feature_names, importance, pruner.mask[0])

                                    # 👈 修改：接收详细信息
                                    removed_info = pruner.prune_step(importance, num_to_remove=REMOVE_PER_ROUND)

                                    # 👈 修改：打印详细原因
                                    print(f"✂️  本轮移除 {len(removed_info)} 个特征 (原因是梯度敏感度得分最低):")
                                    for name, score in removed_info:
                                        print(f"   ❌ [移除] 得分: {score:.6e} | 特征: {name}")

                                    # ==========================================
                                    # E. 保存剪枝过程日志 (核心需求)
                                    # ==========================================
                                    # 我们把这一轮所有特征的得分和状态保存下来
                                    log_data = []

                                    # 获取当前所有的重要性得分 (转numpy)
                                    all_scores = importance.detach().cpu().numpy()
                                    # 获取当前的 mask (转numpy)
                                    current_mask_np = pruner.mask[0].cpu().numpy()

                                    for i, name in enumerate(all_feature_names):
                                        status = "Active" if current_mask_np[i] == 1 else "Removed"
                                        # 标记一下是不是这一轮刚被删的
                                        is_just_removed = any(name == r_name for r_name, _ in removed_info)
                                        if is_just_removed:
                                            status = "Just_Removed"

                                        log_data.append({
                                            "Round": round_idx,
                                            "Feature_Index": i,
                                            "Feature_Name": name,
                                            "Saliency_Score": all_scores[i],
                                            "Status": status
                                        })

                                    # 保存到 CSV
                                    df_log = pd.DataFrame(log_data)
                                    log_file = os.path.join(LOG_DIR, f"pruning_log_round_{round_idx:02d}.csv")
                                    df_log.to_csv(log_file, index=False)
                                    print(f"Saving... 本轮剪枝详情已保存至: {log_file}")

                                # ==========================================
                                # 6. 最终结果分析
                                print(f"6. 最终结果分析")
                                # ==========================================
                                final_features = pruner.get_active_features()
                                print(f"\n🎉 最终筛选出的 {len(final_features)} 个黄金特征:")
                                for f in final_features:
                                    print(f"  - {f}")

                                # 跑一次全量测试集，保存预测和真实值对比结果 ===
                                # 这一步是为了生成高质量的对比图
                                # print(f"根据训练完成的模型，跑一次全部的测试集，并记录全部的预测值和真实值")
                                # model.eval()
                                # all_true, all_pred = [], []
                                # with torch.no_grad():
                                #     for batch in test_loader:
                                #         x, y, a, *_ = batch
                                #         x, y, a = x.to(DEVICE), y.to(DEVICE), a.to(DEVICE)
                                #         a = a * pruner.mask  # 别忘了 mask

                                #         x_mean = x.mean(dim=1, keepdim=True)
                                #         x_std = x.std(dim=1, keepdim=True) + 1e-8
                                #         x_norm = (x - x_mean) / x_std
                                #         preds_norm, _ = model(x_norm, a)
                                #         preds = preds_norm.squeeze(-1) * x_std.squeeze(-1) + x_mean.squeeze(-1)

                                #         all_true.append(y.cpu().numpy())
                                #         all_pred.append(preds.cpu().numpy())

                                # y_true_final = np.concatenate(all_true, axis=0)
                                # y_pred_final = np.concatenate(all_pred, axis=0)

                                # recorder.save_predictions_for_plot(y_true_final, y_pred_final)

                                # === 修改 5: 保存最终特征排名 (图4数据) ===
                                # 重新计算一次最终 Importance
                                final_imp = pruner.compute_saliency(model, train_loader, criterion)
                                recorder.save_feature_importance(all_feature_names, final_imp, pruner.mask[0])
                                #plot_test_sample_hasOne(model, test_loader, DEVICE, pruner=pruner)
                                #plot_evaluate_and_compare_all(model, test_loader, DEVICE, pruner=pruner)
                                '''
                                清空模型的权重值，再开启下一轮循环
                                '''
                                del model
                                del optimizer
                                torch.cuda.empty_cache()
                                gc.collect()
    # print(f"✅ 论文所需数据全部保存在: {PAPER_DATA_DIR}")

    # plot_test_sample_hasOne(model, test_loader, DEVICE, pruner=pruner)
    # plot_evaluate_and_compare_all(model, test_loader, DEVICE, pruner=pruner)

