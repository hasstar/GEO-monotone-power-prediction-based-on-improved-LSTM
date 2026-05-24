import torch
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import IterableDataset
import os
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
# ====== 你自己的模块 ======
# from your_model_file import build_model
# from your_dataset_file import PowerAtmSlidingDatasetRate
# from your_utils import get_layer_dims
import re
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

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)
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

#
# 第一行：times[i]- 滑动窗口的起始时间
# 第二行：times[i + self.Y - 1]- 历史窗口结束时间
# 第三行：times[i + self.Y + self.P - 1]- 完整窗口结束时间（包含预测）
# 第四行：hour- 聚合的小时基准
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

all_model_results = []
# 文件夹路径
folder_path = "VAPpth"  # 🔴 你的文件夹名
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.pth'):  # 只处理 .pth 文件
        MODEL_PATH = f"{folder_path}./{filename}" # 🔴 改这里
        MODEL_NAME = "Fusion"

        # 提取所有数字
        numbers = re.findall(r'\d+', MODEL_PATH)
        print(MODEL_PATH)
        print(f"所有数字: {numbers}")  # ['300', '900', '30', '2']

        # 获取第二个数字（索引1）
        target_number = numbers[1] if len(numbers) > 1 else None
        target_number = int(target_number)
        Yhis = 300
        Phis = target_number
        rate = 30

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Y = Yhis // rate
        P = Phis // rate

        # ==========================================
        # 1. 构建模型
        # ==========================================
        ERA_FILES = [
            "era5_pressure_merged_1hPa_filtered_normalized",
            "era5_pressure_merged_5hPa_filtered_normalized",
            "era5_pressure_merged_30hPa_filtered_normalized",
            "era5_pressure_merged_200hPa_filtered_normalized",
            "era5_pressure_merged_650hPa_filtered_normalized",
            "era5_surface_merged_all_filter_normalized",
        ]

        SPLIT_DIR = "./splits"

        layer_dims = get_layer_dims(SPLIT_DIR, ERA_FILES)

        model = build_model(
            model_name=MODEL_NAME,
            layer_dims=layer_dims,
            seq_len=Y,
            P=P,
            device=DEVICE
        ).to(DEVICE)

        # ==========================================
        # 2. 加载权重
        # ==========================================
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ 模型权重加载成功")
        model.eval()

        # ==========================================
        # 3. 数据加载
        # ==========================================
        test_ds = PowerAtmSlidingDatasetRate(
            split="test",
            Y=Yhis,
            P=Phis,
            rate=rate
        )

        test_loader = DataLoader(test_ds, batch_size=32)

        print("✅ 数据加载完成")


        # ==========================================
        # 4. 推理函数
        # ==========================================
        def normalize(x):
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
            return x_norm, x_mean, x_std


        # ==========================================
        # 5. 推理
        # ==========================================
        all_preds = []
        all_trues = []
        found_target = False

        batch_count = 0
        start_batch_idx = -1

        target_batch_idx = 135  # 🔴 想看第几个batch
        num_batches_to_combine = 5
        # target_time = '2025-12-08T15:34:30.000000000'
        num_batches_after = 1000

        target_time = '2025-12-12'  # 🔴 起始时间条件
        target_end_time = '2025-12-14'  # 🔴 结束时间条件
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x, y, a, l1, l2, l3, l4 = batch

                # 🔴 检查是否找到目标时间
                if not found_target:
                    if target_time in str(l1[0]):
                        print(l1[0])
                        found_target = True
                        start_batch_idx = i
                        print(f"✅ 在batch {i} 找到目标时间: {target_time}")
                        print(f"开始往后处理 {num_batches_after} 个batch...")
                    else:
                        continue  # 跳过，直到找到目标

                # 🔴 2. 检查是否到达结束时间
                # 检查这个batch的第一个样本是否超过结束时间
                first_time_str = str(l1[0])
                if target_end_time in first_time_str:
                    print(f"⏹️ 到达结束时间 {target_end_time}，停止处理")
                    print(f"最后一个batch: {i}, 第一个样本: {first_time_str}")
                    break
                # if i != target_batch_idx:
                #     continue

                if found_target:
                    print(f"处理第 {batch_count + 1} 个batch (索引: {i})")
                    print(f"  时间: {l1[0]}")
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    a = a.to(DEVICE)

                    x_norm, x_mean, x_std = normalize(x)
                    preds_norm, _ = model(x_norm, a)
                    preds = preds_norm.squeeze(-1) * x_std.squeeze(-1) + x_mean.squeeze(-1)

                    all_preds.append(preds.cpu().numpy())
                    all_trues.append(y.cpu().numpy())

                    # 显示这个batch的时间信息
                    print(f"  Batch {i}: 第一个样本开始时间 = {l1[0]}")
                    print(f"           样本数量 = {len(l1)}")

                    batch_count += 1

        # ==========================================
        # 6. 结果
        # ==========================================
        # y_pred = np.concatenate(all_preds, axis=0)
        # y_true = np.concatenate(all_true, axis=0)

        if not found_target:
            print(f"⚠️ 没有找到起始时间 {target_time}")
        else:
            # 合并结果
            if all_preds:
                y_pred = np.concatenate(all_preds, axis=0)
                y_true = np.concatenate(all_trues, axis=0)

                print(f"总样本数: {y_pred.shape[0]}")
                print(f"预测形状: {y_pred.shape}")
                print(f"真实形状: {y_true.shape}")

                y_pred = y_pred.flatten()  # 形状: (2880,)
                y_true = y_true.flatten()  # 形状: (2880,)

                # y_pred = y_pred[0]  # 形状: (2880,)
                # y_true = y_true[0]  # 形状: (2880,)

                print(f"\n✅ 处理完成!")
                print(f"起始batch索引: {start_batch_idx}")
                print(f"处理batch数量: {len(all_preds)}")
                # 保存到总结果列表
                all_model_results.append({
                    'model_name': filename,
                    'ph': Phis,  # 预测长度
                    'predictions': y_pred,
                    'targets': y_true,
                    'batch_count': batch_count,
                    'sample_count': len(y_pred)

                })

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 创建2x3的子图布局
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
axes = axes.flatten()

# 设置字母标签
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
sorted_results = sorted(all_model_results, key=lambda x: x.get('ph', 0))

# 遍历所有模型
for idx, result in enumerate(sorted_results[:5]):  # 只取前5个
    ax = axes[idx]
    model_name = result['model_name']
    y_pred = result['predictions']
    y_true = result['targets']
    ph_value = result.get('ph', 'N/A')

    # 现场计算CDF和KS
    # 排序
    true_sorted = np.sort(y_true)
    pred_sorted = np.sort(y_pred)

    # 计算CDF
    true_cdf = np.arange(len(true_sorted)) / len(true_sorted)
    pred_cdf = np.arange(len(pred_sorted)) / len(pred_sorted)

    # 统一x轴插值
    x = np.linspace(
        min(true_sorted.min(), pred_sorted.min()),
        max(true_sorted.max(), pred_sorted.max()),
        1000
    )

    true_cdf_interp = np.interp(x, true_sorted, true_cdf)
    pred_cdf_interp = np.interp(x, pred_sorted, pred_cdf)

    # 计算KS值
    ks_value = np.max(np.abs(true_cdf_interp - pred_cdf_interp))
    ks_idx = np.argmax(np.abs(true_cdf_interp - pred_cdf_interp))

    # 绘制CDF
    ax.plot(x, true_cdf_interp, label='True CDF', linewidth=2, color='blue')
    ax.plot(x, pred_cdf_interp, label='Pred CDF', linewidth=2, color='orange')

    # 标记KS点
    ax.scatter(x[ks_idx], true_cdf_interp[ks_idx],
               color='red', s=100, zorder=5,
               label=f'KS={ks_value:.6f}')

    # 设置子图属性
    ax.set_title(f'{subplot_labels[idx]}$P$={ph_value}, KS={ks_value:.6f}', fontsize=16, y=-0.15, pad=00)
    ax.set_xlabel('Value', fontsize=14)
    ax.set_ylabel('CDF', fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

# 隐藏多余的空子图
if len(all_model_results) < 6:
    axes[5].set_visible(False)

# plt.suptitle('CDF Comparison and KS Statistic for Different Models', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

plt.savefig('cdf_ks_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()













# # 创建5行2列的子图
# fig, axes = plt.subplots(5, 2, figsize=(15, 20))
#
# # 设置字母标签
# letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
#
# # 绘制每个模型的两张图
# for idx, result in enumerate(all_model_results):
#     if idx >= 5:
#         break
#
#     # 左图：时间序列
#     ax_left = axes[idx, 0]
#     # 右图：CDF
#     ax_right = axes[idx, 1]
#
#     model_name = result['model_name']
#     y_pred = result['predictions']
#     y_true = result['targets']
#     ph_value = result.get('ph', 'N/A')
#
#     # ========== 左图：时间序列 ==========
#     N = min(1000, len(y_true))
#     ax_left.plot(y_true[:N], label='True', linewidth=1.2, color='blue', alpha=0.8)
#     ax_left.plot(y_pred[:N], label='Pred', linewidth=1.2, color='red', alpha=0.7)
#
#     # 计算并显示RMSE
#     rmse = np.sqrt(mean_squared_error(y_true[:N], y_pred[:N]))
#     ax_left.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax_left.transAxes,
#                  fontsize=9, verticalalignment='top',
#                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
#
#     ax_left.set_title(f'{letters[idx]} Time Series: {model_name} (Ph={ph_value})', fontsize=11)
#     ax_left.set_xlabel('Time Step', fontsize=10)
#     ax_left.set_ylabel('Value', fontsize=10)
#     ax_left.legend(fontsize=9)
#     ax_left.grid(True, alpha=0.3)
#
#     # ========== 右图：CDF ==========
#     # 计算CDF
#     true_sorted = np.sort(y_true)
#     pred_sorted = np.sort(y_pred)
#
#     true_cdf = np.arange(len(true_sorted)) / len(true_sorted)
#     pred_cdf = np.arange(len(pred_sorted)) / len(pred_sorted)
#
#     # 插值
#     x = np.linspace(min(true_sorted.min(), pred_sorted.min()),
#                     max(true_sorted.max(), pred_sorted.max()), 1000)
#     true_cdf_interp = np.interp(x, true_sorted, true_cdf)
#     pred_cdf_interp = np.interp(x, pred_sorted, pred_cdf)
#
#     # 计算KS
#     ks_value = np.max(np.abs(true_cdf_interp - pred_cdf_interp))
#
#     # 绘制CDF
#     ax_right.plot(x, true_cdf_interp, label='True CDF', linewidth=2, color='blue')
#     ax_right.plot(x, pred_cdf_interp, label='Pred CDF', linewidth=2, color='red')
#
#     # 标记KS点
#     ks_idx = np.argmax(np.abs(true_cdf_interp - pred_cdf_interp))
#     ax_right.scatter(x[ks_idx], true_cdf_interp[ks_idx], color='darkorange', s=100,
#                      label=f'KS={ks_value:.4f}', zorder=5)
#
#     ax_right.set_title(f'CDF Comparison: KS={ks_value:.4f}', fontsize=11)
#     ax_right.set_xlabel('Value', fontsize=10)
#     ax_right.set_ylabel('CDF', fontsize=10)
#     ax_right.legend(fontsize=9)
#     ax_right.grid(True, alpha=0.3)
#
# plt.suptitle('Model Performance Analysis: Time Series (Left) and CDF Distribution (Right)',
#              fontsize=16, fontweight='bold', y=0.98)
# plt.tight_layout()
# plt.show()













#
# # 创建5个子图
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2行3列，最后一个位置可以空着
# axes = axes.flatten()  # 展平为一维数组
#
# # 设置字母标签
# subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
#
# # 遍历每个模型结果
# for idx, result in enumerate(all_model_results):
#     if idx >= 5:  # 只显示前5个
#         break
#
#     ax = axes[idx]
#     model_name = result['model_name']
#     y_pred = result['predictions']
#     y_true = result['targets']
#     ph_value = result.get('ph', 'N/A')
#
#     # 计算指标
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#
#     # 取前N个点显示（避免太密集）
#     N = min(2000, len(y_true))
#
#     # 绘制时间序列对比
#     time_steps = np.arange(N)
#     ax.plot(time_steps, y_true[:N], label='True', linewidth=1.5, alpha=0.8)
#     ax.plot(time_steps, y_pred[:N], label='Pred', linewidth=1.5, alpha=0.7)
#
#     # 设置子图标题和标签
#     ax.set_title(f'{subplot_labels[idx]} Model: {model_name}\nPh={ph_value}, RMSE={rmse:.4f}', fontsize=10)
#     ax.set_xlabel('Time Step', fontsize=9)
#     ax.set_ylabel('Value', fontsize=9)
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3)
#
#     # 添加字母标签在左上角
#     ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
#             fontsize=12, fontweight='bold', verticalalignment='top')
#
# # 如果有空余的子图，隐藏它们
# for idx in range(len(all_model_results), len(axes)):
#     axes[idx].axis('off')
#
# plt.suptitle('Model Predictions Comparison (First 2000 Time Steps)', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()



























#
# # ==========================================
# # 7. 简单指标
# # ==========================================
# rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
# print(f"RMSE: {rmse:.6f}")
#
# # 取前N个点（太长会乱）
# N = len(y_true)
#
# plt.figure(figsize=(12, 5))
# plt.plot(y_true[:N], label="True", linewidth=2)
# plt.plot(y_pred[:N], label="Pred", linewidth=2, alpha=0.8)
#
# plt.title("Time Domain Comparison")
# plt.xlabel("Time Step")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # 展平（因为可能是 batch × time）
# true_flat = y_true.flatten()
# pred_flat = y_pred.flatten()
#
# # 排序
# true_sorted = np.sort(true_flat)
# pred_sorted = np.sort(pred_flat)
#
# # CDF
# true_cdf = np.arange(len(true_sorted)) / len(true_sorted)
# pred_cdf = np.arange(len(pred_sorted)) / len(pred_sorted)
#
# # KS统计量（最大CDF差）
# # 需要统一x轴对齐（插值方式）
# x = np.linspace(
#     min(true_sorted.min(), pred_sorted.min()),
#     max(true_sorted.max(), pred_sorted.max()),
#     1000
# )
#
# true_cdf_interp = np.interp(x, true_sorted, true_cdf)
# pred_cdf_interp = np.interp(x, pred_sorted, pred_cdf)
#
# ks_value = np.max(np.abs(true_cdf_interp - pred_cdf_interp))
#
# print(f"KS value: {ks_value:.6f}")
#
# plt.figure(figsize=(6, 5))
#
# plt.plot(x, true_cdf_interp, label="True CDF", linewidth=2)
# plt.plot(x, pred_cdf_interp, label="Pred CDF", linewidth=2)
#
# # KS最大点（用于标注）
# ks_idx = np.argmax(np.abs(true_cdf_interp - pred_cdf_interp))
# plt.scatter(
#     x[ks_idx],
#     true_cdf_interp[ks_idx],
#     color="red",
#     zorder=5,
#     label=f"KS={ks_value:.4f}"
# )
#
# plt.title("CDF Comparison with KS Statistic")
# plt.xlabel("Value")
# plt.ylabel("CDF")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()