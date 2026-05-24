import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('all_results_long.csv')

# 2. 修改模型名称：将源代码中的 'Fusion' 替换为论文中的 'VAP-Net'
df['model'] = df['model'].replace('Fusion', 'VAP-Net')

# 3. 定义参数与筛选条件
main_model = 'VAP-Net'
baselines = ['AGLSTM', 'FiLMLSTM', 'SimpleLSTM']
models_comparison = [main_model] + baselines
target_loss = 'W2'
rates = [10, 30, 60]

# 确保数值类型正确
df['Phis'] = pd.to_numeric(df['Phis'])
df['Round'] = pd.to_numeric(df['Round'])

# 4. 开始绘图
# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
# fig.suptitle(f'Comparative Analysis of KS Distance (Loss: {target_loss}, Avg over Y)', fontsize=20, y=1.02)

# 颜色与标记设置，突出主模型 VAP-Net
palette = {
    'VAP-Net': '#D62728',  # 醒目的红色
    'AGLSTM': '#1F77B4',  # 蓝色
    'FiLMLSTM': '#FF7F0E',  # 橙色
    'SimpleLSTM': '#2CA02C'  # 绿色
}
markers = {'VAP-Net': 'o', 'AGLSTM': 's', 'FiLMLSTM': '^', 'SimpleLSTM': 'd'}

sub_labels = ['(a)', '(b)', '(c)']

for i, r in enumerate(rates):
    ax = axes[i]

    # 关键修正：筛选最终剪枝轮次 (Round 9)，并针对多次实验运行求均值
    # 这样可以避免多轮数据堆叠造成的连线混乱（锯齿状曲线）
    subset = df[
        (df['model'].isin(models_comparison)) &
        (df['loss'] == target_loss) &
        (df['rate'] == r)
        # (df['Round'] == 9)
    ]

    # 逻辑核心：对 Yhis (即历史窗口Y) 取平均，聚合多组实验运行结果
    summary = subset.groupby(['model', 'Phis'])['tes_KS_Dist'].mean().reset_index()

    # 绘图
    for model in models_comparison:
        # 确保按 P 的顺序画线
        model_data = summary[summary['model'] == model].sort_values('Phis')
        if not model_data.empty:
            ax.plot(
                model_data['Phis'],
                model_data['tes_KS_Dist'],
                label=model,
                marker=markers[model],
                color=palette[model],
                linewidth=3.0 if model == 'VAP-Net' else 1.8,  # 加粗主模型
                markersize=9,
                alpha=0.9
            )

    # 子图细节优化
    # ax.set_title(f'Temporal Resolution: {r}s', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Prediction Horizon $P$ (s)', fontsize=14)
    if i == 0:
        ax.set_ylabel('KS Distance', fontsize=14)

    ax.set_xticks([100, 300, 900, 1500, 2100, 2700])
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加子图下方的小标题 (a) rate = 10s, (b) rate = 30s, (c) rate = 60s
    ax.text(0.5, -0.32, f'{sub_labels[i]} rate = {r}s',
            transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')

# 统一设置图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1),
           ncol=4, fontsize=14, frameon=True, shadow=True)

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.92])

# 保存图片
plt.savefig('Figure_1_KS_Analysis_Final.png', dpi=300, bbox_inches='tight')
plt.show()