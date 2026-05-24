import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string  # 导入string模块用于生成字母编号

# 设置绘图风格
sns.set(style="whitegrid")

# 1. 加载数据
df = pd.read_csv('all_results_long_filtered.csv')

# 2. 筛选主模型 'Fusion' (VAP-Net)
df_plot = df[df['model'] == 'Fusion'].copy()

# 3. 确定子图组合 (Yhis, rate)
combos = df_plot[['Yhis', 'rate']].drop_duplicates().sort_values(['Yhis', 'rate'])
n_plots = len(combos)

# 4. 定义网格布局 (每行2个子图)
cols = 2
rows = (n_plots + cols - 1) // cols

# 增加图形高度以容纳统一的图例
fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows + 2))
axes = axes.flatten()

# 获取所有损失函数列表
losses = sorted(df_plot['loss'].unique())
palette = sns.color_palette("husl", len(losses))

# 5. 循环绘制每个子图
for i, (index, row) in enumerate(combos.iterrows()):
    y_val = row['Yhis']
    r_val = row['rate']
    ax = axes[i]

    # 筛选当前子图的数据
    subset = df_plot[(df_plot['Yhis'] == y_val) & (df_plot['rate'] == r_val)]
    # 按 P 排序以确保线条正确连接
    subset = subset.sort_values('Phis')

    # 绘制 7 个损失函数的曲线
    for j, loss in enumerate(losses):
        loss_data = subset[subset['loss'] == loss]
        # 使用 lineplot 自动处理可能存在的重复点（取均值并显示置信区间）
        sns.lineplot(data=loss_data, x='Phis', y='tes_KS_Dist',
                     marker='o', label=loss, color=palette[j], ax=ax)

    # 设置坐标轴标签
    ax.set_xlabel('Prediction Horizon $P$ (s)', fontsize=16)
    ax.set_ylabel('KS Distance', fontsize=16)
    # 统一纵坐标范围为0.3~0.85
    ax.set_ylim(0.3, 0.85)
    # 移除每个子图的图例
    ax.get_legend().remove()

    # 在子图下方添加小标题
    subplot_label = f'({string.ascii_lowercase[i]}) Y = {y_val}, Rate = {r_val}'
    ax.text(0.5, -0.15, subplot_label,  # 注意：负数表示在坐标轴下方
            transform=ax.transAxes,  # 使用坐标轴相对坐标
            ha='center',  # 水平居中对齐
            fontsize=18,
            fontweight='bold')

# 移除多余的空白子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 6. 添加统一的图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.98),  # 稍微向上调整位置
           ncol=4,  # 4列排列
           fontsize=18,
           # title='Loss Function',
           frameon=True,
           shadow=True)

# 7. 调整布局，为顶部图例和底部小标题留出空间
plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # 调整底部和顶部边距

# 8. 保存图片
plt.savefig('ks_analysis_by_loss.pdf', dpi=300, bbox_inches='tight')
print("图表已保存为: ks_analysis_by_loss.png")
plt.show()