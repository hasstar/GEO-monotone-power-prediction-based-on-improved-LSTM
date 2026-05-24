import pandas as pd


def generate_latex_table(csv_path):
    # 1. 加载数据
    df = pd.read_csv(csv_path)

    # 2. 筛选 W2 损失函数
    df_w2 = df[df['loss'] == 'W2'].copy()

    # 3. 模型名称映射
    model_mapping = {
        'Fusion': 'VAP-Net',
        'AGLSTM': 'AG-LSTM',
        'FiLMLSTM': 'FiLM-LSTM',
        'SimpleLSTM': 'SimpleLSTM'
    }
    df_w2['model_renamed'] = df_w2['model'].map(model_mapping)

    # 4. 过滤目标模型
    target_models = ['VAP-Net', 'AG-LSTM', 'FiLM-LSTM', 'SimpleLSTM']
    df_filtered = df_w2[df_w2['model_renamed'].isin(target_models)]

    # 5. 计算不同 rate 下的 KS 值平均值 (tes_KS_Dist)
    table_data = df_filtered.groupby(['model_renamed', 'Yhis', 'Phis'])['tes_KS_Dist'].mean().reset_index()

    # 6. 定义表头结构 (Y 与 P 的对应关系)
    y_p_structure = {
        100: [100, 300, 900, 1500, 2100, 2700],
        300: [300, 900, 1500, 2100, 2700],
        900: [900, 1500, 2100, 2700]
    }

    # 7. 构建 LaTeX 字符串
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\small")
    # 1个模型列 + 15个数据列
    latex.append(r"\begin{tabular}{l" + "c" * 15 + "}")
    latex.append(r"\toprule")

    # 第一级表头: Y
    header_y = "Model & "
    header_y += r"\multicolumn{6}{c}{Y=100} & "
    header_y += r"\multicolumn{5}{c}{Y=300} & "
    header_y += r"\multicolumn{4}{c}{Y=900} \\"
    latex.append(header_y)

    # 添加横线
    latex.append(r"\cmidrule(lr){2-7} \cmidrule(lr){8-12} \cmidrule(lr){13-16}")

    # 第二级表头: P
    all_p = []
    for y in [100, 300, 900]:
        all_p.extend([str(p) for p in y_p_structure[y]])
    header_p = " & " + " & ".join(all_p) + r" \\"
    latex.append(header_p)
    latex.append(r"\midrule")

    # 填充模型数据行
    for model in target_models:
        row_values = [model]
        for y in [100, 300, 900]:
            for p in y_p_structure[y]:
                # 提取对应 Y 和 P 的均值
                val = table_data[
                    (table_data['model_renamed'] == model) &
                    (table_data['Yhis'] == y) &
                    (table_data['Phis'] == p)
                    ]['tes_KS_Dist']

                if not val.empty:
                    row_values.append(f"{val.values[0]:.4f}")
                else:
                    row_values.append("-")

        latex.append(" & ".join(row_values) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Average KS distance for W2 loss across different Y and P.}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


# 执行并打印结果
if __name__ == "__main__":
    latex_code = generate_latex_table('all_results_long.csv')
    print(latex_code)