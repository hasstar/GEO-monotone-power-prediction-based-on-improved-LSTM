import os
import pandas as pd
import re

def aggregate_active_features(root_dir):
    summary_data = []

    # 遍历 root_dir 下的所有文件夹
    for folder_name in os.listdir(root_dir):
        # 使用正则匹配文件夹命名规则：AWPL_pruning_logs_输入长度_预测长度_rate_模型_损失函数
        # 例如：AWPL_pruning_logs_900_2700_60_SingleCross_W2
        pattern = r"AWPL_pruning_logs_(\d+)_(\d+)_(\d+)_([^_]+)_([^_]+)"
        match = re.match(pattern, folder_name)
        
        if match:
            input_len = match.group(1)
            pred_len = match.group(2)
            rate = match.group(3)
            model_name = match.group(4)
            loss_func = match.group(5)
            
            file_path = os.path.join(root_dir, folder_name, "pruning_log_round_08.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # 筛选 Status 为 Active 的行
                    active_features = df[df['Status'].str.strip() == 'Active']
                    
                    # 提取特征名称（Feature_Name）
                    # 假设你关注的是这些特征的列表
                    feature_list = active_features['Feature_Name'].tolist()
                    
                    # 将结果加入汇总列表
                    summary_data.append({
                        "Input_Len": input_len,
                        "Pred_Len": pred_len,
                        "Rate": rate,
                        "Model": model_name,
                        "Loss": loss_func,
                        "Active_Features_Count": len(feature_list),
                        "Active_Features": ", ".join(feature_list)
                    })
                except Exception as e:
                    print(f"读取文件 {file_path} 出错: {e}")

    # 转换为 DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # 按照输入长度和模型排序，方便观察规律
    summary_df = summary_df.sort_values(by=["Input_Len", "Model", "Rate"])
    
    return summary_df

# 使用方法：
# 将 '.' 替换为你存放那些文件夹的实际路径
root_path = './' 
result_df = aggregate_active_features(root_path)

# 打印前几行查看
print(result_df.head())

# 保存为 Excel 或 CSV 供论文绘图或制作表格使用
result_df.to_csv("integrated_active_atmospheric_features.csv", index=False, encoding='utf-8-sig')
print("\n汇总完成！结果已保存至 integrated_active_atmospheric_features.csv")