import pandas as pd
import numpy as np

# 配置
INPUT_FILE = "/home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument/Tb_features.csv"
OUTPUT_FILE = "/home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument/Tb_features_filtered.csv"
CORR_THRESHOLD = 0.95  # 建议设为 0.95，给模型留一点冗余空间，0.90 可能太激进

def main():
    print(f"读取文件: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # 确保 ID 列存在，并不参与计算
    if "PatientID" not in df.columns:
        raise ValueError("缺少 PatientID 列")
    
    # 提取特征列
    # 排除非数值列（防止 metadata 混入）
    feature_cols = [c for c in df.columns if c != "PatientID"]
    # 强制转为数值，无法转换的变为 NaN
    df_feats = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    print(f"原始特征数: {len(feature_cols)}")

    # ---------------------------
    # 1. 基础清洗 (NaN & Inf)
    # ---------------------------
    # 删除含有任何 NaN 或 Inf 的列（Radiomics 计算失败的特征通常不可靠）
    # 或者选择用中位数填充：df_feats = df_feats.fillna(df_feats.median())
    cols_before = df_feats.columns
    df_feats = df_feats.dropna(axis=1, how='any')
    # 再次检查 Inf
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    
    print(f"去除 NaN/Inf 后: {len(df_feats.columns)}")

    # ---------------------------
    # 2. 去除常数特征 (方差为 0)
    # ---------------------------
    # 注意：不要设定 >0.01，只去除 == 0 的，防止误删小量纲特征
    variances = df_feats.var()
    keep_var = variances[variances > 0].index.tolist()
    df_feats = df_feats[keep_var]
    print(f"去除常数特征后: {len(keep_var)}")

    # ---------------------------
    # 3. 高相关性筛选 (Smart Drop)
    # ---------------------------
    print("正在计算相关性矩阵（可能需要几秒钟）...")
    corr_matrix = df_feats.corr().abs()
    
    # 获取上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找到需要删除的列
    to_drop = []
    for column in upper.columns:
        # 如果该列与之前的任何列相关性 > 阈值
        if any(upper[column] > CORR_THRESHOLD):
            # 智能保留逻辑：
            # 如果当前列是 "Original"（原始特征），而相关的是 "Wavelet"（衍生特征），
            # 我们其实更想保留 Original。
            # 但为了代码简单，这里采用标准策略：保留先出现的列（通常是原始特征在前面），删除后出现的列。
            to_drop.append(column)
    
    keep_final = [c for c in df_feats.columns if c not in to_drop]
    
    print(f"相关性筛选 (>{CORR_THRESHOLD}): {len(df_feats.columns)} -> {len(keep_final)}")
    print(f"最终保留特征数: {len(keep_final)}")

    # ---------------------------
    # 4. 保存
    # ---------------------------
    # 重新拼接 PatientID
    df_final = pd.concat([df["PatientID"], df[keep_final]], axis=1)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"文件已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()