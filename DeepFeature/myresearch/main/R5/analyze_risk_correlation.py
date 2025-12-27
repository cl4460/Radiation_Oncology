#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 Tb (Radiomics), Df (Deep Features), Cp (Clinical) 三个模型的 OOF risk predictions
计算 Spearman 相关性，并判断是否有互补空间用于 late fusion
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[WARNING] matplotlib/seaborn not available. Visualization will be skipped.")


def load_oof_predictions(path: Path, id_col: str = "PatientID", 
                         risk_col: str = "risk_pred_oof") -> pd.DataFrame:
    """加载 OOF predictions，返回包含 PatientID 和 risk 的 DataFrame"""
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in {path}")
    if risk_col not in df.columns:
        raise ValueError(f"Column '{risk_col}' not found in {path}")
    
    return df[[id_col, risk_col]].copy()


def load_summary_json(path: Path) -> Dict:
    """加载 summary.json 获取 C-index 信息"""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def merge_risks(cp_path: Path, df_path: Path, tb_path: Path,
                id_col: str = "PatientID", risk_col: str = "risk_pred_oof") -> pd.DataFrame:
    """合并三个模型的 OOF risk predictions"""
    cp_df = load_oof_predictions(cp_path, id_col, risk_col)
    df_df = load_oof_predictions(df_path, id_col, risk_col)
    tb_df = load_oof_predictions(tb_path, id_col, risk_col)
    
    # 重命名列以避免冲突
    cp_df = cp_df.rename(columns={risk_col: "risk_Cp"})
    df_df = df_df.rename(columns={risk_col: "risk_Df"})
    tb_df = tb_df.rename(columns={risk_col: "risk_Tb"})
    
    # 合并
    merged = cp_df.merge(df_df, on=id_col, how="inner")
    merged = merged.merge(tb_df, on=id_col, how="inner")
    
    return merged


def compute_spearman_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """计算 Spearman 相关系数矩阵"""
    risk_cols = ["risk_Cp", "risk_Df", "risk_Tb"]
    corr_matrix = np.zeros((3, 3))
    pvalue_matrix = np.zeros((3, 3))
    
    for i, col1 in enumerate(risk_cols):
        for j, col2 in enumerate(risk_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
                pvalue_matrix[i, j] = 0.0
            else:
                corr, pval = spearmanr(df[col1], df[col2], nan_policy='omit')
                corr_matrix[i, j] = corr
                pvalue_matrix[i, j] = pval
    
    corr_df = pd.DataFrame(
        corr_matrix,
        index=["Cp (Clinical)", "Df (Deep)", "Tb (Radiomics)"],
        columns=["Cp (Clinical)", "Df (Deep)", "Tb (Radiomics)"]
    )
    
    pval_df = pd.DataFrame(
        pvalue_matrix,
        index=["Cp (Clinical)", "Df (Deep)", "Tb (Radiomics)"],
        columns=["Cp (Clinical)", "Df (Deep)", "Tb (Radiomics)"]
    )
    
    return corr_df, pval_df


def analyze_correlation_insights(corr_df: pd.DataFrame, pval_df: pd.DataFrame,
                                 summaries: Dict[str, Dict]) -> Dict:
    """分析相关性并给出见解"""
    insights = {
        "correlations": {},
        "interpretation": []
    }
    
    # 提取关键相关性
    corr_tb_df = corr_df.loc["Tb (Radiomics)", "Df (Deep)"]
    corr_tb_cp = corr_df.loc["Tb (Radiomics)", "Cp (Clinical)"]
    corr_df_cp = corr_df.loc["Df (Deep)", "Cp (Clinical)"]
    
    insights["correlations"]["Tb_vs_Df"] = float(corr_tb_df)
    insights["correlations"]["Tb_vs_Cp"] = float(corr_tb_cp)
    insights["correlations"]["Df_vs_Cp"] = float(corr_df_cp)
    
    # 获取 C-index (如果有) - 优先使用 oofnorm 版本
    cidx_tb = summaries.get("tb", {}).get("harrell_c_oof_oofnorm") or summaries.get("tb", {}).get("harrell_c_oof", None)
    cidx_df = summaries.get("df", {}).get("harrell_c_oof_oofnorm") or summaries.get("df", {}).get("harrell_c_oof", None)
    cidx_cp = summaries.get("cp", {}).get("harrell_c_oof_oofnorm") or summaries.get("cp", {}).get("harrell_c_oof", None)
    
    # 判断标准
    high_corr_thresh = 0.8
    low_corr_thresh = 0.3
    good_cindex_thresh = 0.65
    
    # Tb vs Df 分析（最重要的）
    abs_corr_tb_df = abs(corr_tb_df)
    if abs_corr_tb_df > high_corr_thresh:
        insights["interpretation"].append(
            f"⚠️ Tb 与 Df 高度相关 (ρ={corr_tb_df:.3f})："
            f"深特征和 radiomics 学到的排序几乎一样，互补空间很小。"
        )
    elif abs_corr_tb_df < low_corr_thresh:
        if cidx_df is not None and cidx_df < good_cindex_thresh:
            insights["interpretation"].append(
                f"❌ Tb 与 Df 低相关 (ρ={corr_tb_df:.3f})，但 Df C-index 较低 ({cidx_df:.3f})："
                f"说明 Df 大部分是噪声，加进来只会拖后腿。"
            )
        elif cidx_tb is not None and cidx_tb >= good_cindex_thresh and \
             cidx_df is not None and cidx_df >= good_cindex_thresh:
            insights["interpretation"].append(
                f"✅ 理想情况：Tb 与 Df 低相关 (ρ={corr_tb_df:.3f})，"
                f"且两者单模态性能都不错 (Tb C-index={cidx_tb:.3f}, Df C-index={cidx_df:.3f})。"
                f"Late fusion 很有希望！"
            )
        else:
            insights["interpretation"].append(
                f"📊 Tb 与 Df 低相关 (ρ={corr_tb_df:.3f})：有互补潜力，"
                f"但需要检查单模态性能是否足够好。"
            )
    else:
        insights["interpretation"].append(
            f"📈 Tb 与 Df 中等相关 (ρ={corr_tb_df:.3f})：有一定互补空间，"
            f"但可能不如低相关情况。"
        )
    
    # 其他相关性分析
    if abs(corr_tb_cp) > high_corr_thresh:
        insights["interpretation"].append(
            f"⚠️ Tb 与 Cp 高度相关 (ρ={corr_tb_cp:.3f})：radiomics 和临床特征信息重叠较多。"
        )
    
    if abs(corr_df_cp) > high_corr_thresh:
        insights["interpretation"].append(
            f"⚠️ Df 与 Cp 高度相关 (ρ={corr_df_cp:.3f})：深特征和临床特征信息重叠较多。"
        )
    
    return insights


def plot_correlation_heatmap(corr_df: pd.DataFrame, pval_df: pd.DataFrame, 
                             output_path: Path, figsize=(8, 6)):
    """绘制相关性热图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建带星号标注的矩阵（表示显著性）
    annot_matrix = corr_df.copy().astype(str)  # 先转为字符串类型避免 FutureWarning
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            if i != j:
                pval = pval_df.iloc[i, j]
                corr_val = corr_df.iloc[i, j]
                if pval < 0.001:
                    annot_matrix.iloc[i, j] = f"{corr_val:.3f}***"
                elif pval < 0.01:
                    annot_matrix.iloc[i, j] = f"{corr_val:.3f}**"
                elif pval < 0.05:
                    annot_matrix.iloc[i, j] = f"{corr_val:.3f}*"
                else:
                    annot_matrix.iloc[i, j] = f"{corr_val:.3f}"
            else:
                annot_matrix.iloc[i, j] = "1.000"
    
    sns.heatmap(
        corr_df,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"label": "Spearman Correlation (ρ)"},
        ax=ax
    )
    
    ax.set_title("Spearman Correlation Matrix\n(OOF Risk Predictions)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Correlation heatmap: {output_path}")


def plot_scatter_pairs(merged_df: pd.DataFrame, output_path: Path, figsize=(15, 5)):
    """绘制成对散点图"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    pairs = [
        ("risk_Tb", "risk_Df", "Tb (Radiomics) vs Df (Deep)"),
        ("risk_Tb", "risk_Cp", "Tb (Radiomics) vs Cp (Clinical)"),
        ("risk_Df", "risk_Cp", "Df (Deep) vs Cp (Clinical)")
    ]
    
    for idx, (x_col, y_col, title) in enumerate(pairs):
        ax = axes[idx]
        
        # 计算相关性
        corr, pval = spearmanr(merged_df[x_col], merged_df[y_col], nan_policy='omit')
        
        # 散点图
        ax.scatter(merged_df[x_col], merged_df[y_col], alpha=0.5, s=30)
        
        # 添加趋势线
        z = np.polyfit(merged_df[x_col].dropna(), merged_df[y_col].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged_df[x_col].min(), merged_df[x_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'ρ={corr:.3f}')
        
        ax.set_xlabel(x_col.replace("risk_", "").replace("Tb", "Tb (Radiomics)").replace("Df", "Df (Deep)").replace("Cp", "Cp (Clinical)"))
        ax.set_ylabel(y_col.replace("risk_", "").replace("Tb", "Tb (Radiomics)").replace("Df", "Df (Deep)").replace("Cp", "Cp (Clinical)"))
        ax.set_title(f"{title}\nρ={corr:.3f} (p={pval:.2e})", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Scatter plots: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between Tb, Df, and Cp OOF risk predictions"
    )
    parser.add_argument("--cp_oof", type=str, required=True,
                       help="Path to Clinical OOF predictions CSV")
    parser.add_argument("--df_oof", type=str, required=True,
                       help="Path to Deep Features OOF predictions CSV")
    parser.add_argument("--tb_oof", type=str, required=True,
                       help="Path to Radiomics OOF predictions CSV")
    parser.add_argument("--cp_summary", type=str, default=None,
                       help="Path to Clinical summary.json (optional)")
    parser.add_argument("--df_summary", type=str, default=None,
                       help="Path to Deep Features summary.json (optional)")
    parser.add_argument("--tb_summary", type=str, default=None,
                       help="Path to Radiomics summary.json (optional)")
    parser.add_argument("--outdir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--id_col", type=str, default="PatientID",
                       help="ID column name")
    parser.add_argument("--risk_col", type=str, default="risk_pred_oof",
                       help="Risk prediction column name")
    
    args = parser.parse_args()
    
    # 创建输出目录
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Risk Prediction Correlation Analysis")
    print("=" * 80)
    print(f"Clinical OOF: {args.cp_oof}")
    print(f"Deep Features OOF: {args.df_oof}")
    print(f"Radiomics OOF: {args.tb_oof}")
    print()
    
    # 1. 合并 OOF predictions
    print("[1/5] Merging OOF predictions...")
    merged_df = merge_risks(
        Path(args.cp_oof),
        Path(args.df_oof),
        Path(args.tb_oof),
        id_col=args.id_col,
        risk_col=args.risk_col
    )
    print(f"  Merged {len(merged_df)} samples")
    
    # 保存合并后的数据
    merged_df.to_csv(outdir / "merged_risks.csv", index=False)
    print(f"[SAVED] Merged risks: {outdir / 'merged_risks.csv'}")
    
    # 2. 加载 summary JSONs
    summaries = {}
    if args.cp_summary:
        summaries["cp"] = load_summary_json(Path(args.cp_summary))
    if args.df_summary:
        summaries["df"] = load_summary_json(Path(args.df_summary))
    if args.tb_summary:
        summaries["tb"] = load_summary_json(Path(args.tb_summary))
    
    # 3. 计算 Spearman 相关性
    print("\n[2/5] Computing Spearman correlations...")
    corr_df, pval_df = compute_spearman_correlations(merged_df)
    print("\nSpearman Correlation Matrix:")
    print(corr_df.round(3))
    print("\nP-values:")
    print(pval_df.round(4))
    
    # 保存相关性矩阵
    corr_df.to_csv(outdir / "correlation_matrix.csv")
    pval_df.to_csv(outdir / "pvalue_matrix.csv")
    
    # 4. 分析见解
    print("\n[3/5] Analyzing correlations and generating insights...")
    insights = analyze_correlation_insights(corr_df, pval_df, summaries)
    
    # 5. 可视化
    print("\n[4/5] Generating visualizations...")
    plot_correlation_heatmap(corr_df, pval_df, outdir / "correlation_heatmap.png")
    plot_scatter_pairs(merged_df, outdir / "scatter_pairs.png")
    
    # 6. 保存分析报告
    print("\n[5/5] Saving analysis report...")
    report = {
        "summary": {
            "n_samples": int(len(merged_df)),
            "correlations": {
                "Tb_vs_Df": float(corr_df.loc["Tb (Radiomics)", "Df (Deep)"]),
                "Tb_vs_Cp": float(corr_df.loc["Tb (Radiomics)", "Cp (Clinical)"]),
                "Df_vs_Cp": float(corr_df.loc["Df (Deep)", "Cp (Clinical)"]),
            },
            "p_values": {
                "Tb_vs_Df": float(pval_df.loc["Tb (Radiomics)", "Df (Deep)"]),
                "Tb_vs_Cp": float(pval_df.loc["Tb (Radiomics)", "Cp (Clinical)"]),
                "Df_vs_Cp": float(pval_df.loc["Df (Deep)", "Cp (Clinical)"]),
            }
        },
        "c_index": {
            "Tb": summaries.get("tb", {}).get("harrell_c_oof_oofnorm") or summaries.get("tb", {}).get("harrell_c_oof", None),
            "Df": summaries.get("df", {}).get("harrell_c_oof_oofnorm") or summaries.get("df", {}).get("harrell_c_oof", None),
            "Cp": summaries.get("cp", {}).get("harrell_c_oof_oofnorm") or summaries.get("cp", {}).get("harrell_c_oof", None),
        },
        "insights": insights["interpretation"]
    }
    
    # 添加相关性判断
    corr_tb_df = abs(report["summary"]["correlations"]["Tb_vs_Df"])
    if corr_tb_df > 0.8:
        report["late_fusion_potential"] = "LOW - High correlation indicates little complementary information"
    elif corr_tb_df < 0.3:
        cidx_tb = report["c_index"]["Tb"]
        cidx_df = report["c_index"]["Df"]
        if cidx_tb is not None and cidx_df is not None:
            if cidx_tb >= 0.65 and cidx_df >= 0.65:
                report["late_fusion_potential"] = "HIGH - Low correlation with good individual performance"
            elif cidx_df < 0.65:
                report["late_fusion_potential"] = "LOW - Low correlation but Df has poor performance"
            else:
                report["late_fusion_potential"] = "MODERATE - Low correlation, check individual performance"
        else:
            report["late_fusion_potential"] = "MODERATE - Low correlation, C-index not available"
    else:
        report["late_fusion_potential"] = "MODERATE - Moderate correlation"
    
    with open(outdir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # 打印报告
    print("\n" + "=" * 80)
    print("ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nSample size: {report['summary']['n_samples']}")
    print(f"\nSpearman Correlations:")
    for pair, corr in report['summary']['correlations'].items():
        pval = report['summary']['p_values'][pair]
        print(f"  {pair:12s}: ρ = {corr:7.3f} (p = {pval:.2e})")
    
    if any(v is not None for v in report['c_index'].values()):
        print(f"\nHarrell's C-index (OOF):")
        for model, cidx in report['c_index'].items():
            if cidx is not None:
                print(f"  {model:12s}: {cidx:.4f}")
    
    print(f"\n{'='*80}")
    print("INSIGHTS:")
    print("=" * 80)
    for insight in report['insights']:
        print(f"  {insight}")
    
    print(f"\n{'='*80}")
    print(f"LATE FUSION POTENTIAL: {report['late_fusion_potential']}")
    print("=" * 80)
    print(f"\n[SAVED] Analysis report: {outdir / 'analysis_report.json'}")
    print(f"[SAVED] All outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()

