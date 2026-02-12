import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_clean_label(idx_str, row=None):
    idx_str = str(idx_str)
    if "full_text" in idx_str:
        return "Full Text"
    
    label = ""
    if row is not None and 'Method' in row:
        label = row['Method'].replace("-Level", "")
    else:
        label = idx_str

    if row is not None and 'compression_rate' in row and row['compression_rate'] > 0.01:
        comp_pct = row['compression_rate'] * 100 if row['compression_rate'] <= 1.0 else row['compression_rate']
        label += f"\n-{comp_pct:.0f}% Tokens"
    return label

def plot_metrics(df, output_path, title_suffix=""):
    metrics_to_plot = ['rouge1f', 'rouge2f', 'rougeLf', 'bertscore_f1', 'bleu']
    valid_metrics = [m for m in metrics_to_plot if m in df.columns]
    if not valid_metrics: return

    plot_data = df.copy()
    plot_data['DisplayLabel'] = [get_clean_label(idx, row) for idx, row in plot_data.iterrows()]
    df_melted = plot_data.melt(id_vars=['DisplayLabel'], value_vars=valid_metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    chart = sns.barplot(data=df_melted, x='DisplayLabel', y='Score', hue='Metric', palette="viridis")

    plt.title(f"Summary Quality Metrics {title_suffix}", fontsize=15)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylabel("Score (0-100)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def aggregate_results(results_dir):
    files = {
        'Sentence-Level': 'sentence_metrics.csv',
        'Phrase-Level': 'phrase_metrics.csv',
        'Full Text': 'full_text_metrics.csv'
    }
    
    dfs = []
    for method, fname in files.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            d = pd.read_csv(path, index_col=0)
            d['Method'] = method
            if method == 'Full Text':
                d['compression_rate'] = 0.0
            dfs.append(d)
    
    if not dfs:
        raise ValueError("No results files found in directory.")
        
    df_final = pd.concat(dfs)
    # Group sort logic could go here
    return df_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    print(f"Analyzing results in {args.results_dir}...")
    try:
        master_df = aggregate_results(args.results_dir)
        master_path = os.path.join(args.results_dir, "master_benchmark_results.csv")
        master_df.to_csv(master_path)
        print(f"Saved aggregated results to {master_path}")

        # Generate Plot
        plot_path = os.path.join(args.results_dir, "metrics_comparison.png")
        plot_metrics(master_df, plot_path)
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
