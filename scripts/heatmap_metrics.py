import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Fixed tool order
fixed_tool_order = ["VarChat", "Gpt-4o", "MistralAI", "Perplexity", "ScholarAI"]

# Original metric columns in the dataset
original_metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]

# Display names for the metrics
display_metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]

# Load full data
df_full = pd.read_csv("evaluation_data.csv", sep=";")

# Loop over all modes
for mode in ["all", "germline", "somatic"]:
    if mode == "germline":
        df = df_full[df_full["Type"].str.lower() == "germline"]
    elif mode == "somatic":
        df = df_full[df_full["Type"].str.lower() == "somatic"]
    else:
        df = df_full.copy()

    # Compute mean and std
    grouped = df.groupby("Tool name")[original_metrics]
    means = grouped.mean().round(1)
    stds = grouped.std().round(1)

    # Combine mean ± std as string for output
    combined = means.copy()
    for metric in original_metrics:
        combined[metric] = means[metric].astype(str) + " (± " + stds[metric].astype(str) + ")"

    # Reorder
    means = means.reindex(fixed_tool_order)
    stds = stds.reindex(fixed_tool_order)
    combined = combined.reindex(fixed_tool_order)

    # Save table
    combined.to_csv(f"tool_metric_table_{mode}.tsv", sep="\t")

    # Plot heatmap
    data = means.to_numpy()
    annotations = np.array([
        [f"{m:.1f} (± {s:.1f})" for m, s in zip(row_m, row_s)]
        for row_m, row_s in zip(means.to_numpy(), stds.to_numpy())
    ])

    formatted_metrics = [m.replace("&", "&\n") for m in display_metrics]

    plt.figure(figsize=(9, 6))
    sns.set(font_scale=0.9)

    ax = sns.heatmap(
        data,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"label": "Mean Score"},
        xticklabels=formatted_metrics,
        yticklabels=fixed_tool_order,
        annot_kws={"fontsize": 14, "color": "white"}
    )

    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(pad=0.5)
    plt.savefig(f"tool_metric_heatmap_{mode}.png", dpi=300, bbox_inches="tight")
    plt.close()
