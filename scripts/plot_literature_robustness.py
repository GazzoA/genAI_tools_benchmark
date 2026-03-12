import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== SELECT MODE HERE ======
mode = "all"  # Options: "all", "germline", "somatic"
# ==============================

fixed_tool_order = ["VarChat", "Gpt-4o", "MistralAI", "Perplexity", "ScholarAI"]

df = pd.read_csv("evaluation_data.csv", sep=";")

if mode != "all":
    df = df[df["Type"].str.lower() == mode]
df = df[df["References"].isin(["High", "Low"])]

original_metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]

display_metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]

high_color = "#386cb0"
low_color = "#beaed4"
line_color = "gray"
text_color = "black"

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
plt.subplots_adjust(wspace=0.25, hspace=0.4)
axes = axes.flatten()
plot_positions = [0, 1, 3, 4, 5]

for idx, (orig_metric, disp_metric) in enumerate(zip(original_metrics, display_metrics)):
    ax = axes[plot_positions[idx]]
    mean_scores = (
        df.groupby(["Tool name", "References"])[orig_metric]
        .mean()
        .unstack()
        .reindex(fixed_tool_order)
    )

    x = np.linspace(0, len(fixed_tool_order) - 1, len(fixed_tool_order))

    for i in range(len(fixed_tool_order)):
        tool = fixed_tool_order[i]
        high = mean_scores.loc[tool, "High"]
        low = mean_scores.loc[tool, "Low"]
        delta = high - low
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        label = f"Δ={delta:.1f} {direction}"

        ax.text(x[i] + 0.10, (high + low) / 2, label, va="center", fontsize=9, color=text_color)

        ax.plot([x[i], x[i]], [low, high], color=line_color, zorder=1)
        ax.scatter(x[i], high, color=high_color, s=60, label="High Literature" if i == 0 else "", zorder=2)
        ax.scatter(x[i], low, color=low_color, s=60, label="Low Literature" if i == 0 else "", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(fixed_tool_order, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(fixed_tool_order) - 0.2)
    ax.set_ylim(1, 5)
    ax.set_title(disp_metric, fontsize=11)
    if plot_positions[idx] % 3 == 0:
        ax.set_ylabel("Mean Score")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axes[2].axis("off")
axes[2].legend(
    handles=[
        plt.Line2D([0], [0], color=high_color, marker='o', linestyle='None', label='High Literature'),
        plt.Line2D([0], [0], color=low_color, marker='o', linestyle='None', label='Low Literature'),
        plt.Line2D([0], [0], color='none', marker='', linestyle='None', label='↑ = Higher in High Literature Context'),
        plt.Line2D([0], [0], color='none', marker='', linestyle='None', label='↓ = Higher in Low Literature Context')
    ],
    loc='upper left',
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"literature_robustness_dumbbell_2x3_topright_legend_{mode}.png", dpi=300)
plt.close()
