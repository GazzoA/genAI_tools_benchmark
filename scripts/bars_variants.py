import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====== SELECT MODE HERE ======
mode = "germline"  # Options: "all", "germline", "somatic"
# ==============================

tool_colors = {
    "VarChat":    "#e41a1c",  # strong red
    "Gpt-4o":     "#377eb8",  # deep blue
    "MistralAI":  "#4daf4a",  # clear green
    "Perplexity": "#984ea3",  # purple
    "ScholarAI":  "#ff7f00"   # orange
}


df = pd.read_csv("evaluation_data.csv", sep=";")

# Filter by mode
if mode != "all":
    df = df[df["Type"].str.lower() == mode]

# Define variant label
df["Variant ID"] = df["Gene"] + " " + df["Variant"]

# Compute mean and SD per (Variant, Tool)
agg = (
    df.groupby(["Variant ID", "Tool name"])["Summarization Accuracy"]
    .agg(['mean', 'std'])
    .reset_index()
)

# Sort variants by global mean across tools
variant_order = (
    agg[agg["Tool name"] == "VarChat"]
    .sort_values("mean")["Variant ID"]
    .tolist()
)

agg["Variant ID"] = pd.Categorical(agg["Variant ID"], categories=variant_order, ordered=True)

# Plot
plt.figure(figsize=(16, 6))
sns.set(style="whitegrid")

ax = sns.pointplot(
    data=agg,
    x="Variant ID",
    y="mean",
    hue="Tool name",
    palette=tool_colors,
    dodge=0.5,
    markers="",
    linestyles="",
    errorbar=None
)


# Add manual error bars per point
positions = {}
for i, variant in enumerate(variant_order):
    tools = agg[agg["Variant ID"] == variant]["Tool name"].tolist()
    for j, tool in enumerate(tool_colors.keys()):
        if tool in tools:
            xpos = i - 0.3 + j * (0.15)  # adjust for dodge spacing
            row = agg[(agg["Variant ID"] == variant) & (agg["Tool name"] == tool)].iloc[0]
            plt.errorbar(
                x=xpos,
                y=row["mean"],
                yerr=row["std"],
                fmt='o',
                color=tool_colors[tool],
                capsize=4
            )

ax.set_ylabel("Summarization Accuracy (mean ± SD)")
ax.set_xlabel("")
ax.set_ylim(1, 5.49)
ax.set_xticks(range(len(variant_order)))
ax.set_xticklabels(variant_order, rotation=45, ha="right")
ax.legend(
    title="Tool",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=False
)
ax.set_xlim(-0.5, len(variant_order) - 0.5)

for i in range(len(variant_order)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#f0f0f0", zorder=0)



plt.subplots_adjust(left=0.05, right=0.97, bottom=0.25)


from matplotlib.lines import Line2D

# Custom legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=tool,
           markerfacecolor=color, markersize=8)
    for tool, color in tool_colors.items()
]

# Add custom legend
ax.legend(
    handles=legend_elements,
    title="Tool",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=False
)



os.makedirs("variant_pointplots", exist_ok=True)
plt.savefig(f"all_tools_accuracy_pointplot_{mode}.png", dpi=300, bbox_inches="tight")
plt.show()

plt.close()
