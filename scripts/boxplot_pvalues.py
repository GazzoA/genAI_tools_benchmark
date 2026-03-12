import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
import itertools
from statannotations.Annotator import Annotator

# ====== SELECT MODE HERE ======
mode = "all"  # Options: "all", "somatic", "germline"
# ==============================

fixed_tool_order = ["VarChat", "Gpt-4o", "MistralAI", "Perplexity", "ScholarAI"]
tool_colors = {
    "VarChat":    "#e41a1c",  # strong red
    "Gpt-4o":     "#377eb8",  # deep blue
    "MistralAI":  "#4daf4a",  # clear green
    "Perplexity": "#984ea3",  # purple
    "ScholarAI":  "#ff7f00"   # orange
}


df = pd.read_csv("evaluation_data.csv", sep=";")

if mode == "somatic":
    df = df[df["Type"].str.lower() == "somatic"]
elif mode == "germline":
    df = df[df["Type"].str.lower() == "germline"]

metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]

kruskal_results = []
pairwise_results = []

fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=False)

for ax, metric in zip(axes, metrics):
    values = [g[metric].dropna().values for _, g in df.groupby("Tool name")]
    _, kruskal_p = kruskal(*values)
    kruskal_results.append(f"{metric}: p = {kruskal_p:.2e} ({'p < 0.0001' if kruskal_p < 1e-4 else ''})")

    dunn_df = sp.posthoc_dunn(df, val_col=metric, group_col="Tool name", p_adjust="bonferroni")
    dunn_df = dunn_df.reindex(index=fixed_tool_order, columns=fixed_tool_order)
    pairs = []
    pvalues = []

    for a, b in itertools.combinations(fixed_tool_order, 2):
        p = dunn_df.loc[a, b] if pd.notna(dunn_df.loc[a, b]) else dunn_df.loc[b, a]
        pairwise_results.append(f"{metric} | {a} vs {b}: p = {p:.3g}")
        if p < 0.05:
            pairs.append((a, b))
            pvalues.append(p)

    sns.boxplot(
        data=df,
        x="Tool name",
        y=metric,
        hue="Tool name",
        palette=tool_colors,
        order=fixed_tool_order,
        legend=False,
        showfliers=False,
        ax=ax,
        width=0.5
    )
    sns.stripplot(
        data=df,
        x="Tool name",
        y=metric,
        color="black",
        size=3,
        jitter=True,
        order=fixed_tool_order,
        ax=ax
    )

    if pairs:
        annotator = Annotator(ax, pairs, data=df, x="Tool name", y=metric, order=fixed_tool_order)
        annotator.configure(test=None, text_format="star", verbose=False)
        annotator.set_pvalues_and_annotate(pvalues)

    ax.set_ylim(bottom=1, top=ax.get_ylim()[1])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel("")
    ax.set_title("")
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel(metric)

plt.subplots_adjust(wspace=0.25)
plt.tight_layout(pad=0.5)
plt.savefig(f"all_metrics_{mode}_horizontal.png", dpi=300, bbox_inches="tight")
plt.close()

# Save Kruskal–Wallis overall p-values
with open(f"kruskal_overall_pvalues_{mode}.txt", "w") as f:
    f.write("Kruskal–Wallis H-test (Overall p-values per metric):\n")
    for line in kruskal_results:
        f.write(line + "\n")

# Save all pairwise p-values
with open(f"pairwise_dunn_pvalues_{mode}.txt", "w") as f:
    f.write("Dunn's post-hoc test (Bonferroni-corrected pairwise comparisons):\n")
    for line in pairwise_results:
        f.write(line + "\n")
