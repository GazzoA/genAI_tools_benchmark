import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fixed_tool_order = ["VarChat", "Gpt-4o", "MistralAI", "Perplexity", "ScholarAI"]
tool_colors = {
    "VarChat":    "#e41a1c",
    "Gpt-4o":     "#377eb8",
    "MistralAI":  "#4daf4a",
    "Perplexity": "#984ea3",
    "ScholarAI":  "#ff7f00"
}

df = pd.read_csv("evaluation_data.csv", sep=";")

metrics = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]


for mode in ["all", "germline", "somatic"]:
    if mode == "germline":
        df_mode = df[df["Type"].str.lower() == "germline"]
    elif mode == "somatic":
        df_mode = df[df["Type"].str.lower() == "somatic"]
    else:
        df_mode = df.copy()

    df_long = df_mode.melt(
        id_vars=["Tool name", "Type"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    df_long["Metric"] = df_long["Metric"].str.replace(" & ", " &\n", regex=False)
    df_long["Tool name"] = pd.Categorical(df_long["Tool name"], categories=fixed_tool_order, ordered=True)

    plt.figure(figsize=(11, 6))
    sns.set(style="whitegrid", font_scale=0.9)
    ax = sns.barplot(
        data=df_long,
        x="Metric",
        y="Score",
        hue="Tool name",
        hue_order=fixed_tool_order,
        palette=tool_colors,
        errorbar="ci"
    )

    plt.ylabel("Mean Score")
    plt.ylim(1, 5)
    plt.xticks(rotation=0, ha="center")
    plt.xlabel("")
    plt.legend(title="Tool", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"tool_metric_barplot_{mode}.png", dpi=300)
    plt.close()
