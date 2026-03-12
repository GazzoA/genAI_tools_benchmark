import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

tools = ['Gpt-4o', 'MistralAI', 'ScholarAI', 'Perplexity', 'VarChat']

tool_colors = {
    "VarChat":    "#e41a1c",
    "Gpt-4o":     "#377eb8",
    "MistralAI":  "#4daf4a",
    "Perplexity": "#984ea3",
    "ScholarAI":  "#ff7f00"
}

metrics = [
    'Summarization Accuracy',
    'Hallucination Robustness',
    'Readability & Usability',
    'Literature Coverage & Relevance',
    'Time Efficiency'
]


metrics_to_show_radar = [
    'Summarization\n Accuracy',
    'Hallucination\n Robustness',
    'Readability\n & Usability',
    'Literature Coverage\n & Relevance',
    'Time Efficiency'
]

def discretize_value(x):
    if 0 <= x <= 9:
        return 'Low'
    elif 10 <= x < 25:
        return 'Medium'
    elif x >= 25:
        return 'High'

def analyze_results_literature(results, type_dataset):
    area_data = []

    literature_levels = ['Low', 'Medium', 'High']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axs = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(18, 6))

    for idx, level in enumerate(literature_levels):
        subset = results[results['References'] == level]
        if subset.empty:
            continue
        mean_values = subset.groupby("Tool name")[metrics].mean()

        for tool, row in mean_values.iterrows():
            values = row.tolist()
            values += values[:1]

            r = np.array(values)
            theta = np.array(angles)
            area = 0.5 * np.abs(np.dot(r[:-1], np.sin(np.diff(theta)) * r[1:]))

            area_data.append({
                "Data": type_dataset,
                "Literature": level,
                "Tool": tool,
                "Area": area
            })

        area_df = pd.DataFrame(area_data)

        ax = axs[idx]
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'])

        for tool, row in mean_values.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=tool, color=tool_colors[tool])
            ax.fill(angles, values, alpha=0.10, color=tool_colors[tool])
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_to_show_radar)
        ax.set_title(f"Literature: {level}", size=14, y=1.1)
        ax.tick_params(pad=15)
        if idx == 2:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(f'radar_plot_all_literature_{type_dataset}.png'), bbox_inches='tight')
    plt.close()

    return area_df

def plots_generation():
    data = pd.read_csv("evaluation_data.csv", sep=";")

    somatic = data[data['Type'] == 'Somatic']
    germline = data[data['Type'] == 'germline']

    area_df_general = analyze_results_literature(data, 'general')
    area_df_somatic = analyze_results_literature(somatic, 'somatic')
    area_df_germline = analyze_results_literature(germline, 'germline')
    area_df = pd.concat([area_df_general, area_df_somatic, area_df_germline])
    area_df.to_csv("radar_areas.csv", index=False)

if __name__ == '__main__':
    plots_generation()
