# GenAI_tools_benchmark
Benchmarking generative AI tools for literature retrieval and summarization in genomic variant interpretation.

## Requirements

- **Python >= 3.8.10**
- **requirements.txt**
- `pip` and `venv` (or `conda`)

## Installation

### Option pip + venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
### Option conda
```bash
conda create -n genAI_benchmark python=3.8.1 -y
conda activate genAI_benchmark
python -m pip install --upgrade pip
```

### Install dependencies
```bash
pip install -r ./scripts/requirements.txt
```

## `scripts/` contents

### **Python scripts** for the following plots:

- `agreement.py`: Kendall-W evaluation of reviewers agreement
- `bars_tools_metric.py`: barplots for all tools metrics
- `bars_variants.py`: point-plots for all tools Summarization Accuracy
- `boxplot_pvalues.py`: boxplot of the overall p-values for tools metrics distributions
- `heatmap_metrics.py`: heatmaps for all tools metrics
- `plot_literature_robustness.py`: dumbbell plots for literature robustness
- `radar_plots.py`: radar plots for all tools metrics split by literature coverage

### Run `run_plot_creation.sh` to create all the plots
```bash
cd genAI_tools_benchmark/scripts
bash ./run_plot_creation.sh
```
The script automatically creates the directories with custom plots.

## `data/` contents
- `evaluation_data.csv`: dataset with evaluated variants
