########## PLOT CREATION TO EVALUATE BENCHMARK RESULTS ##########

#### 1) REVIEWERS AGREEMENT PLOT - KENDALL-W ####
python agreement.py

#### 2) BARPLOTS FOR ALL TOOLS METRICS ####
python bars_tools_metric.py

#### 3) POINT-PLOTS ALL TOOLS ACCURACY  ####
python bars_variants.py

#### 4) BOXPLOT OVERALL P-VALUES FOR TOOLS METRICS DISTRIBUTIONS ####
python boxplot_pvalues.py

#### 5) HEATMAP FOR ALL TOOLS METRICS ####
python heatmap_metrics.py

#### 6) DUMBBELL PLOT FOR LITERATURE ROBUSTNESS ####
python plot_literature_robustness.py

#### 7) RADAR PLOTS FOR ALL TOOLS METRICS SPLIT BY LITERATURE COVERAGE ####
python radar_plots.py