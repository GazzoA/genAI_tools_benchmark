import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt

INPUT = "evaluation_data.csv"

METRICS = [
    "Summarization Accuracy",
    "Hallucination Robustness",
    "Readability & Usability",
    "Literature Coverage & Relevance",
    "Time Efficiency"
]


df = pd.read_csv(INPUT, sep="\t")
df["VariantID"] = df["Gene"] + "_" + df["Variant"]

def rank_tools(scores):
    return rankdata(-scores, method="average")

def kendall_w(rank_matrix):
    R = np.array(rank_matrix)
    n, m = R.shape   # n = tools, m = reviewers
    row_sums = np.sum(R, axis=1)
    S = np.sum((row_sums - np.mean(row_sums))**2)
    return 12*S/(m*m*(n**3-n))

def majority_winner(scores_df):
    """
    Determine whether ≥2 reviewers share the same top tool.
    Handles ties safely.
    """
    winners = []
    for col in scores_df.columns:
        max_val = scores_df[col].max()
        winners.append(set(scores_df.index[scores_df[col] == max_val]))

    # check if any tool appears in ≥2 winner sets
    all_tools = set.union(*winners)
    for tool in all_tools:
        count = sum(tool in w for w in winners)
        if count >= 2:
            return True
    return False

# -----------------------------
# 1. Tolerance agreement
# -----------------------------

tol_rows = []

for t in sorted(df.Type.unique()):

    dft = df[df.Type == t]

    for metric in METRICS:

        diffs = []
        within1 = []

        for _,g in dft.groupby(["VariantID","Tool name"]):

            vals = g[metric].dropna().values

            if len(vals) < 2:
                continue

            for a,b in combinations(vals,2):
                diffs.append(abs(a-b))

            within1.append(int(max(vals)-min(vals) <= 1))

        tol_rows.append({
            "Type":t,
            "Metric":metric,
            "MeanPairwiseDiff":np.mean(diffs),
            "Within1Rate":np.mean(within1)
        })

tol = pd.DataFrame(tol_rows)
tol.to_csv(f"tolerance_agreement.tsv",sep="\t",index=False)

# -----------------------------
# 2. Rank agreement (Kendall W)
# -----------------------------

rank_rows = []

for t in sorted(df.Type.unique()):

    dft = df[df.Type == t]

    for metric in METRICS:

        Ws = []
        majority = []

        for variant,gv in dft.groupby("VariantID"):

            pivot = gv.pivot(index="Tool name",columns="Reviewer",values=metric)

            if pivot.shape[1] < 3:
                continue

            ranks = pivot.apply(rank_tools)

            W = kendall_w(ranks.values)
            Ws.append(W)

            majority.append(majority_winner(pivot))

        rank_rows.append({
            "Type":t,
            "Metric":metric,
            "MeanKendallW":np.mean(Ws),
            "MajorityTopToolRate":np.mean(majority)
        })

rank = pd.DataFrame(rank_rows)
rank.to_csv(f"rank_agreement.tsv",sep="\t",index=False)

# -----------------------------
# 3. Final ranking robustness
# -----------------------------

final_rows = []

for t in sorted(df.Type.unique()):

    dft = df[df.Type == t]

    for metric in METRICS:

        reviewer_rankings = {}

        for r,gr in dft.groupby("Reviewer"):

            scores = gr.groupby("Tool name")[metric].mean()
            reviewer_rankings[r] = scores

        spears = []

        for r1,r2 in combinations(reviewer_rankings.keys(),2):

            rho,_ = spearmanr(reviewer_rankings[r1], reviewer_rankings[r2])
            spears.append(rho)

        winners = [x.idxmax() for x in reviewer_rankings.values()]

        final_rows.append({
            "Type":t,
            "Metric":metric,
            "MedianSpearman":np.median(spears),
            "MajoritySameWinner":len(set(winners))<=2
        })

final = pd.DataFrame(final_rows)
final.to_csv(f"final_ranking.tsv",sep="\t",index=False)

# -----------------------------
# Plot 1: Agreement summary
# -----------------------------

fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)

for i,t in enumerate(sorted(df.Type.unique())):

    tol_t = tol[tol.Type==t]
    rank_t = rank[rank.Type==t]

    x = np.arange(len(METRICS))

    axes[i].bar(x-0.15, rank_t["MeanKendallW"], width=0.3, label="Kendall's W")
    axes[i].bar(x+0.15, tol_t["Within1Rate"], width=0.3, label="Within ±1 agreement")

    axes[i].set_xticks(x)
    axes[i].set_xticklabels(METRICS, rotation=45, ha="right")
    axes[i].set_title(t)
    axes[i].set_ylim(0,1)

axes[0].set_ylabel("Agreement")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"Figure_S1_inter_rater_agreement.png",dpi=300)
plt.close()

# -----------------------------
# Plot 2: Final ranking robustness
# -----------------------------

fig, axes = plt.subplots(1,2,figsize=(12,5),sharey=True)

for i,t in enumerate(sorted(df.Type.unique())):

    fin = final[final.Type==t]

    x = np.arange(len(METRICS))

    axes[i].bar(x-0.15, fin["MedianSpearman"], width=0.3, label="Spearman ranking correlation")
    axes[i].bar(x+0.15, fin["MajoritySameWinner"].astype(int), width=0.3, label="≥2/3 same winner")

    axes[i].set_xticks(x)
    axes[i].set_xticklabels(METRICS, rotation=45, ha="right")
    axes[i].set_title(t)
    axes[i].set_ylim(0,1)

axes[0].set_ylabel("Robustness")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"Figure_S2_final_ranking_robustness.png",dpi=300)
plt.close()

