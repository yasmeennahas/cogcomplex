"""
Run statistics (Wilcoxon test) between eyes-closed and math tasks at each EEG channel.
"""
import numpy as np
import pandas as pd
import pingouin as pg

from config import *


# Choose import/export paths
import_path = results_directory / "complexity.csv"
export_path = results_directory / "complexity_stats.csv"

# Load in data
df = pd.read_csv(import_path)

# # Getting the data for math and eyes closed only tasks
# df = df[df['task'].str.contains('EC|Ma', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = df.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# # Get the difference between eyes-closed and math for each subject/task/channel
# table = avgs.pivot(index=["subject", "channel"], columns="task", values="complexity").reset_index().rename_axis(None, axis=1)
# table["complexity_diff"] = table["EC"] - table["Ma"]
# diffs = table.drop(columns=["EC", "Ma"])

# Create list to hold results
results = []

# Loop over each channel
for c, cdf in avgs.groupby("channel"):
    # Grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Ma'")["complexity"].values
    # Wilcoxon text comparing complexity values across EC and Ma for all subjects at current channel
    wilc = pg.wilcoxon(x, y)
    wilc.index = pd.Index([c], name="channel")
    # Add mean difference value (to store later for plotting topography)
    wilc["diff"] = np.mean(x - y)
    # Append the wilcoxon analysis of all channel to results list
    results.append(wilc)

# Stack all results into one dataframe
stats = pd.concat(results).reset_index(drop=False)

# Add new p-value column that corrects for multiple comparisons
reject, corrected_pvalues = pg.multicomp(stats["p-val"], method="fdr_bh")
stats["p-val_corrected"] = corrected_pvalues

# Sort dataframe by p-values (lowest at top) for readability
stats = stats.sort_values("p-val")

# Export single dataframe holding all pairwise comparisons
stats.to_csv(export_path, index=False)
