import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats._stats_py
from config import *


"""
PERMUTATION COMPLEXITY
"""

"""
Run statistics (repeated measures ANOVA test) between eyes-closed and cogntively engaging tasks (math/memory/music).
"""
# Choose import/export paths
import_path = results_directory / "perm_complexity.csv"
export_path = results_directory / "anova.csv"

# Load in data
df = pd.read_csv(import_path)

# Remove Eyes Open task
df = df.loc[df.task != 'EO']

# Select only first and second sessions
df = df.loc[(df.session == 1) | (df.session == 2)]

# Average across sessions and channels
no_ch_sesh = df.groupby(["subject", "task"])["complexity"].mean().reset_index()

# Define a dictionary to map old labels to new labels
label_map = {'EC': 'Resting State', 'Ma': 'Math', 'Me': 'Memory', 'Mu': 'Music'}

# Replace old labels with new labels in the 'task' column
no_ch_sesh['task'] = no_ch_sesh['task'].map(label_map)

### Anova of session and task

anova_rm_no_open = pg.rm_anova(data = no_ch_sesh, dv = 'complexity', within = ['task'], subject ='subject', detailed = True)

anova_rm_no_open.to_csv(export_path)

"""
Make a Boxplot comparing EEG perm complexity eyes-closed and cogntively engaging tasks (math/memory/music).
"""
# Choose export path
export_path = results_directory / "boxplot.png"

# Set the seaborn color palette to "cubehelix"
sns.set_palette('magma')

sns.boxplot(data = no_ch_sesh, x = 'task', y = 'complexity', boxprops=dict(alpha=.9))
plt.ylabel('Permutaion Entropy complexity')
plt.savefig(export_path)
plt.show()

"""
Run statistics (pairwise test) between eyes-closed and math tasks.
"""
# Choose export path
export_path = results_directory / "pairwise.csv"

### Pairwise tests across the all data  
pairwise = pg.pairwise_tests(data = no_ch_sesh, dv = 'complexity', within = ['task'], subject = 'subject')

# Add new p-value column that corrects for multiple comparisons
reject, corrected_pvalues = pg.multicomp(pairwise["p-unc"], method="fdr_bh")
pairwise["p-val_corrected"] = corrected_pvalues

# Sort dataframe by p-values (lowest at top) for readability
pairwise = pairwise.sort_values("p-unc")

# Export single dataframe holding all pairwise comparisons
pairwise.to_csv(export_path, index=False)


"""
Run statistics (Wilcoxon test) between eyes-closed and math tasks at each EEG channel.
"""

# MATH EC

# Choose export path
export_path = results_directory / "wilcoxon_ma.csv"

math_ec = df[df['task'].str.contains('EC|Ma', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = math_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

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


# MEMORY EC

# Choose export path
export_path = results_directory / "wilcoxon_me.csv"

me_ec = df[df['task'].str.contains('EC|Me', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = me_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# Create list to hold results
results = []

# Loop over each channel
for c, cdf in avgs.groupby("channel"):
    # Grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Me'")["complexity"].values
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


# MUSIC EC

# Choose export path
export_path = results_directory / "wilcoxon_mu.csv"

mu_ec = df[df['task'].str.contains('EC|Mu', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = mu_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# Create list to hold results
results = []

# Loop over each channel
for c, cdf in avgs.groupby("channel"):
    # Grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Mu'")["complexity"].values
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



"""
LEMPEL-ZIV COMPLEXITY
"""


"""
Run statistics (repeated measures ANOVA test) between eyes-closed and cogntively engaging tasks (math/memory/music).
"""
# Choose import/export paths
import_path = results_directory / "lz_complexity.csv"
export_path = results_directory / "anova_lz.csv"

# Load in data
df = pd.read_csv(import_path)

# Remove Eyes Open task
df = df.loc[df.task != 'EO']

# Select only first and second sessions
df = df.loc[(df.session == 1) | (df.session == 2)]

# Average across sessions and channels
no_ch_sesh = df.groupby(["subject", "task"])["complexity"].mean().reset_index()

# Define a dictionary to map old labels to new labels
label_map = {'EC': 'Resting State', 'Ma': 'Math', 'Me': 'Memory', 'Mu': 'Music'}

# Replace old labels with new labels in the 'task' column
no_ch_sesh['task'] = no_ch_sesh['task'].map(label_map)

### Anova of session and task

anova_rm_no_open = pg.rm_anova(data = no_ch_sesh, dv = 'complexity', within = ['task'], subject ='subject', detailed = True)

anova_rm_no_open.to_csv(export_path)

"""
Make a Boxplot comparing EEG perm complexity eyes-closed and cogntively engaging tasks (math/memory/music).
"""

# Choose export path
export_path = results_directory / "boxplot_lz.png"

# Set the seaborn color palette to "cubehelix"
sns.set_palette('magma')

sns.boxplot(data = no_ch_sesh, x = 'task', y = 'complexity', boxprops=dict(alpha=.9))
plt.ylabel('Permutaion Entropy complexity')
plt.savefig(export_path)
plt.show()

"""
Run statistics (pairwise test) between eyes-closed and math tasks.
"""
# Choose export path
export_path = results_directory / "pairwise_lz.csv"

### Pairwise tests across the all data  
pairwise = pg.pairwise_tests(data = no_ch_sesh, dv = 'complexity', within = ['task'], subject = 'subject')

# Add new p-value column that corrects for multiple comparisons
reject, corrected_pvalues = pg.multicomp(pairwise["p-unc"], method="fdr_bh")
pairwise["p-val_corrected"] = corrected_pvalues

# Sort dataframe by p-values (lowest at top) for readability
pairwise = pairwise.sort_values("p-unc")

# Export single dataframe holding all pairwise comparisons
pairwise.to_csv(export_path, index=False)


"""
Run statistics (Wilcoxon test) between eyes-closed and math tasks at each EEG channel.
"""

# MATH EC

# Choose export path
export_path = results_directory / "wilcoxon_ma_lz.csv"

math_ec = df[df['task'].str.contains('EC|Ma', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = math_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

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


# MEMORY EC

# Choose export path
export_path = results_directory / "wilcoxon_me_lz.csv"

me_ec = df[df['task'].str.contains('EC|Me', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = me_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# Create list to hold results
results = []

# Loop over each channel
for c, cdf in avgs.groupby("channel"):
    # Grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Me'")["complexity"].values
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


# MUSIC EC

# Choose export path
export_path = results_directory / "wilcoxon_mu_lz.csv"

mu_ec = df[df['task'].str.contains('EC|Mu', regex=True)]

# Average within all sessions of each task/subject/channel
avgs = mu_ec.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# Create list to hold results
results = []

# Loop over each channel
for c, cdf in avgs.groupby("channel"):
    # Grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Mu'")["complexity"].values
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



