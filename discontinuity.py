import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats._stats_py



"""
PERMUTATION ENTROPY
"""

import_path = 'directory/complexity_perm.csv'

df = pd.read_csv(import_path)

df = df.loc[df.task == 'EC']

"""
Pearson Correlation
"""

# Select export path
export_path = 'export_directory/correlation.csv'

# Calculate correlation
results = (df.groupby)(["subject"])[["complexity", "discontinuity"]].mean().reset_index()
results = pg.rcorr(results, stars=False)

# Export correlation results
results.to_csv(export_path)

"""
Scatter Plot
"""

# Select export path
export_path = 'directory/avgs.csv'

# Drop NaN values from the dataframe and average across subject
avgs = df.groupby(["subject"])["complexity", "discontinuity"].mean().dropna()

# Export correlation results
avgs.to_csv(export_path)

# Create the scatterplot
fig, ax = plt.subplots()
sns.scatterplot(data=avgs, x="discontinuity", y="complexity", ax=ax, color="purple", alpha = 0.5)

# Add trendline and R-squared value
sns.regplot(data=avgs, x="discontinuity", y="complexity", scatter=False, ax=ax, ci=None, color = "black")
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(avgs["discontinuity"], avgs["complexity"])
plt.text(0.75, 0.05, fr"$r^2$ = {(r_value**2):.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_ylabel("Permutation Entropy Complexity")
ax.set_xlabel("Discontinuity")

# Select export path
export_path = 'directory/scatter_plt.png'

# Save plot
plt.savefig(export_path)

"""
Repeated measures correlation
"""

# Select export path
export_path = 'directory/plot_rm_corr.png'

# Removing the channels
no_channels = df.groupby(["subject", "session"])["complexity", "discontinuity"].mean().reset_index()


# Making the plot
g = pg.plot_rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')
g.set_axis_labels("Discontinuity", "Permutation Entropy Complexity")

# Export/save
g.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\plot_rm_corr.png')

# Runnning the stats
stats_EC = pg.rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')

# Select export path
export_path = 'directory/stats_rm_corr.csv'

# Save stats
stats_EC.to_csv(export_path)


"""
LEMPEL ZIV 
"""

# Import data
import_path = 'directory/complexity_lz.csv'
df_lz = pd.read_csv(import_path)

# Select resting state condition data only
df_lz = df_lz.loc[df_lz.task == 'EC']

"""
Pearson Correlation
"""

# Select export path
export_path = 'export_directory/correlation_lz.csv'

# Calculate correlation
results = (df.groupby)(["subject"])[["complexity", "discontinuity"]].mean().reset_index()
results = pg.rcorr(results, stars=False)

# Export correlation results
results.to_csv(export_path)

"""
Scatter Plot
"""

# Select export path
export_path = 'directory/avgs_lz.csv'

# Drop NaN values from the dataframe and average across subject
avgs = df.groupby(["subject"])["complexity", "discontinuity"].mean().dropna()

# Export correlation results
avgs.to_csv(export_path)

# Create the scatterplot
fig, ax = plt.subplots()
sns.scatterplot(data=avgs, x="discontinuity", y="complexity", ax=ax, color="purple", alpha = 0.5)

# Add trendline and R-squared value
sns.regplot(data=avgs, x="discontinuity", y="complexity", scatter=False, ax=ax, ci=None, color = "black")
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(avgs["discontinuity"], avgs["complexity"])
plt.text(0.75, 0.05, fr"$r^2$ = {(r_value**2):.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_ylabel("Permutation Entropy Complexity")
ax.set_xlabel("Discontinuity")

# Select export path
export_path = 'directory/scatter_plt_lz.png'

# Save plot
plt.savefig(export_path)

"""
Repeated measures correlation
"""

# Select export path
export_path = 'directory/plot_rm_corr_lz.png'

# Removing the channels
no_channels = df.groupby(["subject", "session"])["complexity", "discontinuity"].mean().reset_index()


# Making the plot
g = pg.plot_rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')
g.set_axis_labels("Discontinuity", "Permutation Entropy Complexity")

# Export/save
g.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\plot_rm_corr.png')

# Runnning the stats
stats_EC = pg.rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')

# Select export path
export_path = 'directory/stats_rm_corr_lz.csv'

# Save stats
stats_EC.to_csv(export_path)
