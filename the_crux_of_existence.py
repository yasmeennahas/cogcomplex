import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats._stats_py

## DISCONTINUITY STATS


import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real_disc.csv'

df = pd.read_csv(import_path)

df = df.loc[df.task == 'EC']


# Correlation

results = (df.groupby)(["subject"])[["complexity", "discontinuity"]].mean().reset_index()
results = pg.rcorr(results)

# Export correlation results
results.to_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/correlation.csv')


## R Squared Plot

# Drop NaN values from the dataframe

df_onechan = df.loc[df.channel == "O2"]

avgs = df_onechan.groupby(["subject"])["complexity", "discontinuity"].mean()

# Export correlation results

avgs.to_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/avgs.csv')

# Create the scatterplot
fig, ax = plt.subplots()
sns.scatterplot(data=avgs, x="discontinuity", y="complexity", ax=ax, color="purple", alpha = 0.5)

# Add trendline and R-squared value
sns.regplot(data=avgs, x="discontinuity", y="complexity", scatter=False, ax=ax, ci=None, color = "black")
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(avgs["discontinuity"], avgs["complexity"])
plt.text(0.85, 0.05, fr"$r^2$ = {(r_value**2):.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Show the plot
#plt.show()
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\disc_complexity_scatter_ec.png')


## Repeated measures


# Removing the channels

no_channels = df.groupby(["subject", "session"])["complexity", "discontinuity"].mean().reset_index()

# Export correlation results
no_channels.to_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/no_channels.csv', index = False)

no_channels = pd.read_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/no_channels.csv')

# Plot

pg.plot_rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')
# Export/save
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\plot_rm_corr.png')
plt.close()

# Stats

stats_EC = pg.rm_corr(data = no_channels, x ='discontinuity', y = 'complexity', subject = 'subject')
# Export/save
stats_EC.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\stats_rm_corr.csv')




