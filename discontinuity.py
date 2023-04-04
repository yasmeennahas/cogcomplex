import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
import scipy.stats._stats_py

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real_disc.csv'

df = pd.read_csv(import_path)

# Getting the data for math and eyes closed only tasks

math_ec = df[df['task'].str.contains('EC|Ma', regex=True)]

############################################################################################
# correlation between discontinuity and complexity across tasks
############################################################################################

# Get a new dataframe that has 1 compl and 1 discont per participant (ie, averaging across sessions)

results = (math_ec
    # Get average across sessions for each task and subject combo
    .groupby(["task", "subject"])[["complexity", "discontinuity"]].mean()
    # Get correlation across task for each task
    .groupby("task").corr()
    # Organize columns (clean up)
    .droplevel(-1).drop(columns="complexity").reset_index()
    .drop_duplicates(subset="task").rename(columns={"discontinuity": "comp2disc_r"})
)

# Export correlation results
results.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\tasks_comp2disc_r.csv')

#looking at corr of complexity and discontinuity without parsing tasks

results = (math_ec
    # Get average across sessions for each task and subject combo
    .groupby(["task", "subject"])[["complexity", "discontinuity"]].mean().corr()
    # keeping only complexity
    .drop(columns="complexity").reset_index())

# Export correlation results
results.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\r_comp2disc.csv')


############################################################################################
# looking at scatter plot of discontinutiy and complexity 
############################################################################################

# Drop NaN values from the dataframe
math_ec = math_ec.dropna()

# Create the scatterplot
fig, ax = plt.subplots()
sns.scatterplot(data=math_ec, x="discontinuity", y="complexity", ax=ax)

# Add trendline and R-squared value
sns.regplot(data=math_ec, x="discontinuity", y="complexity", scatter=False, ax=ax)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(math_ec["discontinuity"], math_ec["complexity"])
plt.text(0.85, 0.05, f"R^2 = {r_value:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Show the plot
plt.show()
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\disc_complexity_scatter.png')

