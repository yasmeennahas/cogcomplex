import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats._stats_py


## PERM ENTROPY DATA ANALYSIS

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real_disc.csv'

df = pd.read_csv(import_path)

df = df.loc[df.task != 'EO']

# Averaging across channel complexity

no_channels = df.groupby(["subject", "session", "task"])["complexity"].mean().reset_index()

# Export correlation results
no_channels.to_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/no_channels.csv', index = False)

no_channels = pd.read_csv('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/no_channels.csv')

#boxplot of complexity across all channels in all tasks

# Define a dictionary to map old labels to new labels
label_map = {'EC': 'Resting State', 'Ma': 'Math', 'Me': 'Memory', 'Mu': 'Music'}

# Replace old labels with new labels in the 'task' column
df['task'] = df['task'].map(label_map)

sns.boxplot(data = df, x = 'task', y = 'complexity', hue = 'session')
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\task_complexity.png')
plt.show()

#anova of session and task

anova_rm_no_open = pg.rm_anova(data = no_channels, dv = 'complexity', within = ['task','session'], subject ='subject', detailed = True)

anova_rm_no_open.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\rm_anova_task_no_open.csv')


### paiwise tests across the all data  

pairwise = pg.pairwise_tests(data = no_channels, dv = 'complexity', between = 'task', within = 'session', subject = 'subject')

# Add new p-value column that corrects for multiple comparisons
reject, corrected_pvalues = pg.multicomp(pairwise["p-unc"], method="fdr_bh")
pairwise["p-val_corrected"] = corrected_pvalues

# Sort dataframe by p-values (lowest at top) for readability
pairwise = pairwise.sort_values("p-unc")

# Export single dataframe holding all pairwise comparisons
#stats.to_csv(export_path, index=False)

pairwise.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_all\pairwise.csv', index=False)
