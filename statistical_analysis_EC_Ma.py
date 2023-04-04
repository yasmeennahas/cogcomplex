import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real_disc.csv'

df = pd.read_csv(import_path)

# Getting the data for math and eyes closed only tasks

math_ec = df[df['task'].str.contains('EC|Ma', regex=True)]

############################################################################################
# Boxplots of difference in overall complexity across tasks and conditions
############################################################################################

# boxplot of complexity across all tasks
sns.boxplot(data = math_ec, x = 'task', y = 'complexity')
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\boxplot.png')
plt.show()

# boxplot of complexity across all tasks by session
sns.boxplot(data = math_ec, x = 'task', y = 'complexity', hue = 'session')
plt.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\boxplot_with_sesh.png')
plt.show()

############################################################################################
# rm_anova analysis across Ma and EC
############################################################################################

anova_rm = pg.rm_anova(data = math_ec, dv = 'complexity', within = ['task','session'], subject ='subject', detailed = True)
anova_rm.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\rm_anova_task.csv')

############################################################################################
# pairwise analysis across Ma and EC
############################################################################################

pairwise = pg.pairwise_tests(data = math_ec, dv = 'complexity', between = 'task', within = 'session', subject = 'subject')
pairwise.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\pairwise.csv')

############################################################################################
# Calculating difference in complexity between EC and Ma (you had told me to do this),
# idk what analysis to do or this tho, topomap probably no?
############################################################################################

# extract EC and Ma task complexity into separate DataFrames
ec_df = math_ec[math_ec['task'] == 'EC'].set_index(['subject', 'session', 'channel'])[['complexity']]
ma_df = math_ec[math_ec['task'] == 'Ma'].set_index(['subject', 'session', 'channel'])[['complexity']]

# calculate the difference in complexity between EC and Ma tasks for each channel and session
diff_math_ec = ec_df.subtract(ma_df, fill_value=0).reset_index()

############################################################################################
# wilcoxon analysis across the channels comparing EC and Ma
############################################################################################

#grouping math across the session

no_sesh_math_ec = math_ec.groupby(["subject", "channel", "task"]).agg({'complexity':'mean'}).reset_index()

# create list to hold results
results = []
# Loop over each channel...
for c, cdf in no_sesh_math_ec.groupby("channel"):
    # grab values for each one of the task (EC and Ma)
    x = cdf.query("task=='EC'")["complexity"].values
    y = cdf.query("task=='Ma'")["complexity"].values
    # conduct a wilcoxon analysis comparing complexity values across EC and Ma
    wilc = pg.wilcoxon(x, y)
    wilc.index = pd.Index([c], name="channel")
    # append the wilcoxon analysis to results
    results.append(wilc)
# stack all results into one dataframe
df = pd.concat(results).reset_index(drop=False)
df.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\math_ec\wilcoxon.csv')



