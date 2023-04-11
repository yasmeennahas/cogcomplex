"""
Exploratory visualizations of EEG complexity.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *


# Load in data
import_path = results_directory / "complexity.csv"
df = pd.read_csv(import_path)

################################################################################
# Visualize complexity at each channel at every session.
################################################################################

g = sns.catplot(
    data=df, x="session", y="complexity", hue="task", col="channel",
    kind="point", dodge=True,
    col_wrap=10, height=1.2, aspect=1,
    sharex=True, sharey=False,
    errorbar="se",
)

# Aesthetic adjustments
g.set_titles("{col_name}")

# Export this plot.
export_path = results_directory / "complexity_plot-all_sess_chan.png"
plt.savefig(export_path, dpi=300)
plt.close()


################################################################################
# Visualize complexity at each channel, averaged across all sessions.
################################################################################

# Average within all sessions of each task/subject/channel
avg_sess = df.groupby(["subject", "task", "channel"])["complexity"].mean().reset_index()

# Average within all channels for each task/subject
avg_sess_chan = avg_sess.groupby(["subject", "task"])["complexity"].mean().reset_index()

# Boxplot of complexity across tasks
ax = sns.boxplot(data=avg_sess_chan, x="task", y="complexity")

# Aesthetic adjustments
ax.set_ylabel("complexity across all channels")

# Export this plot.
export_path = results_directory / "complexity_plot-avg_sess_chan.png"
plt.savefig(export_path, dpi=300)
plt.close()
