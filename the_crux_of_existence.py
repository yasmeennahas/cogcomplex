from pathlib import Path
import numpy as np
import mne
import antropy as ant
import pandas as pd
from tqdm import tqdm
from mne.viz import plot_topomap
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pingouin as pg

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real.csv'

df = pd.read_csv(import_path)

df = df.loc[df.task != 'EO']

df = df.loc[(df.session == 1) | (df.session == 2)]


############################################################################################
# Data for each task
############################################################################################

#getting the mean complexity values by channel

EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]

EC_channels = list(EC.index)
Mu_channels = list(Mu.index)
Me_channels = list(Me.index)
Ma_channels = list(Ma.index)

#getting the postion for each electrode

data_directory = Path('C:/Users/Yasmeen/Desktop/thesis_project/data/data_thesis/derivatives/preprocessed data/preprocessed_data')

file_list = sorted(data_directory.glob("*.set"))

raw = mne.io.read_raw_eeglab(file_list[0], preload=True)

raw.info.set_montage('standard_1020')

#list of channels

channels = raw.info.ch_names

#selecting only the data where the channels are the same as that of the standard_1020

EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
EC = EC[EC.index.isin(channels)]
EC = zscore(EC)

Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
# EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
Mu = Mu - EC
Mu = Mu[Mu.index.isin(channels)]
Mu = zscore(Mu)

Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Me = Me - EC
Me = Me[Me.index.isin(channels)]
Me = zscore(Me)

Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]
Ma = Ma - EC
Ma = Ma[Ma.index.isin(channels)]
Ma = zscore(Ma)

############################################################################################
# topograph for each task
############################################################################################

# Marking the significant p-vals

# Re-importing the data

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real.csv'

df_new = pd.read_csv(import_path)

df_new = df_new.loc[(df_new.session == 1) | (df_new.session == 2)]

# MUSIC EC

mu_ec = df_new[df_new['task'].str.contains('EC|Mu', regex=True)]

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

stats['sig_channels'] = stats['p-val_corrected'] <= 0.05

sig_df = stats[['channel','sig_channels']]

# Get the list of channels in the same order as the Mu series
channel_order = Mu.index.tolist()

# Reindex the sig_df DataFrame to have the same order of channels as the Mu series
sig_df = sig_df.set_index('channel').reindex(channel_order).reset_index()




## MAKING TOPOPLOT

# Create a figure with subplots for the topomap plots and colorbar
fig, axes = plt.subplots(ncols=4, figsize=(12, 3), gridspec_kw={"width_ratios": [4,  4, 4, 1]})

# Get min and max for colorbar across al plots
vmin = min([ min(x) for x in [Mu, Ma, Me]])
vmax = max([ max(x) for x in [Mu, Ma, Me]])

# Display the EC topomap in the first subplot
# img_ec, quad = plot_topomap(EC, raw.info, cmap= "BuPu",  vlim = (vmin, vmax), axes=axes[0], show = False)
# axes[0].set_title('Resting State')

# Display the Mu topomap in the second subplot
img_mu, quad = plot_topomap(Mu, raw.info, cmap="BuPu", vlim = (vmin, vmax), mask = sig_df.sig_channels.to_numpy(), mask_params=dict(marker='x', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=5), axes=axes[0], show = False)
axes[0].set_title('Music')

# Display the Ma topomap in the third subplot
img_ma, quad = plot_topomap(Ma, raw.info, cmap="BuPu", vlim = (vmin, vmax), axes=axes[1], show = False)
axes[1].set_title('Math')

# Display the Me topomap in the fourth subplot
img_me, quad  = plot_topomap(Me, raw.info, cmap="BuPu", vlim = (vmin, vmax), axes=axes[2], show = False)
axes[2].set_title('Memory')

# Setting the label for the gradient bar
axes[3].set_title('Permutation Entropy\nComplexity')


plt.colorbar(mappable=img_mu, ax=axes[0:3], cax=axes[3])

plt.savefig('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/topo.png')



#### AND NOW FOR LZ


import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\lz_complexity.csv'

df = pd.read_csv(import_path)

df = df.loc[df.task != 'EO']

df = df.loc[(df.session == 1) | (df.session == 2)]


############################################################################################
# Data for each task
############################################################################################

#getting the mean complexity values by channel

EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]

EC_channels = list(EC.index)
Mu_channels = list(Mu.index)
Me_channels = list(Me.index)
Ma_channels = list(Ma.index)

#getting the postion for each electrode

data_directory = Path('C:/Users/Yasmeen/Desktop/thesis_project/data/data_thesis/derivatives/preprocessed data/preprocessed_data')

file_list = sorted(data_directory.glob("*.set"))

raw = mne.io.read_raw_eeglab(file_list[0], preload=True)

raw.info.set_montage('standard_1020')

#list of channels

channels = raw.info.ch_names

#selecting only the data where the channels are the same as that of the standard_1020

EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
EC = EC[EC.index.isin(channels)]
EC = zscore(EC)

Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
Mu = Mu - EC
Mu = Mu[Mu.index.isin(channels)]
Mu = zscore(Mu)

Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Me = Me - EC
Me = Me[Me.index.isin(channels)]
Me = zscore(Me)

Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]
Ma = Ma - EC
Ma = Ma[Ma.index.isin(channels)]
Ma = zscore(Ma)


############################################################################################
# topograph for each task
############################################################################################


## MAKING TOPOPLOT

# Create a figure with subplots for the topomap plots and colorbar
fig, axes = plt.subplots(ncols=4, figsize=(8, 3), gridspec_kw={"width_ratios": [4,  4, 4, 1]})

# Get min and max for colorbar across al plots
vmin = min([ min(x) for x in [Mu, Ma, Me]])
vmax = max([ max(x) for x in [Mu, Ma, Me]])

# Display the EC topomap in the first subplot
# img_ec, quad = plot_topomap(EC, raw.info, cmap= "BuPu",  vlim = (vmin, vmax), axes=axes[0], show = False)
# axes[0].set_title('Resting State')

# Display the Mu topomap in the second subplot
img_mu, quad = plot_topomap(Mu, raw.info, cmap="BuPu", vlim = (vmin, vmax), axes=axes[0], show = False)
axes[0].set_title('Music')

# Display the Ma topomap in the third subplot
img_ma, quad = plot_topomap(Ma, raw.info, cmap="BuPu", vlim = (vmin, vmax), axes=axes[1], show = False)
axes[1].set_title('Math')

# Display the Me topomap in the fourth subplot
img_me, quad  = plot_topomap(Me, raw.info, cmap="BuPu", vlim = (vmin, vmax), axes=axes[2], show = False)
axes[2].set_title('Memory')

# Setting the label for the gradient bar
axes[3].set_title('Lempel-Ziv \nComplexity')

plt.colorbar(mappable=img_mu, ax=axes[0:3], cax=axes[3])

plt.savefig('C:/Users/Yasmeen/Desktop/thesis_project/results/final_all/topo_lz.png')






