from pathlib import Path
import numpy as np
import mne
import antropy as ant
import pandas as pd
from tqdm import tqdm
from mne.viz import plot_topomap
from scipy.stats import zscore

import_path = r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real.csv'

df = pd.read_csv(import_path)

df["complexity"] = zscore(df["complexity"])

############################################################################################
# Data for each task
############################################################################################

#getting the mean complexity values by channel

EO = df.loc[df.task == 'EO'].groupby(['channel']).mean()["complexity"]
EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]

EO_channels = list(EO.index)
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

EO = df.loc[df.task == 'EO'].groupby(['channel']).mean()["complexity"]
EO = EO[EO.index.isin(channels)]

EC = df.loc[df.task == 'EC'].groupby(['channel']).mean()["complexity"]
EC = EC[EC.index.isin(channels)]

Mu = df.loc[df.task == 'Mu'].groupby(['channel']).mean()["complexity"]
Mu = Mu[Mu.index.isin(channels)]

Me = df.loc[df.task == 'Me'].groupby(['channel']).mean()["complexity"]
Me = Me[Me.index.isin(channels)]

Ma = df.loc[df.task == 'Ma'].groupby(['channel']).mean()["complexity"]
Ma = Ma[Ma.index.isin(channels)]

############################################################################################
# topograph for each task
############################################################################################

fig = plot_topomap(EO, raw.info)
#fig.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\topo_EO.jpg')
#saving won' work, figure out later

fig = plot_topomap(EC, raw.info)
#fig.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\topo_EC.png')

fig = plot_topomap(Mu, raw.info)
#fig.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\topo_Mu.png')

fig = plot_topomap(Ma, raw.info)
#fig.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\topo_Ma.png')

fig = plot_topomap(Me, raw.info)
#fig.savefig(r'C:\Users\Yasmeen\Desktop\thesis_project\results\final_new_stuff\topo_Me.png')

############################################################################################
# topograph for Math and Eyes Closed
############################################################################################

# Create a figure with two subplots arranged side-by-side
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

# Display the first topomap plot in the first subplot
plot_topomap(EC, raw.info, axes=ax1)
ax1.set_title('EC Topomap')

# Display the second topomap plot in the second subplot
plot_topomap(Ma, raw.info, axes=ax2)
ax2.set_title('Ma Topomap')

# Add a title for the entire figure
fig.suptitle('Topomap Plots')

# Adding a colorbar
fig.colorbar()

# Show the figure
plt.show(block = False)
