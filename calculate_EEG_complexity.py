###########################################################################
### CODE FOR COMPLEXITY CALCULATION IN EEG DATA
###########################################################################

from pathlib import Path
import numpy as np
import mne
import antropy as ant
import pandas as pd
from tqdm import tqdm

participants = pd.read_table(r"C:\Users\Yasmeen\Desktop\thesis_project\data\data_thesis\participants.tsv")

data_directory = Path('C:/Users/Yasmeen/Desktop/thesis_project/data/data_thesis/derivatives/preprocessed data/preprocessed_data')

file_list = sorted(data_directory.glob("*.set"))

results = []

for file in tqdm(file_list):

    # Load in the eeg file.
    try:
        raw = mne.io.read_raw_eeglab(file, preload=True)
    except OSError:
        print(f"Could not read file {file}")

    # Parse out subject information.
    file_str = file.name
    subject = int(file_str[3:5]) - 1
    session = int(file_str[6:8])
    task = file_str[9:11]
    
    # Getting discontinuity values for each row.
    participants.set_index('participant_id') 
    column_name = f'(1)Discontinuity of Mind_session{session}'
    disc_value = participants[column_name].iloc[subject]

    # Extract EEG data as numpy array.
    data = raw.get_data()

    # Get entropy for every channel.
    perm = np.apply_along_axis(ant.perm_entropy, axis=1, arr=data, normalize=True)

    df_ = pd.DataFrame(
        {
            "subject": subject,
            "session": session,
            "task": task,
            "channel": raw.ch_names,
            "complexity": perm,
            "discontinuity": disc_value
        }
    )

    results.append(df_)

df = pd.concat(results, ignore_index=True)

df.to_csv(r'C:\Users\Yasmeen\Desktop\thesis_project\results\data_real_disc.csv', index = False)
