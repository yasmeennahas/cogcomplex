"""CODE FOR COMPLEXITY CALCULATION IN EEG DATA

EEG complexity is calculated for across all channels for each relevant subject/session/task.
The output is a single dataframe with one row for each complexity value:
    subject session task    channel complexity
    1       1       EC      Fp1     0.952046758
    1       1       EC      AF3     0.964580791
    1       1       EC      AF7     0.960290503
    ...
"""
from pathlib import Path
import numpy as np
import mne
import antropy as ant
import pandas as pd
from tqdm import tqdm

from config import *


# Identify importing/exporting filepaths
export_filename = results_directory / "complexity.csv"
preproc_directory = data_directory / "derivatives" / "preprocessed data" / "preprocessed_data"

# Get all the EEG filenames.
eeg_filenames = preproc_directory.glob("*.set")
# Reduce EEG filenames to only eyes closed (EC) and math (Ma) tasks
eeg_filenames = [f for f in eeg_filenames if "EC" in f.stem or "Ma" in f.stem]

# Make empty list to append results
results = []

# Loop over all files and calculate complexity (for each channel) for each
for file in tqdm(eeg_filenames, desc="Complexity for all sessions"):

    # Parse out subject, session, and task information
    file_str = file.name
    subject = int(file_str[3:5])
    session = int(file_str[6:8])
    task = file_str[9:11]

    # Load in the EEG file and extract data as a numpy array
    raw = mne.io.read_raw_eeglab(file, preload=True)
    data = raw.get_data()

    # Get permutation entropy (i.e., complexity) for every channel
    complexity = np.apply_along_axis(ant.perm_entropy, axis=1, arr=data, normalize=True)

    df_ = pd.DataFrame(
        {
            "subject": subject,
            "session": session,
            "task": task,
            "channel": raw.ch_names,
            "complexity": complexity,
        }
    )

    results.append(df_)

# Concatenate all results together
df = pd.concat(results, ignore_index=True)

# Export single dataframe holding all results
df.to_csv(export_filename, index=False)
