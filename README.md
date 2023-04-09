# CogComplex

A research project investigating whether complexity is a valid marker for cognitive control in EEG.


## Data

Data comes from a previously-published dataset ([Wang et al., 2022.](https://www.nature.com/articles/s41597-022-01607-9)). Download the dataset in it's entirety from [here](https://openneuro.org/datasets/ds004148/versions/1.0.1) and be sure to specify its location in the `config.py` file.


## Analyses

In addition to standard Python packages (e.g., `numpy`), the analysis scripts rely heavily on the `mne` and `antropy` packages (the latter is used to calculate complexity).

```shell
# Install MNE package for loading in EEG data.
pip install mne

# Install antropy package for calculating complexity.
pip install antropy
```

The `config.py` file holds global variables that are useful to have standardized across all analysis scripts. Mostly this is for specifying directory locations. Be sure to specify the proper directories in that file before running the following analysis code.

```shell
# Calculate entropy across time series.
python calculate_complexity.py  #output: complexity.csv (columns: subject, session, task, channel, complexity)

# Run topography analysis.
python topography.py  #output: topography.png(s) for each task

# Run statistical analysis.
python statistical_analysis_EC_Ma.py  #output: anova, pairwise, wilcoxon, boxplots for Math and Eyes Closed tasks only

# Run discontinuity analysis.
python discontinuity.py  #output: discontinuity to complexity scatterplot, r square, correlation
```
