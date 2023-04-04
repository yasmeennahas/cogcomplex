# CogComplexity

## A research project investigating whether complexity is a valid marker for  cognitive control in EEG.

EEG data collected from [Wang et al., 2022.](https://www.nature.com/articles/s41597-022-01607-9)

```python
# Install antropy package to calculate complexity.

pip install antropy  #output: data_real_disc.csv

# Calculating entropy across time series.

python calculate_EEG_complexity.py

# Running topography analysis.

python topography.py

# Running statistical analysis.

python data_real_analysis.py

# Running discontinuity analysis.

python discontinuity.py

```
