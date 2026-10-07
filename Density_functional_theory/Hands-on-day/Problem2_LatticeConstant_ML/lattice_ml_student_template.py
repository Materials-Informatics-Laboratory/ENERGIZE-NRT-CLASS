#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'data'/'lattice_constants_teaching.csv')
print(df.head())
print(df['split'].value_counts())

# PART A: DFT vs experiment
# TODO 1: plot pbe_lattice_A vs experimental_lattice_A and add y=x

# PART B: raw DFT baseline
# TODO 2: df['delta_a_A'] = experimental - PBE
# TODO 3: df['pbe_abs_error_A'] = abs(delta_a_A); calculate overall PBE MAE
# Optional: plot a histogram of delta_a_A

# PART C: fixed train/test sets
encoded=pd.get_dummies(df,columns=['structure_family'],dtype=float)
train=encoded[encoded['split']=='train'].copy()
test=encoded[encoded['split']=='test'].copy()
features=['pbe_lattice_A','mean_atomic_number','electronegativity_range']+[c for c in encoded.columns if c.startswith('structure_family_')]
# TODO 4: build X_train, y_train, X_test, y_test where y = delta_a_A

# PART D: baseline on held-out materials
# TODO 5: calculate test MAE if we make NO correction (prediction = raw PBE)

# PART E: ML delta correction
# model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
# TODO 6: fit model, predict delta_a for test set, and reconstruct corrected lattice constant

# PART F: compare errors
# TODO 7: calculate pbe_abs_error_A and ml_abs_error_A for each test material
# TODO 8: calculate raw-PBE test MAE and ML-corrected test MAE

# PART G: visualize
# TODO 9: make two panels: raw PBE vs experiment and ML-corrected vs experiment
#          Use the same axes and y=x line. Save problem2_lattice_ML_comparison.png

# PART H: inspect individual failures
# TODO 10: print material, PBE error, ML error, and whether ML improved the case

# DISCUSS:
# 1. Was raw PBE already good?
# 2. Did ML improve average held-out error?
# 3. Did ML improve every material?
# 4. Why is a train/test split essential?
# 5. Why might an unseen structure family be risky?
# 6. Why can PBE be quite good for lattice constants but poor for some band gaps?
