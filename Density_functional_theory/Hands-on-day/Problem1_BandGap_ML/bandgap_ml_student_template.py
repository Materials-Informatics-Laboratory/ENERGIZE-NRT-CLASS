#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from vasp_gap_helper import electronic_gap_from_eigenval

ROOT=Path(__file__).resolve().parent

# PART A: read YOUR live VASP results
si=electronic_gap_from_eigenval(ROOT/'live_vasp'/'Si'/'EIGENVAL')
al=electronic_gap_from_eigenval(ROOT/'live_vasp'/'Al'/'EIGENVAL')
print('Si:',si)
print('Al:',al)

live=pd.read_csv(ROOT/'data'/'live_material_descriptors.csv')
# TODO 1: insert your calculated PBE gaps into live['pbe_gap_eV']

# PART B: inspect the training set
train=pd.read_csv(ROOT/'data'/'band_gap_training.csv')
# TODO 2: plot PBE gap vs experimental gap and add y=x

# PART C: define delta-learning target
# TODO 3: train['delta_gap_eV'] = experimental - PBE
features=['pbe_gap_eV','mean_atomic_number','electronegativity_range','volume_per_atom_A3']
# TODO 4: create X and y

# PART D: fixed train/test split
# TODO 5: use train_test_split(...,test_size=0.25,random_state=7)
# TODO 6: calculate a no-correction baseline MAE (predict delta=0)

# PART E: train a simple model
# model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
# TODO 7: fit model and calculate held-out ML MAE

# PART F: apply model to live Si and Al
# TODO 8: build X_live with the SAME columns, predict delta, then
#          corrected_gap = PBE_gap + predicted_delta

# PART G: did ML help?
# TODO 9: calculate raw-PBE and ML-corrected absolute errors for Si and Al

# PART H: final comparison figure
# TODO 10: make a plot comparing experiment, raw PBE, and ML-corrected gaps
#          for Si and Al. Save as problem1_bandgap_ML_comparison.png

# DISCUSS:
# 1. Was PBE equally accurate for Si and Al?
# 2. Did ML improve both?
# 3. Why is one constant correction for every material a poor model?
# 4. What would make you distrust an ML correction for a new material?
