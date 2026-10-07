IN-CLASS PROBLEM 2: Predicting Experimental Lattice Constants from DFT

Target time: 30-40 minutes. No new VASP calculation is required.

1. Inspect data/lattice_constants_teaching.csv.
2. Run: python lattice_ml_student_template.py
3. Complete the TODOs.
4. Establish raw PBE error BEFORE fitting ML.
5. Train only on rows labeled train and evaluate on rows labeled test.
6. Compare raw PBE and ML-corrected errors and identify cases where ML got worse.

Key concept: ML learns delta_a = a_exp - a_PBE. It must demonstrate added value over an already-good physics baseline.
