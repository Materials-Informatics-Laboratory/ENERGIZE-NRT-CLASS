#!/usr/bin/env python3
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
src=HERE/'demo'/'vacancy_barriers_99_DEMO_SYNTHETIC.csv'
dst=HERE/'vacancy_barriers_99.csv'
shutil.copy2(src,dst)
print('WARNING: copied SYNTHETIC DEMO barriers into vacancy_barriers_99.csv.')
print('Use only to test the software pipeline. Replace with the collected LAMMPS dataset before class.')
