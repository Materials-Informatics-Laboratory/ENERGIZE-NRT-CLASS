#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
base=HERE/'vacancy_barriers_99.csv'
if not base.exists():
    raise SystemExit('Missing vacancy_barriers_99.csv. Instructor must copy the real 99-event LAMMPS dataset here. For a software-only smoke test, run: python use_demo_dataset.py')
barrier_file=HERE.parent/'01_NEB'/'event_100_barrier.txt'
if not barrier_file.exists():
    raise SystemExit('Missing ../01_NEB/event_100_barrier.txt. Run analyze_neb.py first.')
barrier=float(barrier_file.read_text().strip())
old=pd.read_csv(base)
if len(old)!=99: raise SystemExit(f'Expected 99 rows in {base.name}; found {len(old)}')
feat=pd.read_csv(HERE/'event_100_features.csv')
if len(feat)!=1: raise SystemExit('event_100_features.csv must contain exactly one row')
row=feat.copy(); row['barrier_eV']=barrier; row['source']='student_LAMMPS_NEB'
# Match union of columns cleanly.
out=pd.concat([old,row],ignore_index=True,sort=False)
out.to_csv(HERE/'vacancy_barriers_100.csv',index=False)
print(f'Added event 100 barrier = {barrier:.6f} eV')
print('Wrote vacancy_barriers_100.csv with 100 events')
