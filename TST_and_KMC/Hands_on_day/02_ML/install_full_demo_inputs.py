#!/usr/bin/env python3
"""Create demo inputs so the ML/KMC pipeline can be tested without LAMMPS."""
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
shutil.copy2(HERE/'demo'/'vacancy_barriers_99_DEMO_SYNTHETIC.csv',HERE/'vacancy_barriers_99.csv')
shutil.copy2(HERE/'demo'/'event_100_barrier_DEMO_SYNTHETIC.txt',HERE.parent/'01_NEB'/'event_100_barrier.txt')
print('Installed DEMO SYNTHETIC 99-event table and event-100 barrier.')
print('Now run: python add_event_100.py ; python train_barrier_model.py')
