#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
NREP=8

def numeric_tokens(line):
    try: return [float(x) for x in line.split()]
    except Exception: return None

def parse_log(path):
    lines=Path(path).read_text(errors='replace').splitlines()
    expected=9+2*NREP
    candidates=[]
    climbing=False
    for line in lines:
        if 'Setting up climbing' in line: climbing=True
        vals=numeric_tokens(line.strip())
        if vals is not None and len(vals)>=expected:
            candidates.append((climbing,vals))
    if not candidates:
        raise RuntimeError('Could not find NEB status lines in log.lammps. Check whether the job completed.')
    climb=[v for c,v in candidates if c]
    vals=(climb[-1] if climb else candidates[-1][1])
    ebf=float(vals[6]); ebr=float(vals[7]); rdt=float(vals[8])
    rd=np.array(vals[9:9+2*NREP:2],float)
    pe=np.array(vals[10:10+2*NREP:2],float)
    return ebf,ebr,rdt,rd,pe

log=HERE/'log.lammps'
if not log.exists():
    raise SystemExit('Missing log.lammps. Run the SCC NEB job first.')
eb,ebr,rdt,rd,pe=parse_log(log)
rel=pe-pe[0]
np.savetxt(HERE/'event_100_neb_profile.csv',np.c_[rd,pe,rel],delimiter=',',header='reaction_coordinate,energy_eV,relative_energy_eV',comments='')
(HERE/'event_100_barrier.txt').write_text(f'{eb:.10f}\n')
summary={'forward_barrier_eV':eb,'reverse_barrier_eV':ebr,'total_reaction_coordinate':rdt,'n_replicas':NREP}
(HERE/'event_100_neb_summary.json').write_text(json.dumps(summary,indent=2))
plt.figure(figsize=(6,4))
plt.plot(rd,rel,'o-',lw=2)
plt.xlabel('Normalized reaction coordinate')
plt.ylabel('Energy relative to initial state (eV)')
plt.title(f'Event 100 Al-Cu vacancy hop: NEB barrier = {eb:.3f} eV')
plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(HERE/'event_100_neb.png',dpi=200); plt.close()
print(f'Forward migration barrier: {eb:.6f} eV')
print(f'Reverse migration barrier: {ebr:.6f} eV')
print('Wrote event_100_barrier.txt and event_100_neb.png')
