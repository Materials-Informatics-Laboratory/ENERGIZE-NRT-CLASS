#!/usr/bin/env python3
from pathlib import Path
import json, re

HERE=Path(__file__).resolve().parent
meta=json.loads((HERE/'event_100_metadata.json').read_text())
jump_id=int(meta['jump_atom_id'])

def read_atoms(path):
    lines=Path(path).read_text().splitlines()
    start=None
    for i,line in enumerate(lines):
        if line.strip().startswith('Atoms'):
            start=i+1; break
    if start is None: raise RuntimeError(f'No Atoms section in {path}')
    out={}
    for line in lines[start:]:
        s=line.strip()
        if not s: continue
        if re.match(r'^[A-Za-z]',s): break
        tok=s.split()
        if len(tok)>=5 and tok[0].isdigit():
            out[int(tok[0])] = tuple(float(x) for x in tok[2:5])
    return out

final=read_atoms(HERE/'final.data')
if jump_id not in final: raise RuntimeError('Jump atom ID missing from final.data')
x,y,z=final[jump_id]
(HERE/'final.coords').write_text(f'1\n{jump_id} {x:.10f} {y:.10f} {z:.10f}\n')
print(f'Prepared final.coords for jump atom ID {jump_id}')
print(f'Final coordinate: {x:.6f} {y:.6f} {z:.6f} A')
print('Next: qsub submit.sh')
