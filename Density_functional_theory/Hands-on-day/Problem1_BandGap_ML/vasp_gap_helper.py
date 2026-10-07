from pathlib import Path
import numpy as np

def read_eigenval(path):
    lines=Path(path).read_text().splitlines()
    nelect,nkpts,nbands=[int(float(x)) for x in lines[5].split()[:3]]
    E=[]; O=[]; i=6
    for _ in range(nkpts):
        while i<len(lines) and not lines[i].strip(): i+=1
        i+=1
        ek=[]; ok=[]
        for _ in range(nbands):
            p=lines[i].split(); ek.append(float(p[1])); ok.append(float(p[2])); i+=1
        E.append(ek); O.append(ok)
    return nelect,np.array(E),np.array(O)

def electronic_gap_from_eigenval(path,partial_tol=0.05):
    _,E,occ=read_eigenval(path)
    partial=(occ>partial_tol)&(occ<(2.0-partial_tol))
    if np.any(partial):
        return {'metallic':True,'gap_eV':0.0,'VBM_eV':np.nan,'CBM_eV':np.nan}
    filled=E[occ>1.0]; empty=E[occ<=1.0]
    vbm=float(filled.max()); cbm=float(empty.min())
    return {'metallic':False,'gap_eV':max(0.0,cbm-vbm),'VBM_eV':vbm,'CBM_eV':cbm}
