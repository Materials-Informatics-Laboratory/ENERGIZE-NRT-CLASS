#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KB=8.617333262145e-5
HERE=Path(__file__).resolve().parent; OUT=HERE/'kmc_output'
summary_file=OUT/'run_summary.csv'
if not summary_file.exists(): raise SystemExit('Missing kmc_output/run_summary.csv. Run run_kmc.py first.')
df=pd.read_csv(summary_file)
scenarios=list(df['scenario'].drop_duplicates())
# D vs T
plt.figure(figsize=(6.4,4.4))
for s in scenarios:
    d=df[df.scenario==s].sort_values('temperature_K'); plt.semilogy(d.temperature_K,d.D_v_m2_s,'o-',label=s)
plt.xlabel('Temperature (K)'); plt.ylabel(r'Vacancy diffusivity $D_v$ (m$^2$/s)'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'diffusivity_vs_temperature.png',dpi=200); plt.close()
# Arrhenius
fits=[]; plt.figure(figsize=(6.4,4.4))
for s in scenarios:
    d=df[(df.scenario==s)&(df.D_v_m2_s>0)].sort_values('temperature_K'); x=1.0/d.temperature_K.to_numpy(); y=np.log(d.D_v_m2_s.to_numpy())
    if len(d)>=2:
        m,b=np.polyfit(x,y,1); Ea=-m*KB; D0=float(np.exp(b)); fits.append({'scenario':s,'effective_Ea_eV':Ea,'D0_m2_s':D0}); xx=np.linspace(x.min(),x.max(),100); plt.plot(1000*xx,m*xx+b,'--',alpha=.6)
    plt.scatter(1000*x,y,label=s)
plt.xlabel(r'$1000/T$ (K$^{-1}$)'); plt.ylabel(r'$\ln D_v$'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'arrhenius_vacancy_diffusion.png',dpi=200); plt.close()
pd.DataFrame(fits).to_csv(OUT/'arrhenius_summary.csv',index=False)
# MSD at the middle scanned T (or closest to 700 K)
Ts=np.array(sorted(df.temperature_K.unique())); Tsel=float(Ts[np.argmin(np.abs(Ts-700))]); plt.figure(figsize=(6.4,4.4))
for s in scenarios:
    p=OUT/f'msd_{s}_T{int(Tsel)}.csv'
    if p.exists():
        q=pd.read_csv(p); plt.plot(q.time_s,q.msd_A2,label=s)
plt.xlabel('KMC physical time (s)'); plt.ylabel(r'Vacancy MSD ($\AA^2$)'); plt.title(f'Vacancy MSD at {Tsel:g} K'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/'msd_comparison.png',dpi=200); plt.close()
# Optional experimental overlay, only if instructor supplies meaningful vacancy-mobility data.
ref=HERE/'experimental_reference_OPTIONAL.csv'
if ref.exists():
    r=pd.read_csv(ref).dropna()
    if len(r):
        plt.figure(figsize=(6.4,4.4))
        for s in scenarios:
            d=df[df.scenario==s].sort_values('temperature_K'); plt.semilogy(d.temperature_K,d.D_v_m2_s,'o-',label=f'KMC {s}')
        plt.semilogy(r.temperature_K,r.diffusivity_m2_s,'ks',label='experimental reference'); plt.xlabel('Temperature (K)'); plt.ylabel(r'Diffusivity (m$^2$/s)'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'kmc_vs_experiment.png',dpi=200); plt.close()
print('Wrote diffusivity_vs_temperature.png, arrhenius_vacancy_diffusion.png, msd_comparison.png')
print('Wrote arrhenius_summary.csv')
