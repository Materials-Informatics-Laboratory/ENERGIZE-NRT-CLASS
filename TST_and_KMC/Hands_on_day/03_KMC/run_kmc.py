#!/usr/bin/env python3
"""On-lattice vacancy KMC using the trained local-environment barrier model.

One invocation automatically scans all requested temperatures and the three
initial vacancy-placement scenarios.  Vacancy identities are tracked only as
walkers; 'Al-site' and 'Cu-site' describe where vacancies are INITIALIZED.
"""
from pathlib import Path
import sys, argparse, json, math
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'common'))
from alcu_lattice import make_fcc_sites, build_topology, AL, CU, VAC, NN1_VECTORS

KB=8.617333262145e-5  # eV/K
A0=4.032
NX=NY=NZ=16          # 16,384 FCC sites
CU_FRACTION=0.10
N_VACANCIES=8
ATTEMPT_FREQUENCY=1.0e13  # s^-1
DEFAULT_TEMPS=[500,600,700,800,900]
DEFAULT_SCENARIOS=['al_sites','cu_sites','mixed']
DEFAULT_EVENTS=5000
FEATURES=['jump_species','start_1nn_Cu','start_2nn_Cu','end_1nn_Cu','end_2nn_Cu','shared_1nn_Cu']

def build_barrier_lookup(model):
    """Predict every discrete descriptor combination once, vectorized.

    The descriptors are small integer coordination counts.  Precomputing this
    table makes the subsequent KMC loop extremely cheap and illustrates the
    idea that an ML surrogate can be queried many times after training.
    """
    rows=[]; keys=[]
    for species in ('Al','Cu'):
        for s1 in range(13):
            for s2 in range(7):
                for e1 in range(13):
                    for e2 in range(7):
                        for sh in range(5):
                            rows.append((species,s1,s2,e1,e2,sh)); keys.append((0 if species=='Al' else 1,s1,s2,e1,e2,sh))
    X=pd.DataFrame(rows,columns=FEATURES)
    pred=np.asarray(model.predict(X),float)
    table=np.empty((2,13,7,13,7,5),dtype=np.float32)
    for key,val in zip(keys,pred): table[key]=val
    return table

def choose_nonadjacent(candidates,n,nn1,rng,forbidden=None):
    candidates=list(map(int,candidates)); rng.shuffle(candidates); chosen=[]; blocked=set() if forbidden is None else set(map(int,forbidden))
    for s in candidates:
        if s in blocked: continue
        chosen.append(s); blocked.add(s); blocked.update(int(x) for x in nn1[s])
        if len(chosen)==n: return chosen
    raise RuntimeError('Could not choose enough non-adjacent vacancy sites.')

def base_alloy(nsites,rng):
    occ=np.full(nsites,AL,dtype=np.int8)
    ncu=int(round(CU_FRACTION*nsites)); occ[rng.choice(nsites,ncu,replace=False)]=CU
    return occ

def initial_state(base,scenario,nn1,seed):
    rng=np.random.default_rng(seed); occ=base.copy(); forbidden=set(); vacs=[]
    if scenario=='al_sites':
        vacs=choose_nonadjacent(np.where(occ==AL)[0],N_VACANCIES,nn1,rng)
    elif scenario=='cu_sites':
        vacs=choose_nonadjacent(np.where(occ==CU)[0],N_VACANCIES,nn1,rng)
    elif scenario=='mixed':
        na=N_VACANCIES//2; nc=N_VACANCIES-na
        a=choose_nonadjacent(np.where(occ==AL)[0],na,nn1,rng)
        blocked=set(a)
        for s in a: blocked.update(int(x) for x in nn1[s])
        c=choose_nonadjacent(np.where(occ==CU)[0],nc,nn1,rng,blocked)
        vacs=a+c
    else: raise ValueError(scenario)
    for s in vacs: occ[s]=VAC
    return occ,np.array(vacs,dtype=np.int32)

def cu_counts(occ,nn): return np.sum(occ[nn]==CU,axis=1).astype(np.int16)

def run_one(T,scenario,n_events,barrier_table,sites,dims,nn1,nn2,shared,base,seed,outdir):
    rng=np.random.default_rng(seed)
    occ,vacs=initial_state(base,scenario,nn1,seed+17)
    cu1=cu_counts(occ,nn1); cu2=cu_counts(occ,nn2)
    unwrapped=np.zeros((len(vacs),3),dtype=float)
    time_s=0.0
    sample_every=max(1,n_events//250)
    samples=[(0,0.0,0.0)]
    barrier_rows=[]
    seen_environments=set()
    kbt=KB*T
    for event in range(1,n_events+1):
        candidates=[]; rates=[]
        for vid,v in enumerate(vacs):
            v=int(v)
            s1=int(cu1[v]); s2=int(cu2[v])
            for slot,j in enumerate(nn1[v]):
                j=int(j); sp=int(occ[j])
                if sp==VAC: continue
                e1=int(cu1[j]) + (1 if sp==CU else 0)
                e2=int(cu2[j])
                sh=int(np.count_nonzero(occ[shared[v,slot]]==CU))
                species_name='Cu' if sp==CU else 'Al'
                key=(species_name,s1,s2,e1,e2,sh)
                barrier=float(barrier_table[1 if sp==CU else 0,s1,s2,e1,e2,sh])
                rate=ATTEMPT_FREQUENCY*math.exp(-barrier/kbt)
                candidates.append((vid,v,j,slot,sp,barrier,key)); rates.append(rate); seen_environments.add(key)
        rates=np.asarray(rates,float); total=float(rates.sum())
        if not np.isfinite(total) or total<=0: raise RuntimeError('No finite KMC rate available.')
        u=max(rng.random(),1e-15); time_s += -math.log(u)/total
        pick=float(rng.random())*total; idx=int(np.searchsorted(np.cumsum(rates),pick,side='right'))
        if idx>=len(candidates): idx=len(candidates)-1
        vid,v,j,slot,sp,barrier,key=candidates[idx]
        # Vacancy moves from v to j. The minimum-image hop vector is known from the FCC slot.
        unwrapped[vid] += NN1_VECTORS[slot].astype(float)*(A0/2.0)
        # Update Cu-neighbor counts only when a Cu atom moved j -> v.
        if sp==CU:
            cu1[nn1[v]] += 1; cu1[nn1[j]] -= 1
            cu2[nn2[v]] += 1; cu2[nn2[j]] -= 1
        occ[v]=sp; occ[j]=VAC; vacs[vid]=j
        if event % max(1,n_events//5000)==0:
            barrier_rows.append((event,time_s,barrier,key[0],key[1],key[2],key[3],key[4],key[5]))
        if event % sample_every==0 or event==n_events:
            msd=float(np.mean(np.sum(unwrapped**2,axis=1)))
            samples.append((event,time_s,msd))
    sdf=pd.DataFrame(samples,columns=['event','time_s','msd_A2'])
    # Long-time linear fit: final 60% of sampled points.
    start=max(1,int(.40*len(sdf))); x=sdf['time_s'].to_numpy()[start:]; y=sdf['msd_A2'].to_numpy()[start:]
    slope,intercept=np.polyfit(x,y,1); D_A2_s=max(0.0,float(slope)/6.0); D_m2_s=D_A2_s*1e-20
    tag=f'{scenario}_T{int(T)}'
    sdf.to_csv(outdir/f'msd_{tag}.csv',index=False)
    pd.DataFrame(barrier_rows,columns=['event','time_s','accepted_barrier_eV','jump_species','start_1nn_Cu','start_2nn_Cu','end_1nn_Cu','end_2nn_Cu','shared_1nn_Cu']).to_csv(outdir/f'accepted_{tag}.csv',index=False)
    return {'scenario':scenario,'temperature_K':T,'events':n_events,'vacancies':N_VACANCIES,'final_time_s':time_s,'D_v_m2_s':D_m2_s,'unique_ML_environments':len(seen_environments)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--events',type=int,default=DEFAULT_EVENTS); ap.add_argument('--temperatures',nargs='+',type=float,default=DEFAULT_TEMPS); ap.add_argument('--scenarios',nargs='+',default=DEFAULT_SCENARIOS); ap.add_argument('--quick',action='store_true')
    args=ap.parse_args(); n_events=1000 if args.quick else args.events
    model_path=HERE.parent/'02_ML'/'barrier_model.joblib'
    if not model_path.exists(): raise SystemExit('Missing ../02_ML/barrier_model.joblib. Train the ML model first.')
    model=joblib.load(model_path)
    print('Precomputing ML barrier lookup over discrete local environments...')
    barrier_table=build_barrier_lookup(model)
    outdir=HERE/'kmc_output'; outdir.mkdir(exist_ok=True)
    print('Building 16,384-site FCC topology once...')
    sites,dims,site_map=make_fcc_sites(NX,NY,NZ); nn1,nn2,shared=build_topology(sites,dims,site_map)
    base=base_alloy(len(sites),np.random.default_rng(112358))
    rows=[]
    for si,scenario in enumerate(args.scenarios):
        for ti,T in enumerate(args.temperatures):
            print(f'KMC: {scenario:8s} T={T:g} K, {n_events} accepted events')
            row=run_one(T,scenario,n_events,barrier_table,sites,dims,nn1,nn2,shared,base,seed=10000+si*100+ti,outdir=outdir)
            rows.append(row); print(f"  D_v = {row['D_v_m2_s']:.4e} m^2/s; physical time = {row['final_time_s']:.3e} s")
    pd.DataFrame(rows).to_csv(outdir/'run_summary.csv',index=False)
    config={'temperatures_K':args.temperatures,'scenarios':args.scenarios,'events_per_case':n_events,'n_vacancies':N_VACANCIES,'fcc_sites':len(sites),'Cu_fraction':CU_FRACTION,'attempt_frequency_s-1':ATTEMPT_FREQUENCY}
    (outdir/'run_config.json').write_text(json.dumps(config,indent=2))
    print(f'Finished {len(rows)} KMC cases. Wrote {outdir/"run_summary.csv"}')
if __name__=='__main__': main()
