#!/usr/bin/env python3
"""Shared FCC Al-Cu lattice utilities for the vacancy/ML/KMC activity.

Lattice coordinates are integer half-lattice coordinates: Cartesian r = (a/2)*g.
FCC sites are those integer triples with even i+j+k.  This makes nearest-neighbor
and periodic on-lattice operations exact and simple.
"""
from __future__ import annotations
import numpy as np

AL, CU, VAC = 1, 2, 0
NN1_VECTORS = np.array([
    (sx, sy, 0) for sx in (-1,1) for sy in (-1,1)
] + [
    (sx, 0, sz) for sx in (-1,1) for sz in (-1,1)
] + [
    (0, sy, sz) for sy in (-1,1) for sz in (-1,1)
], dtype=int)
NN2_VECTORS = np.array([
    (2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)
], dtype=int)

FEATURE_COLUMNS = [
    'jump_species', 'start_1nn_Cu', 'start_2nn_Cu',
    'end_1nn_Cu', 'end_2nn_Cu', 'shared_1nn_Cu'
]

def make_fcc_sites(nx: int, ny: int, nz: int):
    dims = np.array([2*nx, 2*ny, 2*nz], dtype=int)
    sites = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if (i+j+k) % 2 == 0:
                    sites.append((i,j,k))
    sites = np.asarray(sites, dtype=int)
    site_map = {tuple(g): idx for idx,g in enumerate(sites)}
    return sites, dims, site_map

def build_topology(sites, dims, site_map):
    n = len(sites)
    nn1 = np.empty((n,12), dtype=np.int32)
    nn2 = np.empty((n,6), dtype=np.int32)
    shared = np.empty((n,12,4), dtype=np.int32)
    # Common-neighbor geometry depends only on the oriented FCC hop, so derive
    # the four relative common-neighbor vectors once instead of intersecting
    # Python sets for every site in a large KMC lattice.
    nnset={tuple(v.tolist()) for v in NN1_VECTORS}
    common_rel=[]
    for d in NN1_VECTORS:
        rel=[q.copy() for q in NN1_VECTORS if tuple((q-d).tolist()) in nnset]
        if len(rel)!=4: raise RuntimeError('FCC geometry should have four shared 1NN sites per nearest-neighbor pair.')
        common_rel.append(rel)
    for s,g in enumerate(sites):
        for k,d in enumerate(NN1_VECTORS):
            gg = tuple(((g+d) % dims).tolist())
            nn1[s,k] = site_map[gg]
            for m,q in enumerate(common_rel[k]):
                shared[s,k,m] = site_map[tuple(((g+q)%dims).tolist())]
        for k,d in enumerate(NN2_VECTORS):
            gg = tuple(((g+d) % dims).tolist())
            nn2[s,k] = site_map[gg]
    return nn1, nn2, shared

def periodic_delta_grid(g_from, g_to, dims):
    d = np.asarray(g_to, dtype=int) - np.asarray(g_from, dtype=int)
    for q in range(3):
        L = int(dims[q])
        if d[q] > L//2: d[q] -= L
        if d[q] < -L//2: d[q] += L
    return d

def descriptor_for_event(occ, vac_site, jump_site, nn1, nn2, shared_neighbors=None, jump_neighbor_slot=None):
    species = int(occ[jump_site])
    if species not in (AL, CU):
        raise ValueError('Jump site must contain Al or Cu.')
    start1 = int(np.count_nonzero(occ[nn1[vac_site]] == CU))
    start2 = int(np.count_nonzero(occ[nn2[vac_site]] == CU))
    # After hop, old vacancy site is occupied by the jumping atom and the old
    # atom site is the new vacancy.  The old vacancy is a 1NN of jump_site.
    end1 = int(np.count_nonzero(occ[nn1[jump_site]] == CU)) + (1 if species == CU else 0)
    end2 = int(np.count_nonzero(occ[nn2[jump_site]] == CU))
    if shared_neighbors is not None and jump_neighbor_slot is not None:
        common = shared_neighbors[vac_site, jump_neighbor_slot]
    else:
        common = np.intersect1d(nn1[vac_site], nn1[jump_site], assume_unique=False)
    shared_cu = int(np.count_nonzero(occ[common] == CU))
    return {
        'jump_species': 'Cu' if species == CU else 'Al',
        'start_1nn_Cu': start1,
        'start_2nn_Cu': start2,
        'end_1nn_Cu': end1,
        'end_2nn_Cu': end2,
        'shared_1nn_Cu': shared_cu,
    }

def write_lammps_data(path, sites, dims, occ, a, moved_atom=None):
    """Write atomic LAMMPS data. moved_atom=(site_index, destination_site_index) optional."""
    path = str(path)
    occupied_sites = [i for i,x in enumerate(occ) if int(x) != VAC]
    site_to_id = {s:i+1 for i,s in enumerate(occupied_sites)}
    L = dims.astype(float) * (a/2.0)
    move_from = move_to = None
    if moved_atom is not None:
        move_from, move_to = moved_atom
    with open(path,'w') as f:
        f.write('Al-Cu FCC vacancy event\n\n')
        f.write(f'{len(occupied_sites)} atoms\n')
        f.write('2 atom types\n\n')
        f.write(f'0.0 {L[0]:.10f} xlo xhi\n0.0 {L[1]:.10f} ylo yhi\n0.0 {L[2]:.10f} zlo zhi\n\n')
        f.write('Masses\n\n1 26.9815385\n2 63.546\n\nAtoms # atomic\n\n')
        for s in occupied_sites:
            atom_id = site_to_id[s]
            typ = int(occ[s])
            g = sites[move_to] if (move_from is not None and s == move_from) else sites[s]
            xyz = g.astype(float)*(a/2.0)
            f.write(f'{atom_id} {typ} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n')
    return site_to_id

def generate_event(event_id, seed, nx=4, ny=4, nz=4, a=4.032, cu_fraction=0.10, force_jump_species=None):
    rng = np.random.default_rng(seed)
    sites,dims,site_map = make_fcc_sites(nx,ny,nz)
    nn1,nn2,shared = build_topology(sites,dims,site_map)
    n = len(sites)
    occ = np.full(n, AL, dtype=np.int8)
    ncu = max(1, int(round(cu_fraction*n)))
    cu_sites = rng.choice(n, ncu, replace=False)
    occ[cu_sites] = CU
    vac = int(rng.integers(n))
    slot = int(rng.integers(12))
    jump = int(nn1[vac,slot])
    if force_jump_species is not None:
        target = CU if str(force_jump_species).lower() == 'cu' else AL
        if int(occ[jump]) != target:
            candidates = np.where(occ == target)[0]
            candidates = candidates[candidates != vac]
            swap = int(rng.choice(candidates))
            occ[swap], occ[jump] = occ[jump], occ[swap]
    vac_original_species = int(occ[vac])
    occ[vac] = VAC
    desc = descriptor_for_event(occ,vac,jump,nn1,nn2,shared,slot)
    return {
        'event_id': int(event_id), 'seed': int(seed), 'a': float(a),
        'sites': sites, 'dims': dims, 'site_map': site_map,
        'nn1': nn1, 'nn2': nn2, 'shared': shared, 'occ': occ,
        'vac_site': vac, 'jump_site': jump, 'jump_slot': slot,
        'vac_original_species': 'Cu' if vac_original_species==CU else 'Al',
        'descriptor': desc,
    }
