#!/bin/bash -l

#$ -P me500jc
#$ -N alcu_neb100
#$ -pe omp 8
#$ -l h_rt=02:00:00
#$ -j y
#$ -o neb_job.out

set -euo pipefail
module purge
module load gcc/11.2.0
module load openmpi/3.1.4_gnu-11.2.0
export OMP_NUM_THREADS=1
LMP=/projectnb/me500jc/materials/software/lammps/src/lmp_mpi

if [[ ! -f ../potentials/al-cu-set.eam.alloy ]]; then
  echo "Missing ../potentials/al-cu-set.eam.alloy" >&2
  exit 2
fi

mpirun -np 8 "$LMP" -partition 8x1 -in in.neb > screen.out 2>&1
