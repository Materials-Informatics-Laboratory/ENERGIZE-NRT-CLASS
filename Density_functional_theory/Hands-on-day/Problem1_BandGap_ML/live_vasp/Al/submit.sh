#!/bin/bash -l
#$ -P me500jc
#$ -N bandgap_demo
#$ -j y
#$ -o vasp_job.out
#$ -l h_rt=00:20:00
#$ -pe omp 4

module purge
module use /projectnb/me500jc/materials/software/modules
module load intel/2021.1
module load vasp/6.3.2

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G

mpirun -np "${NSLOTS:-4}" vasp_std > out
