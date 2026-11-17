#!/usr/bin/env bash

#$ -P me500jc
#$ -N alcu_python_kmc
#$ -pe omp 1
#$ -l h_rt=01:00:00
#$ -j y
#$ -o kmc_job.out

set -euo pipefail
python run_kmc.py
python analyze_kmc.py
