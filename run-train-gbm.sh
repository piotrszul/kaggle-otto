#!/bin/bash

#SBATCH --job-name="testjob"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4096

module load R/3.0.2
R --vanilla  < train_gbm.R 


