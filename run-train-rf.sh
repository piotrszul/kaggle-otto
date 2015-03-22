#!/bin/bash

#SBATCH --job-name="testjob"
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

module load R/3.0.2
module load openmpi/1.4.5

R --vanilla  < train_rf.R 


