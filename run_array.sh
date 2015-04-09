#!/bin/bash

#SBATCH --job-name="test_array"
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=131072

echo "Array jobg"
echo "JOB_ ID: $SLURM_ARRAY_JOB_ID"
echo "TASK_ID: $SLURM_ARRAY_TASK_ID"

