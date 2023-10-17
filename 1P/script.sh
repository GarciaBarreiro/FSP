#!/bin/bash

#SBATCH -J ex1
#SBATCH -o ex1_%j.out
#SBATCH -e ex1_%j.err
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 00:02:00
#SBATCH --mem=2G

module load cesga/2020 gcc openmpi/4.1.1_ft3

srun $1
echo "done"
