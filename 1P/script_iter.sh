#!/bin/bash

#SBATCH -J ex1
#SBATCH -o ex1_%j.out
#SBATCH -e ex1_%j.err
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -t 02:30:00
#SBATCH --mem=4G

module load cesga/2020 gcc openmpi/4.1.1_ft3

for ((i = 1; i <= 1000000000; i *= 10)); do
    for j in {1..10}; do
        srun -N 4 -n 16 $1 $i
    done
done


echo "done"
