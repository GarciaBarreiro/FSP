#!/bin/bash

#SBATCH -J ex1
#SBATCH -o ex1_%j.out
#SBATCH -e ex1_%j.err
#SBATCH -N 8
#SBATCH -n 64
#SBATCH -t 00:20:00
#SBATCH --mem=200G

module load cesga/2020 gcc openmpi/4.1.1_ft3

for nN in {2,4,8}; do
  for np in {2,4,8,16,32}; do
    echo $np $nN
    if [ $np -ge $nN ]; then
      echo "Executing..."
      #for M in {50,100,1000,5000,10000,50000}; do
      for M in {10000,50000}; do
        #for N in {50,100,1000,5000,10000,50000}; do
        for N in {10000,50000}; do
          echo $M $N
          for i in {1..3}; do
            srun -N $nN -n $np $1 $M $N $N $M
          done
        done
      done
    fi
  done
done

echo "done"
