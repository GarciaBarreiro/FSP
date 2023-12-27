#!/bin/bash

#SBATCH -J ex1
#SBATCH -o ex1_%j.out
#SBATCH -e ex1_%j.err
#SBATCH -N 8
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH --mem=10G

module load cesga/2020 gcc openmpi/4.1.1_ft3

for nN in {2,4,8}; do
  for np in {2,4,16,32}; do
    echo $np $nN
    if [ $np -ge $nN ]; then
      echo "Executing..."
      #for N in {50,100,1000,5000,10000,50000}; do
      for N in {4,5,8}; do
        #for F in {50,100,1000,5000,10000,50000}; do
        for F in {4,5,8}; do
          echo $N $F
          for i in {1..3}; do
            srun -N $nN -n $np $1 $N $N $F $F
          done
        done
      done
    fi
  done
done

echo "done"
