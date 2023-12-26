#!/bin/bash

#SBATCH -J ex1
#SBATCH -o ex1_%j.out
#SBATCH -e ex1_%j.err
#SBATCH -N 16
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH --mem=4G

module load cesga/2020 gcc openmpi/4.1.1_ft3

for nN in {1,2,4,8}; do
  for np in {1,2,4,8,16,32}; do
    echo $np $nN
    if [ $np -ge $nN ]; then
      echo "Executing..."
      for i in {1..5}; do
        srun -N $nN -n $np $1 $2 $3
        echo $1 $2 $3
      done
    fi
  done
done

echo "done"
