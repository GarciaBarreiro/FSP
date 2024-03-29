#!/bin/bash

#SBATCH -J pae_1
#SBATCH -o pae_1_%j.out
#SBATCH -e pae_1_%j.err
#SBATCH -N 32
#SBATCH -n 256
#SBATCH -t 01:00:00
#SBATCH --mem=4G

module load cesga/2020 gcc openmpi/4.1.1_ft3

for nN in {1,2,4,8,16,32}; do
  for np in {1,2,4,8,16,32}; do
    echo $np $nN
    if [ $np -ge $nN ]; then
      echo "Executing..."
      for i in {1..5}; do
        srun -N $nN -n $np $1
      done
    fi
  done
done

echo "done"
