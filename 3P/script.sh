#!/bin/bash

#SBATCH -J eu
#SBATCH -o eu_%j.out
#SBATCH -e eu_%j.err
#SBATCH -N 10
#SBATCH -n 14
#SBATCH -t 24:00:00
#SBATCH --mem=128G

module load cesga/2020 gcc openmpi/4.1.1_ft3

# for nN in {1..21}; do
#   for np in {2..21}; do
#     echo $np $nN
#     if [ $np -ge $nN ]; then
#       echo "Executing..."
#       for i in {1..5}; do
#         srun -N $nN -n $np $1 0 $2 $3 $4
#         echo "---------------------"
#       done
#       for i in {1..5}; do
#         srun -N $nN -n $np $1 1 $2 $3 $4
#         echo "---------------------"
#       done
#     echo "---------------------"
#     fi
#   done
# done

for mat_s in {100..50000..100}; do
    for i in {1..5}; do
        srun -N 4 -n 8 $1 0 $mat_s $mat_s $mat_s
        echo "-----------------"
    done
    for i in {1..5}; do
        srun -N 4 -n 8 $1 1 $mat_s $mat_s $mat_s
        echo "-----------------"
    done
done

echo "done"
