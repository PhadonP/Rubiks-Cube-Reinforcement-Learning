#!/bin/bash
#SBATCH --job-name=Puzzle15Job
#SBATCH --account=ml20
#SBATCH --time=00:05:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Puzzle15.out
#SBATCH --mail-user=pphi15@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 train.py -c config/puzzle15.ini -n 10epochs