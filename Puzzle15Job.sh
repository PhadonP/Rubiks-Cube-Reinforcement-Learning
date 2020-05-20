#!/bin/bash
#SBATCH --job-name=Puzzle15Job
#SBATCH --account=ml20
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Puzzle15.out

module load python/3.7.3-system
source pipEnv/bin/activate
python3 train.py -c config/puzzle15.ini -n realRun -mp True