#!/bin/bash
#SBATCH --job-name=Puzzle24Job
#SBATCH --account=ml20
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Puzzle24longRun.out

module load python/3.7.3-system
source pipEnv/bin/activate
python3 train.py -c config/puzzle24.ini -mp True