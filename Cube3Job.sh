#!/bin/bash
#SBATCH --job-name=Cube3Job
#SBATCH --account=ml20
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Cube3.out

module load python/3.7.3-system
source pipEnv3/bin/activate
python3 train.py -c config/cube3.ini -mp True