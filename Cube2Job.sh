#!/bin/bash
#SBATCH --job-name=Cube2Job
#SBATCH --account=ml20
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Cube2.out

module load python/3.7.3-system
source pipEnv3/bin/activate
python3 train.py -c config/cube2.ini -mp True