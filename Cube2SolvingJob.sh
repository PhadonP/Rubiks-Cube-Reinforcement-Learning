#!/bin/bash
#SBATCH --job-name=Cube2SolvingJob
#SBATCH --account=ml20
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --output=Cube2Solves2.out
#SBATCH --mail-user=pphi15@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load pytorch/1.3-cuda10
python3 solve.py -n saves/Cuben-2,ScrambleDepth=20,10000epochs.pt -c config/cube2.ini 