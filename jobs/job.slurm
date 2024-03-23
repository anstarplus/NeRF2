#!/bin/bash
#SBATCH --job-name=train-nerf2-inue      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=48G                 # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=za1320@princeton.edu
module purge
module load anaconda3/2023.9
conda activate torch-env

cd ..
python -u nerf2_runner.py --config configs/dichasus-crosslink.yml --dataset_type dichasus-crosslink
# python -u nerf2_runner.py --config configs/dichasus-crosslink.yml --dataset_type dichasus-crosslink --mode test

# python -u nerf2_runner.py --config configs/dichasus-fdd.yml --dataset_type dichasus-fdd