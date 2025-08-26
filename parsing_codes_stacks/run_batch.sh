#!/bin/bash
#SBATCH --job-name=union_merge
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load your python environment if needed
module load anaconda3
# Or activate a virtual environment:
# source ~/path/to/venv/bin/activate

echo "Job started on $(date)"
echo "Running on $(hostname)"
echo "Starting union merge script..."

python union_sw_stacks.py

echo "Job finished on $(date)"
