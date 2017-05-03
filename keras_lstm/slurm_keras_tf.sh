#!/bin/bash
#
#SBATCH --job-name=ak_experiment
#SBATCH --partition=gpu
#SBATCH --reservation=mhealth
#SBATCH --gres=gpu:p1080:1
#SBATCH --time=10:00:00
#SBATCH --mem=30000
#SBATCH --output=ak_experiment_%A.out
#SBATCH --error=ak_experiment_%A.err
#SBATCH --mail-user=ak6179@nyu.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/intel/2.7.12
module load cuda/8.0.44

python2.7 -m pip install tensorflow --upgrade --user
python2.7 -m pip install keras --upgrade --user
python2.7 -m pip install pandas --upgrade --user
python2.7 -m pip install nltk --upgrade --user
python2.7 -m pip install gensim --upgrade --user

cd /scratch/ak6179/lstm/keras_lstm

python2.7 -u lstm.py > experiments/experiment.1.log

