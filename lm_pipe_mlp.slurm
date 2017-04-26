#!/bin/bash
#
#SBATCH --job-name=act444_21
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --output=lstm_experiment_%A.out
#SBATCH --error=lstm_experiment_%A.err
#SBATCH --mail-user=act444@nyu.edu

# Log what we're running and where.
# echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python3/intel/3.5.3
module load cuda/8.0.44
module load nltk/3.2.2

python3 -m pip install -U pip setuptools --user
python3 -m pip install cmake --user
python3 -m pip install numpy --upgrade --user
python3 -m pip install -U scikit-learn --upgrade --user
python3 -m pip install https://download.pytorch.org/whl/cu80/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl --upgrade --user
python3 -m pip install torchvision --upgrade --user
python3 -m pip install tqdm --upgrade --user
python3 -m nltk.downloader 'punkt'
python3 -m nltk.downloader 'stopwords'
python3 -m nltk.downloader 'snowball_data'
python3 -m nltk.downloader 'wordnet'

cd /scratch/act444/github/quora-question-pairs/models

python3 -u lstm_main_mlp.py --din 20 --cuda --demb 300 --dhid 300 --dropout 0.5 --epochs 40 --embinit glove --decinit xavier_u --hidinit xavier_u --nlayers 1 --optimizer --clip 0.25 --vocabsize 50000 --loginterval 2000 --lr 0.0002 --pipeline --ppunc --pntok --save model_21 > ../expirimental_results/experiment.21.log 
