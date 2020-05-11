#!/bin/bash
#SBATCH --job-name=train_en
#SBATCH -t 8:40:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared_course

module purge
module load 2019
module load eb
module load Python/3.7.5-foss-2018b
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243

echo "starting now"

# Evaluating languages
#python3 predict_and_eval_all_languages_separately.py

# Installing requirements
#pip3 install --user --no-cache-dir -r ./requirements.txt
#bash ./scripts/download_ud_data.sh

# Concat treebanks and build vocabulary
#python3 concat_treebanks.py --output_dir "expmixvocab"
# python3 train.py --name "english_only_expmix" 
python3 gputest.py >> gputestout
# Train english only with right params
python3 train_english_only.py

echo "doneme up"
