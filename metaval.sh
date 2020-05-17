#!/bin/bash
#SBATCH --job-name=meta_validate
#SBATCH -t 1:40:00
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
#pip3 install --user --no-cache-dir torch torchvision
#pip3 install --user --no-cache-dir -r ./requirements.txt
#bash ./scripts/download_ud_data.sh

# Concat treebanks and build vocabulary
#python3 concat_treebanks.py --output_dir "expmixvocab2"
#python3.7 train.py --name "english_only_expmix4" 

# Do some DESPERATE gpu testing, you need torch==1.4.0
#nvidia-smi
#python3 gputest.py >> gputestout
##pip3 install --user torch==1.4.0 
# Train english only with right params
#python3 train_english_only.py

#python3 predict_and_eval_all_languages_separately.py

for i in 2 3 4 5 6 7 8; do
    #dirname="episode$i"
    #tar -xzvf "$dirname/model.tar.gz" "$dirname"
    var="EPOCH_NO=$i"
    sed -i "1s/.*/$var/" metavalidate.py
    python3 metavalidate.py 
done

echo "doneme up"
