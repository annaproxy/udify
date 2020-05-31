#!/bin/bash
#SBATCH --job-name=1e4ft
#SBATCH -t 12:40:00
#SBATCH -N 1
#SBATCH --partition=gpu_titanrtx_shared_course

module purge
module load 2019
module load eb
module load Python/3.6.6-foss-2019b
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
#python3.7 metatest_all.py --updates 1 --output_lr 1e-3 --start_from_pretrain 1 --more_lr 1 --which swedish
#python3.7 metatest_all.py --updates 1 --output_lr 1e-4 --start_from_pretain 1 --more_lr 1 --which all
#python3.7 metatest_all.py --updates 1 --output_lr 1e-4 --model_dir metalearn_0.001_0.0001_True_5VAL --more_lr 1
#python3.7 metalearn_train.py --meta_lr 1e-4 --inner_lr 1e-4 --updates 3 --more_lr 1
#python3.7 metalearn_train.py --meta_lr 1e-3 --inner_lr 1e-4 --updates 3 --more_lr 1
#python3.7 metalearn_train.py --meta_lr 1e-3 --inner_lr 1e-4 --updates 5 --more_lr 1
#python3.7 metalearn_train.py --meta_lr 5e-5 --inner_lr 1e-3 --updates 1 --more_lr 1
#python3.7 metalearn_train.py --meta_lr 5e-5 --inner_lr 1e-3 --updates 5 --more_lr 1
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir metalearn_0.001_0.0001_True_1VAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir metalearn_0.001_0.0001_True_5VAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir metalearn24_0.001_0.0001_True_3VAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir metalearn24_0.001_5e-05_True_3VAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir finetune_1e-05_TrueVAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir finetune24_5e-05_TrueVAL
python3.7 finetune_train.py --lr 5e-05 --more_lr 1
python3.7 metatest_all.py --updates 3 --output_lr 1e-4 --more_lr 1 --model_dir finetune24_seed2_5e-05_TrueVAL
python3.7 metatest_all.py --updates 3 --output_lr 1e-3 --more_lr 1 --model_dir finetune24_seed2_5e-05_TrueVAL
#python3.7 metatest_all.py --updates 3 --output_lr 1e-3 --more_lr 1 --model_dir finetune24_0.001_TrueVAL


#finetune24_5e-05_TrueVAL
echo "doneme up"
