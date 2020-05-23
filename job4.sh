#!/bin/bash
#SBATCH --job-name=metalearn
#SBATCH -t 13:40:00
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
#python3.7 finetune_train.py --lr 5e-5
python3.7 finetune_train.py --lr 5e-5 --more_lr 1
#python3.7 finetune_train.py --lr 1e-5  
python3.7 finetune_train.py --lr 1e-5 --more_lr 1
python3.7 finetune_train.py --lr 1e-4 --more_lr 1

for f in *; do
    #echo $f
    if [ -d ${f} ]; then
	if [[ $f == finetun* ]]; then  # 
        	# Will not run if no directories are available
        	python3.7 metatest_all.py --updates 1 --output_lr 1e-4 --model_dir $f --more_lr 1
        	python3.7 metatest_all.py --updates 1 --output_lr 1e-3 --model_fir $f --more_lr 1
        	#python3.7 metatest_all.py --updates 1 --output_lr 5e-5 --model_fir $f --more_lr 1
            
	fi
    fi
done

echo "doneme up"
