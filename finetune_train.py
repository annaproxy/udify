"""This file Finetunes on 7 languages in a non-episodic manner"""
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
import os
import torch
from torch.optim import Adam
from torch import autograd
import numpy as np 
from udify import util 
import json 
import subprocess
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=None, type=float, help="Adaptation LR")
parser.add_argument("--episodes", default=400, type=int, help="Episode amount")
parser.add_argument("--warmup_steps", default=40, type=int, help="Episode amount")
parser.add_argument("--small_test", default=0, type=int, help="only 1 language for debuggin")
parser.add_argument("--include_japanese", default=0, type=int, help="Include Japanese next to Bulgarian as validation language")
parser.add_argument("--more_lr", default = 0, type=int, help="Update BERT less fast in outer ")

args = parser.parse_args()


training_tasks = []
training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
if args.small_test == 0:
    training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
    training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
    training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
    training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
    training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))

validation_test_set = "data/expmix/" + "UD_Bulgarian-BTB" + "/" + "bg_btb-ud" + "-dev.conllu"

print("All Data Loaded")
train_params = get_params("finetuning")

SAVE_EVERY = 10
EPISODES = args.episodes
LR = args.lr
LR_SMALL = LR / 15.0
patience = 3
warmup_steps = args.warmup_steps
MORE_LR =  args.more_lr == 1 

MODEL_SAVE_NAME = "finetune_" + str(LR) + "_" + str(MORE_LR)
MODEL_VAL_DIR = MODEL_SAVE_NAME + "VAL"
MODEL_FILE = "logs/english_expmix_deps/2020.05.17_01.08.52/" #'./best.th'
VAL_WRITER = MODEL_VAL_DIR + '/val_las.txt'


if not os.path.exists(MODEL_VAL_DIR):
    subprocess.run(["mkdir", MODEL_VAL_DIR])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
    subprocess.run(["cp", "-r", MODEL_FILE +"/vocabulary", MODEL_VAL_DIR])
    subprocess.run(["cp", MODEL_FILE +"/config.json", MODEL_VAL_DIR])

model = Model.load(train_params, MODEL_FILE).cuda()
model.train()

if not MORE_LR:
    optimizer =  Adam(model.parameters(), LR)
else:
    optimizer =  Adam([{'params': model.text_field_embedder.parameters(), 'lr':LR_SMALL}, 
                    {'params':model.decoders.parameters(), 'lr':LR}, 
                    {'params':model.scalar_mix.parameters(), 'lr':LR}], LR)
                    
losses = []; task_num_tokens_seen = np.zeros(len(training_tasks))

best_validation_LAS = 0.0
best_iteration = -9
best_filename = None

for i, episode in enumerate(range(EPISODES)):
    for j, task in enumerate(training_tasks):
        input_set = next(task)[0]
        loss = model(**input_set)['loss']
        task_num_tokens_seen[j] += len(input_set['tokens']['tokens'][0])
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (i+1) % SAVE_EVERY == 0:
        print('Loss: %.3f' % np.mean(losses[-100:]))

        # Save current model
        SAVE_FILENAME = "best.th"
        torch.save(model.state_dict(), os.path.join(MODEL_VAL_DIR, SAVE_FILENAME))
        archive_model(MODEL_VAL_DIR, files_to_archive=train_params.files_to_archive, archive_path =MODEL_VAL_DIR)

        # Save predictions also
        current_pred_file = os.path.join(MODEL_VAL_DIR,'predictions', "temp_val.conllu")
        current_output_file = os.path.join(MODEL_VAL_DIR,'performance', "temp_performance" + str(i) + ".json")

        # Evaluate with horrible wrapper
        util.predict_and_evaluate_model(
            "udify_predictor",
            train_params,
            MODEL_VAL_DIR,
            validation_test_set,
            current_pred_file,
            current_output_file,
            batch_size=16
        )

        # The wrapper can only output json so we read it
        with open(current_output_file, 'r') as f:
            performance_dict = json.load(f)
            val_LAS = performance_dict['LAS']['aligned_accuracy']
            with open(VAL_WRITER,"a") as f:
                f.write(str(i) + '\t' + str(val_LAS))
                f.write('\n')

        # bookkeeping, save next best model, etc
        if val_LAS > best_validation_LAS:
            best_validation_LAS = val_LAS
            best_iteration = i
            patience = 5
            # Remove previous best
            if best_filename is not None:
                subprocess.run(["rm", best_filename ])

            # Save next best model 
            backup_path = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")
            torch.save(model.state_dict(), backup_path)
            best_filename = backup_path
        else:
            patience -= 1
    if patience == 0 and i > warmup_steps:
        print("Patience ran out, quitting", i)
            
print("Best iteration:", best_iteration, best_filename)
subprocess.run(["cp", best_filename, MODEL_VAL_DIR + "/best.th"])
archive_model(MODEL_VAL_DIR, files_to_archive=train_params.files_to_archive, archive_path =MODEL_VAL_DIR)

with open(MODEL_VAL_DIR + "/best_iter.txt", "w") as f:
    f.write(str(best_iteration))
    f.write('\n')
    f.write(best_filename)
print("Archived best iteration.")