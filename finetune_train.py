"""This file Finetunes on 7 languages in a non-episodic manner"""
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
import os
import torch
from torch.optim import Adam
from torch import autograd
import numpy as np 
from udify import util 
import json 
import subprocess

training_tasks = []
training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))

validation_test_set = "data/expmix/" + "UD_Bulgarian-BTB" + "/" + "bg_btb-ud" + "-dev.conllu"

print("All Data Loaded")
train_params = get_params()

SAVE_EVERY = 10
EPISODES = 400
LR = 5e-5
patience = 3
warmup_steps = 50

MODEL_SAVE_NAME = "finetune_5e5"
MODEL_VAL_DIR = MODEL_SAVE_NAME + "VAL"
MODEL_FILE = "logs/english_expmix_deps/2020.05.17_01.08.52/" #'./best.th'

if not os.path.exists(MODEL_VAL_DIR):
    subprocess.run(["mkdir", MODEL_VAL_DIR])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
    subprocess.run(["cp", "-r", MODEL_FILE +"/vocabulary", MODEL_VAL_DIR])
    subprocess.run(["cp", MODEL_FILE +"/config.json", MODEL_VAL_DIR])

model = Model.load(train_params, MODEL_FILE).cuda()
model.train()

optimizer =  Adam(model.parameters(), LR)
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

        SAVE_FILENAME = "best.th"
        torch.save(model.state_dict(), os.path.join(MODEL_VAL_DIR, SAVE_FILENAME))
        current_pred_file = os.path.join(MODEL_VAL_DIR,'predictions', "temp_val.conllu")
        current_output_file = os.path.join(MODEL_VAL_DIR,'performance', "temp_performance" + str(i) + ".json")

        util.predict_and_evaluate_model(
            "udify_predictor",
            train_params,
            MODEL_VAL_DIR,
            validation_test_set,
            current_pred_file,
            current_output_file,
            batch_size=16
        )
        # Easiest way is just to fucking die and dump it to json and then read it.
        with open(current_output_file, 'r') as f:
            performance_dict = json.load(f)
            val_LAS = performance_dict['LAS']['aligned_accuracy']
            print(val_LAS)

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
            
print("Best iteration:", best_iteration, best_filename)
print("Success, Epoch's Last iteration loss: {}".format(losses[-1]))
normalized_tokens_seen = task_num_tokens_seen / np.max(task_num_tokens_seen)
print("Number of Tokens seen per task: {}, relative to maximum: {}".format(task_num_tokens_seen, normalized_tokens_seen))
np.save('task_num_tokens_seen.npy', task_num_tokens_seen)
