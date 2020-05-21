# -*- coding: utf-8 -*-
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
from allennlp.models.archival import archive_model
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from udify.predictors import predictor
from learn2learn.algorithms import MAML
import numpy as np
import torch
from torch.optim import Adam
from torch import autograd
import subprocess
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

# Get some samples from train set
validation_iterator = get_language_dataset("UD_Bulgarian-BTB", "bg_btb-ud")
validation_test_set = "data/expmix/" + "UD_Bulgarian-BTB" + "/" + "bg_btb-ud" + "-dev.conllu"

print("All Data Loaded")

SAVE_EVERY=10
EPISODES = 200
INNER_LR = 1e-4
META_LR = 5e-5
patience = 3
warmup_steps = 50

MODEL_FILE = "logs/english_expmix_deps/2020.05.17_01.08.52/"
MODEL_SAVE_NAME = "metalearn_5e5"
MODEL_VAL_DIR = MODEL_SAVE_NAME + "VAL"

if not os.path.exists(MODEL_VAL_DIR):
    subprocess.run(["mkdir", MODEL_VAL_DIR])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
    subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
    subprocess.run(["cp", "-r", MODEL_FILE +"/vocabulary", MODEL_VAL_DIR])
    subprocess.run(["cp", MODEL_FILE +"/config.json", MODEL_VAL_DIR])

train_params = get_params()
m = Model.load(train_params, MODEL_FILE,).cuda()
meta_m = MAML(m, INNER_LR, first_order=True, allow_unused=True).cuda()
optimizer =  Adam(meta_m.parameters(), META_LR)

with open("some_results.txt", "w") as f:
    f.write("Model ready\n")

with open("val_las.txt", "w") as f:
    f.write("Model ready\n")

best_validation_LAS = 0.0
best_iteration = -9
best_filename = None


task_num_tokens_seen = np.zeros(len(training_tasks))

for iteration in range(EPISODES):
    iteration_loss = 0.0

    """Inner adaptation loop"""
    for j, task_generator in enumerate(training_tasks):
        learner = meta_m.clone()
        support_set = next(task_generator)[0]
        query_set = next(task_generator)[0]
        inner_loss = learner.forward(**support_set)['loss']
        print("\t", inner_loss.item())
        learner.adapt(inner_loss, first_order=True)
        eval_loss = learner.forward(**query_set)['loss']

        # Bookkeeping
        task_num_tokens_seen[j] += len(support_set['tokens']['tokens'][0])
        task_num_tokens_seen[j] += len(query_set['tokens']['tokens'][0])
        iteration_loss += eval_loss
        del eval_loss 
        del inner_loss 
        del support_set 
        del query_set
        del learner

    # Sum up and normalize over all 7 losses
    iteration_loss /= len(training_tasks)
    optimizer.zero_grad()
    iteration_loss.backward()
    optimizer.step()

    # Bookkeeping
    print(iteration, "meta", iteration_loss.item())
    with open("some_results.txt", "a") as f:
        f.write(str(iteration) + " meta " + str(iteration_loss.item()))
        f.write("\n")
    del iteration_loss 

    # Meta validation
    if (iteration+1) % SAVE_EVERY == 0:
        # Do update
        vallearner = meta_m.clone()
        batch = next(validation_iterator)
        print("Batch", batch)
        metaval_loss = vallearner.forward(**batch[0])['loss']
        vallearner.adapt(metaval_loss, first_order=True)

        # Save model, this is necessary for prediction
        SAVE_FILENAME = "best.th"
        torch.save(vallearner.module.state_dict(), os.path.join(MODEL_VAL_DIR, SAVE_FILENAME))
        archive_model(MODEL_VAL_DIR, files_to_archive=train_params.files_to_archive, archive_path =MODEL_VAL_DIR)

        # Predict entire validation set
        current_pred_file = os.path.join(MODEL_VAL_DIR,'predictions', "temp_val.conllu")
        current_output_file = os.path.join(MODEL_VAL_DIR,'performance', "temp_performance" + str(iteration) + ".json")
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
            with open("val_las.txt","a") as f:
                f.write(str(val_LAS))
                f.write('\n')

        # Check if we are doing better!
        if val_LAS > best_validation_LAS:
            best_validation_LAS = val_LAS
            best_iteration = iteration
            patience = 3
            # Remove previous best
            subprocess.run(["rm", best_filename ])

            # Save META model for later inference
            backup_path = os.path.join(MODEL_VAL_DIR, "model" + str(iteration) + ".th")
            torch.save(meta_m.module.state_dict(), backup_path)
            best_filename = backup_path
        else:
            patience -= 1

        del vallearner
        del metaval_loss
        del batch
        del val_LAS
    if patience == 0 and iteration > warmup_steps:
        print("Patience ran out, quitting", iteration)



print("Best iteration:", best_iteration, best_filename)
subprocess.run(["cp", best_filename, "best.th"])
archive_model(MODEL_VAL_DIR, files_to_archive=train_params.files_to_archive, archive_path =MODEL_VAL_DIR)
print("Archived best iteration.")


normalized_tokens_seen = task_num_tokens_seen / np.max(task_num_tokens_seen)
print("Number of Tokens seen per task: {}, relative to maximum: {}".format(task_num_tokens_seen, normalized_tokens_seen))
np.save('task_num_tokens_seen.npy', task_num_tokens_seen)



