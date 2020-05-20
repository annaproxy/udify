"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from udify.predictors import predictor
from learn2learn.algorithms import MAML

import torch
from torch.optim import Adam
from torch import autograd

training_tasks = []
#training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
#training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
#training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
#training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
#training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
#training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))
validation_iterator = get_language_dataset("UD_Bulgarian-BTB", "bg_btb-ud", True)

print("All Data Loaded")

INNER_LR = 1e-4
META_LR = 5e-5
MODEL_FILE = "logs/english_expmix_deps/2020.05.17_01.08.52/"
MODEL_SAVE_NAME = "metalearn"

train_params = get_params()
m = Model.load(train_params, MODEL_FILE,).cuda()
meta_m = MAML(m, INNER_LR, first_order=True, allow_unused=True).cuda()
optimizer =  Adam(meta_m.parameters(), META_LR)

with open("some_results.txt", "w") as f:
    f.write("Model ready\n")

best_validation_las = 0.0
for iteration in range(100):
    iteration_loss = 0.0
    for task_generator in training_tasks:
        learner = meta_m.clone()
        support_set = next(task_generator)[0]
        query_set = next(task_generator)[0]
        inner_loss = learner.forward(**support_set)['loss']
        print("\tone forward loss: ", inner_loss.item())
        learner.adapt(inner_loss, first_order=True)
        eval_loss = learner.forward(**query_set)['loss']
        iteration_loss += eval_loss
        del eval_loss 
        del inner_loss 
        del support_set 
        del query_set
        del learner
    iteration_loss /= len(training_tasks)
    optimizer.zero_grad()
    iteration_loss.backward()
    optimizer.step()
    print("meta", iteration_loss.item())
    with open("some_results.txt", "a") as f:
        f.write("meta " + str(iteration_loss.item()))
        f.write("\n")
    del iteration_loss 
    with torch.no_grad():
        if iteration+1 % 10 == 0:
            udify_predictor = predictor.UdifyPredictor(meta_m.module, validation_iterator)
            for batch in validation_iterator:
                preds = udify_predictor.predict_batch_instance(batch)
                print(preds, type(preds))
                raise ValueError("Hi")
            torch.save(meta_m.module.state_dict(), MODEL_SAVE_NAME + "_" + iteration + ".th")




