"""This file Finetunes on 7 languages in a non-episodic manner"""
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model

import torch
from torch.optim import Adam
from torch import autograd
import numpy as np 
#raise ValueError("Done")
# Tran&Bisazza: Italian, Norwegian, 
# Czech, Russian, Hindi, Korean, Arabic

training_tasks = []
training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))

print("All Data Loaded")

train_params = get_params()

BATCH_SIZE=16; SAVE_EVERY=40
Path_model = './best.th'

model = Model.load(train_params, Path_model).cuda()
model.train()

optimizer =  Adam(model.parameters(), 1e-4)
losses = []; task_num_tokens_seen = np.zeros(len(training_tasks))

for i, episode in enumerate(range(160)):
    for j, task in enumerate(training_tasks):
        input_set = next(task)[0]
        loss = model(**input_set)['loss']
        task_num_tokens_seen[j] += len(input_set['tokens']['tokens'][0])
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (i+1) % 10 == 0:
        print('Loss: %.3f' % np.mean(losses[-100:]))
            
    if (i+1) % SAVE_EVERY == 0:
        torch.save(model.state_dict(),"model_checkpoint_{}.th".format(i+1))
        print("Model #{} has been saved.".format(i+1))
        


print("Success, Epoch's Last iteration loss: {}".format(losses[-1]))
normalized_tokens_seen = task_num_tokens_seen / np.max(task_num_tokens_seen)
print("Number of Tokens seen per task: {}, relative to maximum: {}".format(task_num_tokens_seen, normalized_tokens_seen))
np.save('task_num_tokens_seen.npy', task_num_tokens_seen)