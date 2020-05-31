EPOCH_NO=8
print(EPOCH_NO)
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from learn2learn.algorithms import MAML
import subprocess 
import os 
import torch
from udify import util 
from torch.optim import Adam
from torch import autograd
import argparse



bulgarian_iterator  = get_language_dataset('UD_Bulgarian-BTB','bg_btb-ud')
train_params = get_params()
m = Model.load(train_params, "episode" + str(EPOCH_NO)).cuda()
optimizer =  Adam(m.parameters(), 1e-4)

support_set = next(bulgarian_iterator)[0]
loss = m.forward(**support_set)['loss']

optimizer.zero_grad()
loss.backward()
optimizer.step()
DIR_NAME = "validate_" + str(EPOCH_NO)
subprocess.run(["mkdir", DIR_NAME])
model_save_place= DIR_NAME + "/weights.th"
torch.save(m.state_dict(), model_save_place)
print("SAVED", model_save_place)
"""
DIR_NAME = "validate_" + str(EPOCH_NO)
subprocess.run(["echo", "STARTING TAR"])
subprocess.run(["mkdir", "allval" + str(EPOCH_NO)])
subprocess.run(["tar", "-c", "-z", "-v", "-f", "allval" + str(EPOCH_NO) + "/model.tar.gz", model_save_place, 
            "logs/english_only_expmix4/2020.05.13_01.43.52/vocabulary/", "logs/english_only_expmix4/2020.05.13_01.43.52/config.json" ])
"""

