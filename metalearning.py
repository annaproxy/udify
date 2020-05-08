#from learn2learn import 
import os
import copy
import datetime
import logging
import argparse

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.common.util import import_submodules
from allennlp.commands.train import train_model
from i_hate_params_i_want_them_all_in_one_file import get_params 
from udify import util
import learn2learn as metalearn 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
train_params = get_params()


#m = Model.load(train_params, "./pretrained",)
#
# print(m)

meta_m = metalearn.algorithms.MAML(m, 1e-4, True)

raise ValueError("Succesfully loaded model and MAML")
predict_params = train_params.duplicate()
