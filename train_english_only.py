"""
Training script useful for debugging UDify and AllenNLP code
"""

import os
import copy
import datetime
import logging
import argparse
import sys
#print(sys.getdefaultencoding())
#raise ValueError("j")
from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.common.util import import_submodules
from allennlp.commands.train import train_model

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="english_expmix_deps", type=str, help="Log dir name")
parser.add_argument("--base_config", default="config/udify_base.json", type=str, help="Base configuration file")
parser.add_argument("--config", default="config/ud/en/udify_bert_finetune_en_ewt.json", type=str, nargs="+", help="Overriding configuration files")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
parser.add_argument("--resume", type=str, help="Resume training with the given model")
parser.add_argument("--lazy", default=None, action="store_true", help="Lazy load the dataset")
parser.add_argument("--cleanup_archive", action="store_true", help="Delete the model archive")
#parser.add_argument("--replace_vocab", action="store_true", help="Create a new vocab and replace the cached one")
parser.add_argument("--archive_bert", action="store_true", help="Archives the finetuned BERT model after training")
parser.add_argument("--predictor", default="udify_predictor", type=str, help="The type of predictor to use")
args = parser.parse_args()

log_dir_name = args.name
if not log_dir_name:
    file_name = args.config[0] if args.config else args.base_config
    log_dir_name = os.path.basename(file_name).split(".")[0]

configs = []

if not args.resume:
    serialization_dir = os.path.join("logs", log_dir_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))

    overrides = {}
    if args.device is not None:
        overrides["trainer"] = {"cuda_device": args.device}
    if args.lazy is not None:
        overrides["dataset_reader"] = {"lazy": args.lazy}
    configs.append(Params(overrides))
    #configs.append(Params({'config':}))
    configs.append(Params.from_file(args.config))
    configs.append(Params.from_file(args.base_config))
else:
    serialization_dir = args.resume
    configs.append(Params.from_file(os.path.join(serialization_dir, "config.json")))

train_params = util.merge_configs(configs)
predict_params = train_params.duplicate()

if "vocabulary" in train_params:
    # Remove this key to make AllenNLP happy
    train_params["vocabulary"].pop("non_padded_namespaces", None)

import_submodules("udify")


try: 
    # Vocabulary should be there already!
    #util.cache_vocab(train_params)
    train_model(train_params, serialization_dir, recover=bool(args.resume))
except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt, skipping training")
    
"""
m = Model.load(train_params, "./pretrained",)
print(m)


raise ValueError("Succesfully loaded model")


import_submodules("udify")


try:
    util.cache_vocab(train_params)
    train_model(train_params, serialization_dir, recover=bool(args.resume))
except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt, skipping training")
"""
#print("Setting test files")
#dev_file = predict_params["validation_data_path"]
#test_file = predict_params["test_data_path"]
#test_file = "data/ud/multilingual/test.conllu"
#test_pred = "testpred.conllu"
#test_eval = "testEVAL.json"
#dev_pred, dev_eval, test_pred, test_eval = [
#    os.path.join(serialization_dir, name)n_ewt-ud-t
#    for name in ["dev.conllu", "dev_results.json", "test.conllu", "test_results.json"]
#]
#serialization_dir = "./pretrained"

#if dev_file != test_file:
#    util.predict_and_evaluate_model(args.predictor, predict_params, serialization_dir, dev_file, dev_pred, dev_eval)

#util.predict_and_evaluate_model(args.predictor, predict_params, serialization_dir, test_file, test_pred, test_eval)
#print("Predictions done")


if args.archive_bert:
    bert_config = "config/archive/bert-base-multilingual-cased/bert_config.json"
    util.archive_bert_model(serialization_dir, bert_config)

util.cleanup_training(serialization_dir, keep_archive=not args.cleanup_archive)

