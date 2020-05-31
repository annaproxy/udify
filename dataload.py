from allennlp.common.params import Params
from allennlp.common.util import lazy_groups_of
from allennlp.training.trainer_pieces import TrainerPieces
from udify import util
from udify.models.udify_model import UdifyModel
import os 
import datetime 
import argparse
from allennlp.common import Params
from udify import util
import logging
from allennlp.nn import util as nn_util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="english_only", type=str, help="Log dir name")
parser.add_argument("--base_config", default="config/udify_base.json", type=str, help="Base configuration file")
parser.add_argument("--config", default="config/ud/en/udify_bert_finetune_en_ewt.json", type=str, nargs="+", help="Overriding configuration files")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
parser.add_argument("--resume", type=str, help="Resume training with the given model")
parser.add_argument("--lazy", default=None, action="store_true", help="Lazy load the dataset")
parser.add_argument("--cleanup_archive", action="store_true", help="Delete the model archive")
parser.add_argument("--replace_vocab", action="store_true", help="Create a new vocab and replace the cached one")
parser.add_argument("--archive_bert", action="store_true", help="Archives the finetuned BERT model after training")
parser.add_argument("--predictor", default="udify_predictor", type=str, help="The type of predictor to use")

args = parser.parse_args()
#print(type(args))

#raise ValueError()
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
    #for config_file in args.config:
    configs.append(Params.from_file(args.config))
    configs.append(Params.from_file(args.base_config))
else:
    serialization_dir = args.resume
    configs.append(Params.from_file(os.path.join(serialization_dir, "config.json")))

params = util.merge_configs(configs)

parameter_filename = "./config/ud/en/udify_bert_finetune_en_ewt.json"
#serialization_dir = "./"
# missing overrides arg
#params = Params.from_file(parameter_filename, "")
#params, serialization_dir = get_params()
if "vocabulary" in params:
    # Remove this key to make AllenNLP happy
    params["vocabulary"].pop("non_padded_namespaces", None)

# Special logic to instantiate backward-compatible trainer.
pieces = TrainerPieces.from_params(params,  # pylint: disable=no-member
                                serialization_dir,
                                recover=False,
                                cache_directory=None,
                                cache_prefix=None)

raw_train_generator = pieces.iterator(pieces.train_dataset,
                                            num_epochs=1,
                                            shuffle=False)

train_generator = lazy_groups_of(raw_train_generator, 1)

test = next(train_generator)[0]
print(test)
raise ValueError()
print(type(test))
from i_hate_params_i_want_them_all_in_one_file import get_params
train_params = get_params()
#m = UdifyModel.load(train_params, "./pretrained",)
#test = nn_util.move_to_device(test, self._cuda_devices[0])
print(type(test), test)
outputs = m.forward(**test[0])
print(outputs['loss'])
print(type(outputs))
