
import argparse 
import os 
import datetime 
from allennlp.common.params import Params
from allennlp.common.util import lazy_groups_of
from allennlp.training.trainer_pieces import TrainerPieces
from udify import util

def get_language_dataset(language, language2, validate=False, cpu_for_some_reason=False):
    """
    A helper function that returns an Iterator[List[A]]
    Args:
        language: the uppercased variant, ie. UD_Russian-Taiga or Portugese-GSD
        language2: the lowercased variant, ie. ru_taiga-ud or pt_gsd-ud
        Why are these files named this way? I do not know, one of the mysteries of udify
    """
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
    #parser.add_argument("--lazy", default=None, action="store_true", help="Lazy load the dataset")
    #args = parser.parse_args()

    configs = []
    the_params = {
        'name':'clean_dataload',
        'base_config':'config/udify_base.json',
        #'device':-1,
        'predictor':'udify_predictor',
    }

    serialization_dir = os.path.join("logs", the_params['name'],
                    datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))

    overrides = {}
    #if args.device is not None:
    #    overrides["trainer"] = {"cuda_device": args.device}
    #if args.lazy is not None:
    if cpu_for_some_reason:
        overrides["trainer"] = {"cuda_device": -1}
    overrides["dataset_reader"] = {"lazy": True}

    trainpath = os.path.join("data/ud-treebanks-v2.3",  language , language2 + "-train.conllu")
    valpath = os.path.join("data/ud-treebanks-v2.3",  language , language2 + "-dev.conllu")
    testpath = os.path.join("data/ud-treebanks-v2.3",  language , language2 + "-test.conllu")

    configs.append(Params(overrides))
    configs.append(Params({
        "train_data_path": 
            trainpath if os.path.exists(valpath) else testpath,
        "validation_data_path": 
            valpath if os.path.exists(valpath) else (trainpath if os.path.exists(trainpath) else testpath),
        "test_data_path": 
            testpath,
        "vocabulary": {
            "directory_path": os.path.join("data/vocab/english_only_expmix4/vocabulary")
        }
    }))
    configs.append(Params.from_file("./config/ud/en/udify_bert_finetune_en_ewt.json"))
    configs.append(Params.from_file(the_params['base_config']))
   
    params = util.merge_configs(configs)

    if "vocabulary" in params:
        # Remove this key to make AllenNLP happy
        params["vocabulary"].pop("non_padded_namespaces", None)
    util.cache_vocab(params)
    # Special logic to instantiate backward-compatible trainer.
    pieces = TrainerPieces.from_params(params,  # pylint: disable=no-member
                                    serialization_dir,
                                    recover=False,
                                    cache_directory=None,
                                    cache_prefix=None)
    if validate:
        raw_train_generator = pieces.iterator(pieces.validation_dataset,
                                                num_epochs=1,
                                                shuffle=False)
    else:
        raw_train_generator = pieces.iterator(pieces.train_dataset,
                                                num_epochs=1,
                                                shuffle=True)

    return lazy_groups_of(raw_train_generator, 1)