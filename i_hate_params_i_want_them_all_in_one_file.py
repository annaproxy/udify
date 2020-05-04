
import os 
import datetime 
import argparse
from allennlp.common import Params
from udify import util

def get_params():

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

    return util.merge_configs(configs)