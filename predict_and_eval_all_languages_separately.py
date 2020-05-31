from udify import util 
import logging 
from allennlp.common import Params
from i_hate_params_i_want_them_all_in_one_file import get_params 
import subprocess 
import os 
from udify.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default=None, type=str, help="Directory from which to start testing if not starting from pretrain")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = args.model_dir
SAVE_DIR = "zeroshot_" + MODEL_DIR 
subprocess.run(["mkdir", SAVE_DIR])

all_files = list()

with open("all_expmix_test.txt", "r") as f:
    for line in f:
        all_files.append(line.strip())

for test_file in all_files:
    current_gold_file = test_file 
    language_name = test_file.split('/')[2]

    predictions_file = SAVE_DIR + '/' + language_name + '_predicted.conllu'
    performance_file =  SAVE_DIR + '/' + language_name + '_performance.json'

    util.predict_and_evaluate_model(
        "udify_predictor",
        get_params("zeroshottesting"),
        MODEL_DIR, 
        current_gold_file,
        predictions_file,
        performance_file,
        batch_size=16
    )
    print("Wrote", performance_file)

