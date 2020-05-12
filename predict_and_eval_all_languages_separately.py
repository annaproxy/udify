from udify import util 
import logging 
from allennlp.common import Params
from i_hate_params_i_want_them_all_in_one_file import get_params 
import subprocess 
import os 
from udify.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


all_files = list()
with open("all_expmix_test.txt", "r") as f:
    for line in f:
        all_files.append(line.strip())

for test_file in all_files:
    current_gold_file = test_file 
    language_name = test_file.split('/')[2] + '_predicted.conllu'
    performance_name = test_file.split('/')[2] + '_results.json'
    
    current_pred_file = os.path.join('predictions',language_name)
    current_output_file = os.path.join('performance', performance_name)


    evaluation = evaluate(load_conllu_file(current_gold_file), load_conllu_file(current_pred_file))
    util.save_metrics(evaluation, current_output_file)
    raise ValueError("Check your output")
    
    current_pred_file = os.path.join('predictions',language_name)
    current_output_file = os.path.join('performance', performance_name)

    util.predict_and_evaluate_model(
        "udify_predictor",
        get_params(),
        "pretrained_expmix",
        current_gold_file,
        current_pred_file,
        current_output_file,
        batch_size=16
    )
    print("Wrote", current_output_file)
    
   # raise ValueError("Something is working at least")