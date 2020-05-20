from allennlp.models.archival import archive_model
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from torch.optim import Adam
import torch
import subprocess
from udify import util
import os 

# The model on which to Meta_test
MODEL_DIR_PRETRAIN = "logs/english_expmix_deps/2020.05.17_01.08.52/"
MODEL_DIR_FINETUNE = "episode6/"
MODEL_DIR = MODEL_DIR_FINETUNE

all_files = list()

with open("all_expmix_test.txt", "r") as f:
    for line in f:
        all_files.append(line.strip())

# The language on which to evaluate 
for filename in all_files:
    LAN_TEST = filename
    LAN_DEV = filename.replace('test', 'dev')
    # example expmix/UD_Finnish-TDT/fi_tdt-ud-


    # Create directory and copy relevant files there for later
    SERIALIZATION_DIR = 'resultsvalidation' + LAN1
    subprocess.run(["mkdir", SERIALIZATION_DIR])
    subprocess.run(["cp", "-r", MODEL_DIR +"/vocabulary", SERIALIZATION_DIR])
    subprocess.run(["cp", MODEL_DIR +"/config.json", SERIALIZATION_DIR])

    # Set up model and iterator and optimizer
    lan_iterator = get_language_dataset(LAN1,LAN2)
    train_params = get_params()
    m = Model.load(train_params, MODEL_DIR).cuda()
    optimizer =  Adam(m.parameters(), 1e-4)

    # Do one forward pass
    support_set = next(lan_iterator)[0]
    loss = m.forward(**support_set)['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_save_place= SERIALIZATION_DIR + "/best.th"
    torch.save(m.state_dict(), model_save_place)
    print("SAVED", model_save_place)

    # Now tar up results with allennlp
    archive_model(SERIALIZATION_DIR, files_to_archive=train_params.files_to_archive, archive_path = SERIALIZATION_DIR)


    ###### GET TO TESTING #######
    current_pred_file = os.path.join('predictions',LAN1)
    current_output_file = os.path.join('performance', LAN1)

    util.predict_and_evaluate_model(
        "udify_predictor",
        get_params(),
        SERIALIZATION_DIR,
        LAN_TEST,
        current_pred_file,
        current_output_file,
        batch_size=16
    )
    print("Wrote", current_output_file)