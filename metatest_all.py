from allennlp.models.archival import archive_model
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from torch.optim import Adam
import torch
import subprocess
from udify import util
import os 
from naming_conventions import languages, languages_lowercase
from get_language_dataset import get_language_dataset

# The model on which to Meta_test
MODEL_DIR_PRETRAIN = "logs/english_expmix_deps/2020.05.17_01.08.52/"
MODEL_DIR_FINETUNE = "metalearn_5e5VAL"
MODEL_DIR = MODEL_DIR_FINETUNE

WHERE_TO_SAVE = "metatesting" + MODEL_DIR

subprocess.run(["mkdir", WHERE_TO_SAVE])

# The language on which to evaluate 
for i, language in enumerate(languages):
    test_file = "data/expmix/" + language + "/" + languages_lowercase[1] + "-test.conllu"
    val_iterator = get_language_dataset(language, languages_lowercase[1], validate=True)

    # Create directory and copy relevant files there for later
    SERIALIZATION_DIR = WHERE_TO_SAVE + '/resultsvalidation' + language
    
    subprocess.run(["mkdir", SERIALIZATION_DIR])
    subprocess.run(["cp", "-r", MODEL_DIR +"/vocabulary", SERIALIZATION_DIR])
    subprocess.run(["cp", MODEL_DIR +"/config.json", SERIALIZATION_DIR])

    # Set up model and iterator and optimizer
    train_params = get_params()
    m = Model.load(train_params, MODEL_DIR).cuda()
    optimizer =  Adam(m.parameters(), 1e-4)

    # Do one forward pass
    support_set = next(val_iterator)[0]
    loss = m.forward(**support_set)['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save model, then zip model
    model_save_place= SERIALIZATION_DIR + "/best.th"
    torch.save(m.state_dict(), model_save_place)
    archive_model(SERIALIZATION_DIR, files_to_archive=train_params.files_to_archive, archive_path = SERIALIZATION_DIR)

    # Get specific predictions for this finetuned guy
    current_pred_file = os.path.join('predictions',language +'.conllu')
    current_output_file = os.path.join('performance',language+'.json')

    util.predict_and_evaluate_model(
        "udify_predictor",
        get_params(),
        SERIALIZATION_DIR,
        test_file,
        current_pred_file,
        current_output_file,
        batch_size=16
    )
    print("Wrote", current_output_file)