from allennlp.models.archival import archive_model
from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
from torch.optim import Adam
import torch
import subprocess
from udify import util
import os 
from naming_conventions import languages, languages_lowercase, final_languages, final_languages_lowercase, no_train_set, no_train_set_lowercase
from get_language_dataset import get_language_dataset, get_test_set
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--start_from_pretrain", default=0, type=int, help="Whether to start from pretrain")
parser.add_argument("--model_dir", default=None, type=str, help="Directory from which to start testing if not starting from pretrain")
parser.add_argument("--output_lr", default=None, type=float, help="Fast adaptation output learning rate")
parser.add_argument("--updates", default=1, type=int, help="Amount of inner loop updates")
parser.add_argument("--more_lr", default = 0, type=int, help="Update BERT less fast in outer ")
parser.add_argument("--which", default="all", type=str, help="Whether to evaluate only swedish and after. [all|swedish|nontrain]")
args = parser.parse_args()

# The model on which to Meta_test
MODEL_DIR_PRETRAIN = "logs/english_expmix_deps/2020.05.17_01.08.52/"
MODEL_DIR_FINETUNE = args.model_dir
MODEL_DIR = MODEL_DIR_FINETUNE if args.start_from_pretrain == 0 else MODEL_DIR_PRETRAIN
MODEL_NAMEDIR = (MODEL_DIR if args.start_from_pretrain==0 else 'ONLY')
LR = args.output_lr
LR_SMALL = LR / 15.0

UPDATES = args.updates
MORE_LR =  args.more_lr == 1 
WHERE_TO_SAVE = "metatesting_" + str(LR) + "_" + str(MORE_LR) + str(UPDATES) + '_' + MODEL_NAMEDIR + '_averaging'
if args.which == "all":
    the_languages = languages 
    the_languages_lowercase = languages_lowercase
elif args.which == "notrain":
    the_languages = no_train_set
    the_languages_lowercase = no_train_set_lowercase
else:
    the_languages = final_languages
    the_languages_lowercase = final_languages_lowercase

print("Saving all to directory", WHERE_TO_SAVE)
print("Running from", MODEL_DIR, "with learning rate", LR)
subprocess.run(["mkdir", WHERE_TO_SAVE])

# The language on which to evaluate 
for i, language in enumerate(the_languages):
    test_file = get_test_set(language, the_languages_lowercase[i])
    val_iterator = get_language_dataset(language, the_languages_lowercase[i], validate=True)

    # Create directory and copy relevant files there for later
    SERIALIZATION_DIR = WHERE_TO_SAVE + '/resultsvalidation' + language
    # Try with 5 different batches from validation set.
    for TRY in range(5):
        try:
            support_set = next(val_iterator)[0]
        except StopIteration:
            print("Stopping because val set too small")
            break 
        subprocess.run(["mkdir", SERIALIZATION_DIR])
        subprocess.run(["cp", "-r", MODEL_DIR +"/vocabulary", SERIALIZATION_DIR])
        subprocess.run(["cp", MODEL_DIR +"/config.json", SERIALIZATION_DIR])

        # Set up model and iterator and optimizer
        train_params = get_params("metatesting")
        m = Model.load(train_params, MODEL_DIR).cuda()

        if not MORE_LR:
            optimizer =  Adam(m.parameters(), LR)
        else:
            optimizer =  Adam([{'params': m.text_field_embedder.parameters(), 'lr':LR_SMALL}, 
                            {'params':m.decoders.parameters(), 'lr':LR}, 
                            {'params':m.scalar_mix.parameters(), 'lr':LR}], LR)
        # Do one forward pass
        
        for mini_epoch in range(UPDATES):
            loss = m.forward(**support_set)['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save model, then zip model
        model_save_place= SERIALIZATION_DIR + "/best.th"
        torch.save(m.state_dict(), model_save_place)
        archive_model(SERIALIZATION_DIR, files_to_archive=train_params.files_to_archive, archive_path = SERIALIZATION_DIR)

        # Get specific predictions for this finetuned guy
        current_pred_file = os.path.join(WHERE_TO_SAVE,language +'_predictions' + str(TRY) +'.conllu')
        current_output_file = os.path.join(WHERE_TO_SAVE,language+'_performance' + str(TRY) +'.json')

        util.predict_and_evaluate_model(
            "udify_predictor",
            get_params("metatesting"),
            SERIALIZATION_DIR,
            test_file,
            current_pred_file,
            current_output_file,
            batch_size=16
        )
        print("Wrote", current_output_file, "removing", SERIALIZATION_DIR)
        subprocess.run(["rm", "-r", "-f", SERIALIZATION_DIR])
