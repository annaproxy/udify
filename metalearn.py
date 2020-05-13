#

from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
import learn2learn as metalearn
from torch.optim import Adam
from torch import autograd

# Tran&Bisazza: Italian, Norwegian, 
# Czech, Russian, Hindi, Korean, Arabic

training_tasks = []
training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))

print("All Data Loaded")

train_params = get_params()
m = Model.load(train_params, "logs/english_only_expmix4/2020.05.13_01.43.52",)
meta_m = metalearn.algorithms.MAML(m, 1e-4, True)

# TODO BERT params different update?
optimizer =  Adam(meta_m.parameters(), 1e-4)
print("Model ready")


for iteration in range(100):
    iteration_loss = 0.0
    for task_generator in training_tasks:
        learner = meta_m.clone()
        print("cloned")
        support_set = next(task_generator)[0]
        query_set = next(task_generator)[0]
        print("support and query set")
        inner_loss = learner.forward(**support_set)['loss']
        print("one forward loss: ", inner_loss)
        learner.adapt(inner_loss, first_order=True)
        print("learner adapted")
        eval_loss = learner.forward(**query_set)['loss']
        iteration_loss += eval_loss
        print("Finished task")
    iteration_loss /= len(training_tasks)
    optimizer.zero_grad()
    iteration_loss.backward()
    optimizer.step()
    print("Success", iteration_loss.item())




