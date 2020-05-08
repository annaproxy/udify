# 

from get_language_dataset import get_language_dataset
from i_hate_params_i_want_them_all_in_one_file import get_params
from allennlp.models.model import Model
import learn2learn as metalearn 
from torch.optim import Adam

generator1 = get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud')
generator2 = get_language_dataset('UD_Japanese-GSD','ja_gsd-ud')
print("All Data Loaded")

train_params = get_params()
m = Model.load(train_params, "./pretrained",)
meta_m = metalearn.algorithms.MAML(m, 1e-4, True)
# TODO BERT params different update
optimizer =  Adam(meta_m.parameters(), 1e-3, betas=(0.9,0.99),weight_decay=0.01) 
tasks = [generator1, generator2]
print("Model ready")

for iteration in range(10):
    iteration_loss = 0.0
    for task_generator in tasks:
        print("Starting task")
        learner = meta_m.clone()
        print("cloned")
        support_set = next(task_generator)[0]
        query_set = next(task_generator)[0]
        print("support and query set")
        train_loss = learner.forward(**support_set)['loss']
        print("one forward loss: ", train_loss)
        raise ValueError()
        learner.adapt(train_loss)
        print("learner adapted")
        eval_loss = learner.forward(**query_set)['loss']
        iteration_loss += eval_loss
        print("Finished task")
    iteration_loss /= len(tasks)
    optimizer.zero_grad()
    iteration_loss.backward()
    optimizer.step()
    print("Success", iteration_loss.item())
    break

"""for batch in generator1: 
    for key in batch[0]:
        the_len = len(batch[0][key])
        print(key, the_len, type(batch[0][key]))
        if the_len == 2: 
            for key2 in batch[0][key]:
                print('\t',key2, len(batch[0][key][key2]), type(batch[0][key][key2]))
    break
"""