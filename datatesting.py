import allennlp.data.dataset_readers as anlp
#from allennlp.data.dataloader import DataLoader
from allennlp.models.model import Model
from udify.models.udify_model import UdifyModel

from torch.utils.data import DataLoader
from i_hate_params_i_want_them_all_in_one_file import get_params
#from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from udify.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.data.iterators import BasicIterator
#from universal_dependencies import UniversalDependenciesDatasetReader
import learn2learn as l2l
from allennlp.data.vocabulary import Vocabulary
reader = UniversalDependenciesDatasetReader()


#raise ValueError("Gelukt!")
dev_dataset = reader.read('data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu')
v = Vocabulary.from_instances(dev_dataset)
#
#dev_iterator = BasicIterator(dev_dataset)
#dev_loader = DataLoader(dev_dataset, batch_size=8)
#hello = dev_dataset[0]
test = dev_dataset[0]
for x in test:
    test[x].index(v)
test = test.as_tensor_dict()
train_params = get_params()
m = UdifyModel.load(train_params, "./pretrained",)
print("laden gelukt")
result = m.forward(test)
print(result)
raise ValueError("jjj")
print(type(dev_dataset), type(dev_dataset[0]))
#print(hello.as_tensor_dict())
#boy = next(iter(dev_iterator))
#print(type(boy))
#for batch in iter(iterator):

#print(test)
raise ValueError()

#it = DataLoader(dev_dataset)
#print(type(it), type(dev_dataset))
#meta = l2l.data.MetaDataset(dev_dataset)
#t = l2l.data.TaskDataset(dev_dataset)
#print(type(t))

raise ValueError("Gelukt!")
