import allennlp.data.dataset_readers as anlp
from allennlp.data.dataloader import DataLoader
from universal_dependencies import UniversalDependenciesDatasetReader

reader = UniversalDependenciesDatasetReader()

#raise ValueError("Gelukt!")
dev_dataset = reader.read('data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu')
it = DataLoader(dev_dataset)
print(type(it))
raise ValueError("Gelukt!")
