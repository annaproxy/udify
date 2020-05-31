# UDify with Meta-Learning
An extension of the UDify model, now with meta-learning!

# Codebase structure

## Pre-training
The code for pre-training can be found in `train_english_only.py`. 
It is vital that the entire vocabulary is used during pre-training (that is, vocabulary of all languages you would later want to train or test on).

## Meta-Learning
All code for the meta-training process can be found in `metalearn_train.py`. Currently, this only trains on the default training languages + Bulgarian as a validation language.

## Non-Episodic Learning
All code for the meta-training process can be found in `finetune_train.py`. Currently, this only trains on the default training languages + Bulgarian as a validation language.

## Evaluation 
We have two ways of evaluating:

* Zero-Shot
    This can simply be done by passing a model directory to `predict_and_eval_all_languages_separately.py`
* Few-Shot / Meta-Testing
    This is defined in `metatest_all.py`. 
    Currently, this is *very inefficient* thanks to the allennlp Predictor class. 

## Analysis
All analysis is done using `uriel.py`, `regressor.py` and of course `analysis.py`. 
