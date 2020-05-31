# Manual Pipeline for pre-training on English
* Collect all relevant datasets in one directory (manually done)
* Concat them and build vocabulary (using concat_treebanks.py, create_vocabs.py or train.py with valueerror)
* Point the train_english_only to correct vocabulary directory
* Run train_english_only. Make sure all the config files are correct:
    - Seed
    - Batch size
    - Vocabulary directory
    - Training files (only English-EWT)

