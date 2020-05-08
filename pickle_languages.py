# 

from get_language_dataset import get_language_dataset

generator = get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud')
for batch in generator: 
    for key in batch[0]:
        the_len = len(batch[0][key])
        print(key, the_len, type(batch[0][key]))
        if the_len == 2: 
            for key2 in batch[0][key]:
                print('\t',key2, len(batch[0][key][key2]), type(batch[0][key][key2]))
    break