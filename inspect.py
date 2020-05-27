from get_language_dataset import get_language_dataset

czech = get_language_dataset('UD_Czech-PDT','cs_pdt-ud')

print(next(czech))
