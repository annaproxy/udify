import json

languages = [
    "UD_Arabic-PADT",
    "UD_Armenian-ArmTDP",
    "UD_Breton-KEB",
    "UD_Bulgarian-BTB",
    "UD_Buryat-BDT",
    "UD_Czech-PDT",
    "UD_English-EWT",
    "UD_Faroese-OFT",
    "UD_Finnish-TDT",
    "UD_French-Spoken",
    "UD_German-GSD",
    "UD_Hindi-HDTB",
    "UD_Hungarian-Szeged",
    "UD_Italian-ISDT",
    "UD_Japanese-GSD",
    "UD_Kazakh-KTB",
    "UD_Korean-Kaist",
    "UD_Norwegian-Nynorsk",
    "UD_Persian-Seraji",
    "UD_Russian-SynTagRus",
    "UD_Swedish-PUD",
    "UD_Tamil-TTB",
    "UD_Telugu-MTG",
    "UD_Upper_Sorbian-UFAL",
    "UD_Urdu-UDTB",
    'UD_Vietnamese-VTB'
]

for lan in languages:
    with open("results/" + lan + "_results_zero.json", "r") as f:
        zero_results = json.load(f)

    with open("results/" + lan + "_results_metareal.json", "r") as f:
        meta_results = json.load(f)

    print(lan[3:], '&', round(zero_results['LAS']['aligned_accuracy'], 3), 
        '&', round(meta_results['LAS']['aligned_accuracy'],3), '\\\\')


