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

tran_expen = [34.55,40.20,45.16,0,19.19,59.97,84.64,58.28,50.65, #finnish
    54.12, 68.30, 32.94, 52.72, 75.45, 12.92, 33.56, 33.67, 72.09, #norwe
    44.34, 59.53, 76.02, 18.09, 54.47, 36.66, 23.46, 29.72
] 
tran_expmix = [68.20,58.95,52.62,0,23.11,84.36,81.65,61.98,62.29,
    59.54, 70.93, 85.66, 61.11, 89.44, 24.10, 44.56, 81.73, 85.11,
    56.92, 81.91, 78.70, 32.78, 64.03, 49.74, 63.06, 29.71
]

train_lans = [0,5,6,11,13,16,17,19]
low_lans = [1,2,4,7,15,23]
for i,lan in enumerate(languages):
    with open("results/" + lan + "_results_zero.json", "r") as f:
        zero_results = json.load(f)

    with open("results/" + lan + "_results_finetunelowlr160.json", "r") as f:
        ft_results_low = json.load(f)

    with open("results/" + lan + "_results_finetune160.json", "r") as f:
        ft_results = json.load(f)

    with open("results/" + lan + "_results_metazero.json", "r") as f:
        meta_results = json.load(f)

    with open("results/" + lan + "_results_metalowlr160.json", "r") as f:
        meta_results_low = json.load(f)

    with open("results/" + lan + "_results_metalowlr5.json", "r") as f:
        meta_results_low2 = json.load(f)
    

    en_zero = round(zero_results['LAS']['aligned_accuracy'], 3)
    meta_zero_high = round(meta_results['LAS']['aligned_accuracy'],3)
    finetune_low = round(ft_results_low['LAS']['aligned_accuracy'],3)
    finetune_high = round(ft_results['LAS']['aligned_accuracy'],3)
    meta_zero_low = round(meta_results_low2['LAS']['aligned_accuracy'],3)
    meta_zero_low2 = round(meta_results_low['LAS']['aligned_accuracy'],3)

    tran_en_zero = round(tran_expen[i]/100,3)
    tran_expmix_zero = round(tran_expmix[i]/100,3)

    color = '\\rowcolor{LightCyan}' if i in train_lans else ("\\rowcolor{LightRed}" if i in low_lans else '')
    print(color, lan[3:], 
        '&', en_zero, 
        '&', finetune_high,
        '&', finetune_low,
        '&', meta_zero_high,
        '&', meta_zero_low, 
        '&', meta_zero_low2,
        '&', tran_en_zero,
        '&', tran_expmix_zero,  '\\\\')



