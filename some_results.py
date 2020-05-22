import json
from naming_conventions import languages
print(len(languages))

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

middle = ["English","NE","META"]
how_many_meta = len(middle)
print("\\begin{tabular}{|l|rrr|" + how_many_meta * 'r' + "||rr|}\\hline")
print("Language & \\multicolumn{3}{c|}{Zero-Shot} & \\multicolumn{"+str(how_many_meta) + 
            "}{c|}{Meta-Testing} & \\multicolumn{2}{c|}{Tran\\&Bisazza}\\\\\\hline")
print("&English&NE&META&" + ('&'.join(middle)) + "&expEn&expMix\\\\\\hline")


for i,lan in enumerate(languages):
    with open("results/" + lan + "_results_zero.json", "r") as f:
        zero_results = json.load(f)

    with open("results/" + lan + "_results_finetunelowlr_real.json", "r") as f:
        finetune_zero_results = json.load(f)

    with open("results/" + lan + "_results_metalearnlowlr_real.json", "r") as f:
        meta_zero_results = json.load(f)

    with open("results/finetunekid/" + lan + "_performance.json", "r") as f:
        finetune_few_results = json.load(f)
    
    with open("results/metakid/" + lan + "_performance.json", "r") as f:
        meta_few_results = json.load(f)

    with open("results/metatest1e4finetune1e5/" + lan + "_performance.json", "r") as f:
        lowlr_finetune_results = json.load(f)

    with open("results/metatest1e4metalearn1e5/" + lan + "_performance.json", "r") as f:
        lowlr_metalearn_results = json.load(f)

    with open("results/metatest5e5metakid/" + lan + "_performance.json", "r") as f:
        metakid_lowmetatest = json.load(f)

    with open("results/metatest5e5finetune_5e5/" + lan + "_performance.json", "r") as f:
        metakid_lowfttest = json.load(f)

    with open("results/finetuneonly/" + lan + "_performance.json", "r") as f:
        finetuneonly = json.load(f)

    with open("results/metatestonly5e5/" + lan + "_performance.json", "r") as f:
        finetuneonly5e5 = json.load(f)


    en_zero = round(zero_results['LAS']['aligned_accuracy'], 3)
    tran_en_zero = round(tran_expen[i]/100,3)
    tran_expmix_zero = round(tran_expmix[i]/100,3)

    finetune_zero = round(finetune_zero_results['LAS']['aligned_accuracy'],3)
    meta_zero = round(meta_zero_results['LAS']['aligned_accuracy'],3)

    finetune_few = round(finetune_few_results['LAS']['aligned_accuracy'],3)
    meta_few = round(meta_few_results['LAS']['aligned_accuracy'],3)


    lowlr_finetune = round(lowlr_finetune_results['LAS']['aligned_accuracy'],3)
    lowlr_meta = round(lowlr_metalearn_results['LAS']['aligned_accuracy'],3)


    meta_lowtest = round(metakid_lowmetatest['LAS']['aligned_accuracy'],3)
    ft_lowtest = round(metakid_lowfttest['LAS']['aligned_accuracy'],3)

    metatest_only = round(finetuneonly['LAS']['aligned_accuracy'],3)
    metatest_only5e5 = round(finetuneonly5e5['LAS']['aligned_accuracy'],3)


    meta_fewstr = ('{\\bf ' + str(meta_few) +'}') if (meta_few - finetune_few) > 0.02 and (meta_few - meta_zero) > 0.02 else meta_few
    meta_zerostr = ('{\\bf ' + str(meta_zero) +'}') if (meta_zero - meta_few) > 0.02 and i not in train_lans else meta_zero



    #\begin{tabular}{|l|r|rr|rrrr||rr|}\hline
    #
    color = '\\rowcolor{LightCyan}' if i in train_lans else ("\\rowcolor{LightRed}" if i in low_lans else '')
    print()
    
    print(color, lan[3:].replace('_',''), 
        '&', en_zero, 
        '&', finetune_zero,
        '&', meta_zero, 
        '&', metatest_only,
        '&', lowlr_finetune,
        '&', metakid_lowfttest,
        '&', tran_en_zero,
        '&', tran_expmix_zero,  '\\\\')



