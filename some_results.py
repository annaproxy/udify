import json
import numpy as np 
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

def round_LAS(r):
    return round(r['LAS']['aligned_accuracy'], 3)

for i,lan in enumerate(languages):
    yeah_dictionary = {
        "english" : {"zero":{}, "test":{}}, "ne" : {"zero":{}, "test":{}}, "meta": {"zero":{}, "test":{}}
    }
    # Load all inconsistently named json files into a dictionary:
    # yeah_dictionary[model][zero|test][Training Learning Rate][Meta Testing Learning Rate]
    with open("results/" + lan + "_results_zero.json", "r") as f:
        zero_results = json.load(f)
        yeah_dictionary["english"]["zero"][None] = zero_results

    with open("results/" + lan + "_results_finetunelowlr_real.json", "r") as f:
        finetune_zero_results = json.load(f)
        yeah_dictionary["ne"]["zero"][1e-4] = finetune_zero_results

    with open("results/" + lan + "_results_metalearnlowlr_real.json", "r") as f:
        meta_zero_results = json.load(f)
        yeah_dictionary["meta"]["zero"][1e-4] = meta_zero_results

    with open("results/finetunekid/" + lan + "_performance.json", "r") as f:
        finetune_few_results = json.load(f)
        yeah_dictionary["ne"]["test"][1e-4] = {}
        yeah_dictionary["ne"]["test"][1e-4][1e-4] = finetune_few_results
    
    with open("results/metakid/" + lan + "_performance.json", "r") as f:
        meta_few_results = json.load(f)
        yeah_dictionary["meta"]["test"][1e-4] = {}
        yeah_dictionary["meta"]["test"][1e-4][1e-4] = meta_few_results

    with open("results/metatest1e4finetune1e5/" + lan + "_performance.json", "r") as f:
        lowlr_finetune_results = json.load(f)
        yeah_dictionary["ne"]["test"][1e-5] = {}
        yeah_dictionary["ne"]["test"][1e-5][1e-4] = lowlr_finetune_results

    with open("results/metatest1e4metalearn1e5/" + lan + "_performance.json", "r") as f:
        lowlr_metalearn_results = json.load(f)
        yeah_dictionary["meta"]["test"][1e-5] = {}
        yeah_dictionary["meta"]["test"][1e-5][1e-4] = lowlr_metalearn_results

    with open("results/metatest5e5metakid/" + lan + "_performance.json", "r") as f:
        metakid_lowmetatest = json.load(f)
        yeah_dictionary["meta"]["test"][1e-4][5e-5] = metakid_lowmetatest

    with open("results/metatest5e5finetune_5e5/" + lan + "_performance.json", "r") as f:
        metakid_lowfttest = json.load(f)
        yeah_dictionary["ne"]["test"][1e-4][5e-5] = metakid_lowfttest

    with open("results/finetuneonly/" + lan + "_performance.json", "r") as f:
        finetuneonly = json.load(f)
        yeah_dictionary["english"]["test"][None] = {}
        yeah_dictionary["english"]["test"][None][1e-4] = finetuneonly

    with open("results/metatestonly5e5/" + lan + "_performance.json", "r") as f:
        finetuneonly5e5 = json.load(f)
        yeah_dictionary["english"]["test"][None][5e-5] = finetuneonly5e5

    if False:
        print("=====",lan,"======")
        for k in yeah_dictionary:
            m = yeah_dictionary[k]["test"]
            zero = yeah_dictionary[k]["zero"]
            print(k, "zero", [(k,round_LAS(v)) for k,v in zero.items()])
            for lr in m:
                learning_rate = lr
                print(k, "metatest", lr, [(k,round_LAS(v)) for k,v in m[lr].items()])

    tran_en_zero = round(tran_expen[i]/100,3)
    tran_expmix_zero = round(tran_expmix[i]/100,3)

    color = '\\rowcolor{LightCyan}' if i in train_lans else ("\\rowcolor{LightRed}" if i in low_lans else '')
    print()
    
    # These are the best results
    lijstje = [ round_LAS(yeah_dictionary["english"]["zero"][None]),
         round_LAS(yeah_dictionary["ne"]["zero"][1e-4]),
         round_LAS(yeah_dictionary["meta"]["zero"][1e-4]),
         round_LAS(yeah_dictionary["english"]["test"][None][1e-4]),
         round_LAS(yeah_dictionary["ne"]["test"][1e-5][1e-4]),
         round_LAS(yeah_dictionary["meta"]["test"][1e-4][1e-4]),
         tran_en_zero,
         tran_expmix_zero]
    best = np.argmax(lijstje)
    lijstje = [('{\\bf' + str(l) + '}' if i == best else str(l)) for i,l in enumerate(lijstje)]

    print(color, lan[3:].replace('_','') + ' &', 
        ' & '.join(lijstje), '\\\\'  
    )


