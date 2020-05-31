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

CSV_FILE = "peteete_all.csv"

with open(CSV_FILE, "w") as f:
    f.write("language,zeroenglish,zerofinetune,zerometa,testenglish,testfinetune,testmeta,tranExpEn,tranExpMix\n")


files = list()
paramtuples = list()
with open("boys.txt", "r") as f: 
    for line in f:
        yes = line.strip() 
        # metatesting_0.001_True1_metalearn_0.001_0.0001_True_1VAL
        # metatesting_0.001_True1_finetune_0.0001_TrueVAL
        files.append(line.strip())
        yes = yes.split('_')
        fine =  yes[3]
        if 'ONLY' in line: 
            paramtuples.append((yes[1],'only', '-'))
        elif fine == 'finetune':
            paramtuples.append((yes[1],fine, yes[4].replace))
        else:
            paramtuples.append((yes[1],fine, yes[4]+'-'+yes[5]))


train_lans = [0,5,6,11,13,16,17,19]
low_lans = [1,2,4,7,15,23]

middle = ["English","NE","META"]
how_many_meta = len(middle)
print("\\begin{tabular}{|l|rrr|" + how_many_meta * 'r' + "||rr|}\\hline")
#print("Language & \\multicolumn{3}{c|}{Zero-Shot} & \\multicolumn{"+str(how_many_meta) + 
#            "}{c|}{Meta-Testing} & \\multicolumn{2}{c|}{Tran\\&Bisazza}\\\\\\hline")
#print("&English&NE&META&" + ('&'.join(middle)) + "&expEn&expMix\\\\\\hline")
print("Language & \\multicolumn{1}{c}{Zero}& \\multicolumn{10}{c}{Metatest}\\\\\\hline")
print('&', '&'.join(['-']*0+[p[0] for p in paramtuples]), '\\\\\\hline')
print('&', '&'.join(['-']*0+[p[1] for p in paramtuples]), '\\\\\\hline')
print('&', '&'.join(['-']*0+[p[2] for p in paramtuples]), '\\\\\\hline')

def round_LAS(r, s=3):
    return round(r['LAS']['aligned_accuracy'], s)

for i,lan in enumerate(languages):
    #if lan == 'UD_Swedish-PUD':
    #    break

    with open("results/" + lan + "_results_zero.json", "r") as f:
        true_zero_results = json.load(f)

    with open("results/" + lan + "_results_finetunelowlr_real.json", "r") as f:
        finetune_zero_results = json.load(f)

    with open("results/" + lan + "_results_metalearnlowlr_real.json", "r") as f:
        meta_zero_results = json.load(f)

    #UD_Vietnamese-VTB_resultsZero_finetune_5e-05.json
    #UD_Vietnamese-VTB_resultsZero_metalearn_0.001_0.0001.json
    with open("resultsNewest/zero_temp/" + lan + "_resultsZero_finetune_5e-05.json", "r") as f:
        finetune_zero_results2 = json.load(f)

    with open("resultsNewest/zero_temp/" + lan + "_resultsZero_metalearn_0.001_0.0001.json", "r") as f:
        meta_zero_results2 = json.load(f)

    with open("resultsLargerBatch/zero_temp/" + lan + "_resultsZero_finetune24_5e-05.json", "r") as f:
        finetune_zero_results3 = json.load(f)

    with open("resultsLargerBatch/zero_temp/" + lan + "_resultsZero_metalearn24_0.001_5e-05.json", "r") as f:
        meta_zero_results3 = json.load(f)

    yeah = []
    for z in files:
        boy = []

        for experiment in range(0,4):
            try:
                with open("resultsLargerBatch/"+ z + "/" + lan + "_performance"+str(experiment)+".json", "r") as f:
                    results = json.load(f)
                    boy.append(results)
                
            except FileNotFoundError:
                try:
                    with open("resultsNewest/"+ z + "/" + lan + "_performance"+str(experiment)+".json", "r") as f: 
                        results = json.load(f)
                        boy.append(results)
                except:
                    #print("resultsNewest/"+ z + "/" + lan + "_performance"+str(experiment)+".json")
                    #raise ValueError("Not Found")
                    break
        
        if len(boy) > 0:
            h = np.mean([round_LAS(z,9) for z in boy])
            std = round(np.std([round_LAS(z,9) for z in boy]),3)
            #print("H", [round_LAS(z,9) for z in boy])
            yeah.append((h, std ))

        else:
            yeah.append((0,0))
    zero_results = [(round_LAS(b),3) for b in [true_zero_results, finetune_zero_results2, meta_zero_results2, finetune_zero_results2, finetune_zero_results3]]
    tran_en_zero = round(tran_expen[i]/100,3)
    tran_expmix_zero = round(tran_expmix[i]/100,3)
    meta_results = [ b for b in yeah ]
    with open(CSV_FILE, "a") as f:
        z = [   str(round_LAS(z,9)) for z in [true_zero_results, finetune_zero_results2, meta_zero_results2]]
        m = [str(j[0]) for j in yeah]
        t = [str(tran_en_zero), str(tran_expmix_zero)]
        f.write(lan[3:].split('-')[0])
        f.write(',')
        f.write(','.join(z))
        f.write(',')

        f.write(','.join(m))
        f.write(',')
        f.write(','.join(t))
        f.write('\n')


    color = '\\rowcolor{LightCyan}' if i in train_lans else ("\\rowcolor{LightRed}" if i in low_lans else '')
    print()
    
    # These are the best results
    lijstje = [z for z in  zero_results] #+ [tran_en_zero, tran_expmix_zero]
    #print(lijstje)
    #print(lijstje)
    best = np.argmax([z[0] for z in lijstje])
    lijstje = [round(x[0],3) for x in lijstje] # + [tran_en_zero, tran_expmix_zero]
    lijstje = [('{\\bf' + str(l) + '}' if i == best else str(l)) for i,l in enumerate(lijstje)]

    if i not in train_lans: 
        print(color, lan[3:].replace('_','')[:6] + ' &', 
            ' & '.join(lijstje), '\\\\'  
        )


