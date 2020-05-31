import os 
import json
import numpy as np 
from naming_conventions import languages, languages_readable
from uriel import Similarities
import uriel
import copy
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt 
from scipy import stats
from matplotlib import colors
import seaborn as sns

"""Hardcode All This"""
train_lans = [0,5,6,11,13,16,17,19]
low_lans = [1,2,4,7,15,23]

tran_expen = np.array([34.55,40.20,45.16,0,19.19,59.97,84.64,58.28,50.65, #finnish
    54.12, 68.30, 32.94, 52.72, 75.45, 12.92, 33.56, 33.67, 72.09, #norwe
    44.34, 59.53, 76.02, 18.09, 54.47, 36.66, 23.46, 29.72
]) / 100
tran_expmix = np.array([68.20,58.95,52.62,0,23.11,84.36,81.65,61.98,62.29,
    59.54, 70.93, 85.66, 61.11, 89.44, 24.10, 44.56, 81.73, 85.11,
    56.92, 81.91, 78.70, 32.78, 64.03, 49.74, 63.06, 29.71
])/ 100

class ModelAnalysis():
    def __init__(self, model_str, seed_list=[1,2,5,6,7]):
        spl = model_str.split('_')
        name = spl[0]
        params = spl[1] 
        z = ""
        if 'finetune' not in name:
            params += '_' + spl[2] 
            z = "_3"
        self.model_str = name + "24_seedX_" + params + "_True" + z +  "VAL_averaging"
        self.model_str2 = name + "24_seedX_" + params + "_True" + z +  "VAL"

        self.seed_list = seed_list
        self.files_highlr = {}
        self.files_lowlr = {}
        self.files_zeroshot = defaultdict(lambda:[])
        self.las_highlr = {}
        self.las_lowlr = {}
        self.las_zeroshot = {}

        self.whole_dict = {}
        for l in languages:
            self._set_files_for_language(l)
        for l in languages:
            self._set_las_scores(l)
        print(name, [len(self.las_highlr[z]) for z in self.las_highlr], [len(self.las_lowlr[z]) for z in self.las_lowlr] )
        
    def _set_files_for_language(self, lan):
        files_highlr = []
        files_lowlr = []
        for outerseed in self.seed_list:
            testingstr = "metatesting_0.001_True3_" + self.model_str.replace('X', str(outerseed))
            testingstrlow = "metatesting_0.0001_True3_" + self.model_str.replace('X', str(outerseed))
            zeroshotstr = "finalresults/zeroshot_" + self.model_str2.replace('X', str(outerseed)) + "/" + lan + "_performance.json"

            if os.path.exists(zeroshotstr):
                self.files_zeroshot[lan].append(zeroshotstr)
            #else:
                #print("File not found", zeroshotstr)
                #raise ValueError()

            for innerseed in range(0,5):
                f = "finalresults/" + testingstr + "/" + lan + "_performance" + str(innerseed) +".json"
                f2 = "finalresults/" + testingstrlow + "/" + lan + "_performance" + str(innerseed) +".json"
                if os.path.exists(f):
                    files_highlr.append(f)
                if os.path.exists(f2):
                    files_lowlr.append(f2)
                #else:
                #    print("File not found", f2)
            
        self.files_highlr[lan] = files_highlr
        self.files_lowlr[lan] = files_lowlr

    def _set_las_scores(self, lan):
        scores = [] 
        for f in self.files_highlr[lan]:
            with open(f,'r') as results:
                result = json.load(results)
                scores.append(result['LAS']['aligned_accuracy'])
        #print(scores)
        self.las_highlr[lan] = np.array(scores)
        scores = []
        for f in self.files_lowlr[lan]:
            with open(f,'r') as results:
                result = json.load(results)
                scores.append(result['LAS']['aligned_accuracy'])
        self.las_lowlr[lan] = np.array(scores)
        scores = []
        for f in self.files_zeroshot[lan]:
            with open(f,'r') as results:
                result = json.load(results)
                scores.append([result['LAS']['aligned_accuracy']]*5)
        self.las_zeroshot[lan] = np.array(scores)

    def get_mean_sd_high(self, lan, r=99):
        b = self.las_highlr[lan]
        return round(np.mean(b), r), round(np.std(b),r)

    def get_mean_sd_low(self, lan, r=99):
        b = self.las_lowlr[lan]
        return round(np.mean(b), r), round(np.std(b), r)

    def get_mean_sd_zero(self, lan, r=99):
        b = self.las_zeroshot[lan]
        return round(np.mean(b), r), round(np.std(b), r)

class FileAnalysis(ModelAnalysis):
    def __init__(self, filenames, name, zero=False):
        self.name = name
        self.zero = zero
        self.las_lowlr = {}
        if zero:
            self.zero_init(filenames)
        else: 
            self.files_highlr = defaultdict(lambda:[])
            self.files_lowlr = defaultdict(lambda:[])
            self.las_highlr = {}
            
            
            for filename in filenames:
                for lan in languages:
                    for innerseed in range(0,5):
                        f = "finalresults/metatesting_0.001_True3_" + filename + "/" + lan + "_performance" + str(innerseed) +".json"
                        f2 = "finalresults/metatesting_0.0001_True3_" + filename + "/" + lan + "_performance" + str(innerseed) +".json"
                        if os.path.exists(f):
                            self.files_highlr[lan].append(f)
                        if os.path.exists(f2):
                            self.files_lowlr[lan].append(f2)
                            #if innerseed == 0:
                            #    print("Using file", f2)
        for lan in languages:
            self._set_las_scores(lan)

    def zero_init(self, filenames):
        self.files_lowlr = defaultdict(lambda:[])
        for filename in filenames:
            for lan in languages:
                f2 = "finalresults/zeroshot_" + filename + "/" + lan + "_performance.json"
                if os.path.exists(f2):
                    r = 1
                    if len(filenames)==1:
                        r = 3 
                    for i in range(3):
                        self.files_lowlr[lan].append(f2)
            
            
    def _set_las_scores(self, lan):
        if self.zero:
            scores = []
            for f in self.files_lowlr[lan]:
                with open(f,'r') as results:
                    result = json.load(results)
                    scores.append(result['LAS']['aligned_accuracy'])
            self.las_lowlr[lan] = np.array(scores)
        else:
            scores = [] 
            for f in self.files_highlr[lan]:
                with open(f,'r') as results:
                    result = json.load(results)
                    scores.append(result['LAS']['aligned_accuracy'])
            #print(scores)
            #self.las_highlr[lan] = np.array(scores)
            scores = []
            for f in self.files_lowlr[lan]:
                with open(f,'r') as results:
                    result = json.load(results)
                    scores.append(result['LAS']['aligned_accuracy'])
            self.las_lowlr[lan] = np.array(scores)

   
    
    def print_all(self):
        for lan in languages:
            print('---')
            print(lan,'\t',self.get_mean_sd_high(lan, 3), self.get_mean_sd_low(lan, 3))

class MetaListAnalysis():
    def __init__(self, filelist, nameslist):
        self.filelist = filelist
        self.names = nameslist
        self.accuracy_significance = {}
        self.correlation_significance = {}
        self.correlations = {}
        self.lookup = {name:i for i,name in enumerate(nameslist)}
        for f in filelist:
            for i,lan in enumerate(languages):
                self.accuracy_significance[lan] = {}
                for name1,model1 in zip(nameslist,filelist+[0,0]):
                    self.accuracy_significance[lan][name1]= {}
                    for name2,model2 in zip(nameslist,filelist+[0,0]):
                        if name2 != name1:
                            
                            if 'tran-en' in name1:
                                array1= [tran_expen[i]]*5
                            elif 'tran-mix' in name1:
                                array1 = [tran_expmix[i]]*5
                            else:
                                array1 = model1.las_lowlr[lan]

                            if 'tran-en' in name2:
                                array2= [tran_expen[i]]*5
                            elif 'tran-mix' in name2:
                                array2 = [tran_expmix[i]]*5
                            else:
                                array2 = model2.las_lowlr[lan]
                            p_value = stats.ttest_ind(array1, array2 , equal_var=False).pvalue 
                            #print("setting", name1,name2,lan)
                            self.accuracy_significance[lan][name1][name2] = p_value

    def print_latex(self, filename, train=False, print_sd =False):
        
        with open(filename,'w') as f:
            f.write(' &' + ' & '.join(self.names) + '\\\\\\hline\n')

        for i, lan in enumerate(languages):
            readable_lan = languages_readable[i]

            lijstje = np.array([m.get_mean_sd_low(lan,7)[0] for m in self.filelist[:-2]] + [tran_expen[i], tran_expmix[i]])
            sds = np.array([m.get_mean_sd_low(lan,7)[1] for m in self.filelist[:-2]] + [0,0])

            #print([m.name for m in self.filelist])
            #print(lijstje)
            max_index = np.nanargmax(lijstje)
            notmax_lijstje = np.delete(lijstje, max_index)
            max_index2 = np.nanargmax(notmax_lijstje)
            names2 = np.delete(np.array(self.names), max_index)
            color = "\\rowcolor{LightRed}" if i in low_lans else ''


            #print(max_index2, max_index, readable_lan)
            significance = self.accuracy_significance[lan][self.names[max_index]][names2[max_index2]]
            #print("Is it significant?", readable_lan, self.names[max_index], names2[max_index2], significance)

            #print( '\t', significance )
            lijstje = ['*{\\bf ' + str(round(l,3)) + '}' 
                        if (i == max_index and significance < 0.01 and max_index < (len(self.names)-2)) 
                        else ('{\\bf ' + str(round(l,3)) + '}' if (i==max_index)
                        else str(round(l,3)) )
                        for i,l in enumerate(lijstje)]
            lijstje = [ l + ('\\tiny{$\\pm$ '+str(round(sd,3))+'}' if z< (len(self.names)-2) and print_sd else '') for z, (l, sd) in enumerate(zip(lijstje, sds))]
            if i not in train_lans and not train:
                # Write normal resource
                with open(filename,'a') as f:
                    f.write(color),
                    f.write(readable_lan + ' & ')
                    f.write(' & '.join(lijstje))
                    f.write('\\\\\n')
                # Write low resources
            elif i in train_lans and train:
                with open(filename,'a') as f:
                    f.write(readable_lan + ' & ')
                    f.write(' & '.join(lijstje))
                    f.write('\\\\\n')
    def compare_two_columns(self, name1, name2):
        count = 0
        for i, lan in enumerate(languages):
            if i not in train_lans and 'ulg' not in lan:
                significance = self.accuracy_significance[lan][name1][name2]
                print(lan, significance)
                if significance < 0.01:
                    count += 1
        print(count)
        return count

    def plot_diffs(self, experiment_list=["english","maml","x-ne"], comparator = "x-maml"):
        plt.rcParams["axes.grid"] = False
        diffs = np.zeros((17,len(experiment_list)))
        pvalues = np.zeros((17,len(experiment_list)))
        labels = np.empty((17,len(experiment_list)),dtype=object)
        enum = 0
        real_lans = []
        for i, lan in enumerate(languages):
            
            if i not in train_lans and 'ulg' not in lan:
                for j,setup in enumerate(experiment_list):
                    lookup = self.filelist[self.lookup[setup]]
                    mean_comp = self.filelist[self.lookup[comparator]].get_mean_sd_low(lan,7)[0]*100
                    if type(lookup) is str:
                        if 'en' in lookup:
                            mean_comp = tran_expen[i]*100
                        else:
                            #print(lan, i, tran_expmix[i]*100,mean_comp)
                            mean_setup = tran_expmix[i]*100

                    else:
                        mean_setup = lookup.get_mean_sd_low(lan,7)[0]*100
                    diffs[enum,j] = mean_comp - mean_setup
                    pvalue = self.accuracy_significance[lan][comparator][setup]
                    pvalues[enum,j] = pvalue
                    labels[enum, j] = str(round(diffs[enum,j],2)) + ('*' if pvalues[enum,j] < 0.01 else '')
                enum += 1
                
                real_lans.append(languages_readable[i])
        fig, ax = plt.subplots()
        print(labels)
        rdgn = sns.diverging_palette(h_neg=10, h_pos=250, s=99, l=55, sep=3, as_cmap=True)
        #labels =  np.array([['A','B'],['C','D'],['E','F']])
        g = sns.heatmap(diffs, annot=labels, ax=ax,  fmt = '',
                    cmap=rdgn, vmin=-3, center=0, vmax=30)
            # We want to show all ticks...
        ax.set_yticklabels(real_lans, rotation=1)
        ax.set_xticklabels(experiment_list, horizontalalignment='center')
        for low in [0,1,2,3,9,14]:
            ax.get_yticklabels()[low].set_color("red")
        ax.set_xlabel("Baseline")
        #g.set_axis_labels("Baseline", "Language")
        #ax.set_xticks(np.arange(5))
        #ax.set_yticks(np.arange(len(real_lans)))
        # ... and label them with the respective list entries.
        #ax.set_xticklabels(setups)
        #ax.set_yticklabels(real_lans)
        
        #im, cbar = tools.heatmap(diffs, real_lans, setups, ax=ax,
       #             cmap="RdYlGn", vmin=28, center=0, vmax=-5)
       # texts = tools.annotate_heatmap(im, pvalues, valfmt="{x:.1f}", fontsize=8)
        ax.set_title("X-MAML Improvement" + (" (Zero-Shot)" if "zero" in experiment_list[0] else " (Few-Shot) "))
        fig.tight_layout()
        plt.show() 

    def plot_diffs_pairwise(self):
        plt.rcParams["axes.grid"] = False
        diffs = np.zeros((9,4))
        pvalues = np.zeros((9,4))
        labels = np.empty((9,4),dtype=object)
        enum = 0
        real_lans = []

        zeros = ["zero-eng","zero-maml","zero-x-ne","zero-x-maml"]
        fews =  ["english","maml","x-ne","x-maml"]

        for i, lan in enumerate(languages):
            if i  in train_lans or 'ulg' in lan:
                print(lan)
                for j,setup in enumerate(zeros):
                    #print(zeros[])
                    lookup = self.filelist[self.lookup[setup]]
                    mean_comp = self.filelist[self.lookup[fews[j]]].get_mean_sd_low(lan,7)[0]*100
                    if type(lookup) is str:
                        if 'en' in lookup:
                            mean_comp = tran_expen[i]*100
                        else:
                            #print(lan, i, tran_expmix[i]*100,mean_comp)
                            mean_setup = tran_expmix[i]*100

                    else:
                        mean_setup = lookup.get_mean_sd_low(lan,7)[0]*100
                    diffs[enum,j] = mean_comp - mean_setup
                    pvalue = self.accuracy_significance[lan][fews[j]][setup]
                    pvalues[enum,j] = pvalue
                    labels[enum, j] = str(round(diffs[enum,j],2)) + ('*' if pvalues[enum,j] < 0.01 else '')
                enum += 1
                
                real_lans.append(languages_readable[i])
        fig, ax = plt.subplots()
        print(labels)
        rdgn = sns.diverging_palette(h_neg=10, h_pos=250, s=99, l=55, sep=3, as_cmap=True)
        #labels =  np.array([['A','B'],['C','D'],['E','F']])
        g = sns.heatmap(diffs, annot=labels, ax=ax,  fmt = '',
                    cmap=rdgn, vmin=-3, center=0, vmax=6)
            # We want to show all ticks...
        ax.set_yticklabels(real_lans, rotation=1)
        ax.set_xticklabels(fews, horizontalalignment='center')
        #for low in [0,1,2,3,9,14]:
        #    ax.get_yticklabels()[low].set_color("red")
        ax.set_xlabel("Model")
        #g.set_axis_labels("Baseline", "Language")
        #ax.set_xticks(np.arange(5))
        #ax.set_yticks(np.arange(len(real_lans)))
        # ... and label them with the respective list entries.
        #ax.set_xticklabels(setups)
        #ax.set_yticklabels(real_lans)
        
        #im, cbar = tools.heatmap(diffs, real_lans, setups, ax=ax,
       #             cmap="RdYlGn", vmin=28, center=0, vmax=-5)
       # texts = tools.annotate_heatmap(im, pvalues, valfmt="{x:.1f}", fontsize=8)
        #ax.set_title("X-MAML Improvement" + (" (Zero-Shot)" if "zero" in experiment_list[0] else " (Few-Shot) "))
        ax.set_title("Improvement of Few-Shot over Zero-Shot", fontsize=11)
        fig.tight_layout()
        plt.show() 

    def get_results(self, which):
        model = self.filelist[which] 
        print("Doing", model.name)
        l = len(model.las_lowlr['UD_Arabic-PADT'])
        self.correlations[model.name] = []
        for index in range(l):
            filename = str(which) + str(index) + '.csv'
            with open(filename,'w') as f: 
                f.write('language,' + model.name + '\n')
            for lan in languages:
                if 'Bulg' not in lan:
                    with open(filename, 'a') as f:
                        f.write(lan + ',')
                        f.write(str(model.las_lowlr[lan][index]))
                        f.write('\n')
            sim = Similarities(uriel.lang2iso, uriel.feature_names, uriel.expMix, filename)
            table = sim.create_table()
            self.correlations[model.name].append(table)
        return self.correlations[model.name]

    def compare_correlations(self):
        if not os.path.exists('correlations.pickle'):

            for i in range(len(self.filelist[:-2])):
                self.get_results(i)

            with open("correlations.pickle", "wb") as f:
                pickle.dump(self.correlations, f)
        else:
            with open('correlations.pickle', 'rb') as f:
                self.correlations = pickle.load(f)

        bigtable = np.zeros((8*5,8))

        enum = -1
        yticklabels = []
        for lan in uriel.COMPARE_LANS:
            self.correlation_significance[lan] = {}
            for feature in ["syntax_knn"]:
                enum += 1
                yticklabels.append(lan) #+"_"+feature)
                for j, name1 in enumerate(self.names[:-2]):
                    self.correlation_significance[lan][name1] = {}
                    bigtable[enum,j] = np.mean(np.array([d[name1][lan][feature] for d in self.correlations[name1]]))
                    for name2 in self.names[:-2]:
                        if name1 != name2: 
                            lang = uriel.iso2lang[lan]
                            #print(type(name1))
                            #if name1 == 'eng': name1 = "english"
                            #if name2 == 'eng': name2 = "english"
                            #print(self.correlations[name1])
                            array1 = [d[name1][lan][feature] for d in self.correlations[name1]]
                            array2 = [d[name2][lan][feature] for d in self.correlations[name2]]
                            p_value = stats.ttest_ind(array1,array2 ,equal_var=False).pvalue 
                            self.correlation_significance[lan][name1][name2] = p_value
                            #if p_value < 0.1:
                            with open("hi222.txt", "a") as f:
                                f.write(lang+' '+feature+' '+name1+' '+name2 + ' ')
                                f.write(str(round(np.mean(np.array(array1)),4)) + ' ')
                                f.write(str(round(np.mean(np.array(array2)),4)) + ' ')
                                f.write(str(p_value))
                                f.write('\n')
        fig, ax = plt.subplots()
        
        rdgn = sns.diverging_palette(145, 280, s=85, l=25, n=7, as_cmap=True) #sns.diverging_palette(h_neg=10, h_pos=250, s=99, l=55, sep=3, as_cmap=True)
       
        #labels =  np.array([['A','B'],['C','D'],['E','F']])
        g = sns.heatmap(np.array(bigtable[:8])[3,1,0,2,4,5,6,7], annot=True, ax=ax,
                    cmap=rdgn, vmin=-1, center=0, vmax=1)
        ax.set_yticks(np.arange(len(yticklabels))+0.5, )
        ax.set_xticks(np.arange(len(self.filelist[:-2]))+0.5)
        ax.set_yticklabels(yticklabels, rotation=1, verticalalignment='center')
        ax.set_xticklabels([b.name for b in self.filelist[:-2]], rotation=30, horizontalalignment='center')
        ax.set_xlabel("Model")
        ax.set_ylabel("Language for syntax features")
        plt.show()
        

f = FileAnalysis(["finetune24_MAML_0.0001_TrueVAL_averaging"], "bad")
english = FileAnalysis(["ONLY_averaging"], "english")
maml = FileAnalysis(["metalearn24_MAMLC_0.001_5e-05_True_3VAL_averaging", 
                    "metalearn24_MAMLC2_0.001_5e-05_True_3VAL_averaging"
                    "metalearn24_MAML9C_0.001_5e-05_True_3VAL_averaging",
                    "metalearn24_MAML10C_0.001_5e-05_True_3VAL_averaging"], "maml")
ne = FileAnalysis([
    "finetune24_seed1_0.0001_TrueVAL_averaging",
    "finetune24_seed6_0.0001_TrueVAL_averaging",
    "finetune24_seed7_0.0001_TrueVAL_averaging", # 7
    "finetune24_seed8_0.0001_TrueVAL_averaging"
],"x-ne")
xmaml = FileAnalysis([ "metalearn24_seed1_0.001_5e-05_True_3VAL_averaging",
                    "metalearn24_seed2_0.001_5e-05_True_3VAL_averaging",
                    "metalearn24_seed5_0.001_5e-05_True_3VAL_averaging",
                    "metalearn24_seed6_0.001_5e-05_True_3VAL_averaging"],"x-maml")

zerone = FileAnalysis(["finetune24_seed7_0.0001_TrueVAL", 
                        "finetune24_seed6_0.0001_TrueVAL",
                        "finetune24_seed8_0.0001_TrueVAL"
                        ], "zero-x-ne", zero=True)
zeroen = FileAnalysis(["english"],"zero-eng", zero=True)
zerox = FileAnalysis(["metalearn24_seed1_0.001_5e-05_True_3VAL", 
                    "metalearn24_seed2_0.001_5e-05_True_3VAL",
                    "metalearn24_seed5_0.001_5e-05_True_3VAL",
                    "metalearn24_seed6_0.001_5e-05_True_3VAL"], "zero-x-maml", zero=True)

zeromaml = FileAnalysis(["metalearn24_MAMLC_0.001_5e-05_True_3VAL", 
                    "metalearn24_MAMLC2_0.001_5e-05_True_3VAL"
                    "metalearn24_MAML9C_0.001_5e-05_True_3VAL",
                    "metalearn24_MAML10C_0.001_5e-05_True_3VAL"], "zero-maml", zero=True)


# Our Meta Analysis class
meta = MetaListAnalysis(
    [english,maml,ne, xmaml, zeroen, zeromaml, zerone, zerox, "tran-en", "tran-mix"], 
    ["english","maml","x-ne","x-maml","zero-eng", "zero-maml", "zero-x-ne", "zero-x-maml", "tran-en", "tran-mix"])

"""Latex"""
#meta.print_latex("all_lans.tex", print_sd=True)
#meta.print_latex("train_lans.tex", True, print_sd=True)
#meta.print_latex("test_lans_small.tex",)
#meta.print_latex("train_lans_small.tex", True,)

"""Plotting"""
#meta.plot_diffs()
#meta.plot_diffs_pairwise()


"""Getting p-values for each two columns"""
#meta.compare_two_columns("english","x-maml")
#meta.compare_two_columns("maml","x-maml")
#meta.compare_two_columns("x-ne","x-maml")
#meta.compare_two_columns("zeroen","zerox")
#meta.compare_two_columns("zerone","zerox")
#meta.compare_two_columns("zerox","x-maml")

"""Doing correlation study"""
#meta.compare_correlations()
