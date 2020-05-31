"""
Computes correlations, similarities, all related to URIEL.
"""
import lang2vec.lang2vec as l2v
from scipy import spatial
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch.nn.functional as F
import torch
plt.style.use('ggplot')

lang2iso = {
    "Arabic": "arb",
    "Armenian": "hye",
    "Breton": "bre",
    "Buryat": "bxm",
    "Czech": "ces",
    "English": "eng",
    "Faroese": "fao",
    "Finnish": "fin",
    "French": "fra",
    "German": "deu",
    "Hindi": "hin",
    "Hungarian": "hun",
    "Italian": "ita",
    "Japanese": "jpn",
    "Kazakh": "kaz",
    "Korean": "kor",
    "Norwegian": "nor",
    "Persian": "pes",
    "Russian": "rus",
    "Swedish": "swe",
    "Tamil": "tam",
    "Telugu": "tel",
    "UpperSorbian": "hsb",
    "Urdu" : "urd",
    "Vietnamese" : "vie"
}

iso2lang = {v:k for k,v in lang2iso.items()}

feature_names = [
    "syntax_knn",
    "phonology_knn",
    "inventory_knn",
    "fam", # Membership in language families and subfamilies
    "geo", # Distance from fixed points on Earth's surface
]
#FILE = 'performancenew.csv'
COMPARE_LANS = ["eng","ces","hin","arb","ita","kor","nor", "rus"]

expMix = [lang2iso["Arabic"], lang2iso["Czech"], lang2iso["English"],
          lang2iso["Hindi"], lang2iso["Italian"], lang2iso["Korean"],
          lang2iso["Norwegian"], lang2iso["Russian"]]


class Similarities():
    '''
    Similarities based on URIEL features
    '''
    def __init__(self, lang2iso, feature_names, expMix, csv_file=None):
        self.lang2iso = lang2iso
        self.feature_names = feature_names
        self.expMix = expMix
        self.langs = l2v.available_languages()
        self.experiments = pd.read_csv(csv_file ,header=0).columns.tolist()[1:]
        self.csv_file = csv_file

    def check_availability(self):
        '''
        Checks the availability of the languages under consideration
        (defined in lang2iso)
        '''
        for x in self.lang2iso:
            print(x, lang2iso[x] in self.langs)

    def generate_distances(self, feature_name):
        '''
        Generates the distances for a specific URIEL feature
        (defined in feature_names).

        Input:
        ---------------
        feature_name: str (from feature_names)

        Output:
        feature_data: dictionary where all keys are language names
                      and the values are lists of scores
        '''
        feature_data = defaultdict(list)
        for compare_lang in self.expMix:
            features = l2v.get_features(list(lang2iso.values()), feature_name)
            for lang, iso in self.lang2iso.items():
                a = features[compare_lang]
                b = features[iso]
                c, d = [], []
                for x,y in zip(a,b):
                    if not (x == '--' or y == '--'):
                        c.append(x)
                        d.append(y)
                distance = 1 - spatial.distance.cosine(c,d)
                feature_data[lang].append(str(round(distance, 3)))
        return feature_data

    def featureData2latex(self, feature_name, feature_data):
        '''
        Prints the feature_data to latex table format.

        Input:
        ---------------
        feature_name: str
        feature_data: dictionary
        '''
        print("\n\n" + feature_name)
        print('-'*80)
        for lang in feature_data:
            print(lang + ' & ' + ' & '.join(feature_data[lang]) + ' \\\\')

    def create_avg_table(self, experiments= None):
        if experiments is None:
            experiments = self.experiments
        comparison_languages = COMPARE_LANS
        super_features = {}
        table = dict()
        for i, experiment in enumerate(experiments):
            #print("doing experiment: {}".format(experiment))
            #print()
            table[experiment] = {}
            final_xvalues = []

            for j, compare_lang in enumerate(comparison_languages):
                #print("doing language: {}".format(compare_lang))
                #table[experiment][compare_lang] = {}

                for feature_name in ["syntax_knn"]:
                    table[experiment][feature_name] = {}

                    x_values = []
                    langs = []
                    if i==0 and j==0:
                        features = l2v.get_features(list(lang2iso.values()), feature_name)
                    
                        super_features[feature_name] = features
                    else:
                        features = super_features[feature_name]
                    

                    for lang, iso in lang2iso.items():
                        langs.append(lang)
                        a = torch.tensor(features[compare_lang]).unsqueeze(0).to('cuda')
                        b = torch.tensor(features[iso]).unsqueeze(0).to('cuda')

                        distance = F.cosine_similarity(a, b).item()

                        x_values.append(distance)
                final_xvalues.append(x_values)
            test = torch.tensor(final_xvalues)
            #print("WHAT", test.shape, test)    
            #print("LENGTH" , len(torch.mean(test, dim=0)))
            thexvalues = torch.mean(test, dim=0)
            ##---------------- Determines the y-values ----------------------------
            df = pd.read_csv(self.csv_file)
            y_values = df[experiment].tolist()

            pearson, _, _ = self.compute_correlations(thexvalues, y_values)
            table[experiment]["syntax_knn"] = pearson
            #print(pearson)
            #raise ValueError()

        return table

    def create_table(self, experiments= None):
        if experiments is None:
            experiments = self.experiments
        comparison_languages = COMPARE_LANS
        super_features = {}
        table = dict()
        for i, experiment in enumerate(experiments):
            #print("doing experiment: {}".format(experiment))
            #print()
            table[experiment] = {}
            for j, compare_lang in enumerate(comparison_languages):
                #print("doing language: {}".format(compare_lang))
                table[experiment][compare_lang] = {}

                for feature_name in feature_names:
                    x_values = []
                    langs = []
                    if i==0 and j==0:
                        features = l2v.get_features(list(lang2iso.values()), feature_name)
                    
                        super_features[feature_name] = features
                    else:
                        features = super_features[feature_name]
                    

                    for lang, iso in lang2iso.items():
                        langs.append(lang)
                        a = torch.tensor(features[compare_lang]).unsqueeze(0).to('cuda')
                        b = torch.tensor(features[iso]).unsqueeze(0).to('cuda')
                        #print(a, b)
                      
                        #print(a.shape, b.shape)
                        #raise ValueError()
                        distance = F.cosine_similarity(a, b).item()

                        x_values.append(round(distance, 9))
                        
                    ##---------------- Determines the y-values ----------------------------
                    df = pd.read_csv(self.csv_file)
                    y_values = df[experiment].tolist()

                    pearson, _, _ = self.compute_correlations(x_values, y_values)
                    table[experiment][compare_lang][feature_name] = pearson

        return table

    def save_tables(self, tables):
        for experiment, _ in tables.items():
            self.save_table_latex(tables[experiment], experiment)


    def save_table_latex(self, table, experiment):
        languages = [lang for lang, _ in table.items()]
        languages_string = " & ".join(languages)
        features = [featurename for featurename, _ in next(iter(table.items()))[1].items()]

        with open('correlations/' + experiment+".txt", 'w') as f:
            f.write(languages_string+"\n")
            for feature in features:
                f.write(feature + " & " + " & ".join([str(round(table[language][feature], 2)) for language in languages]) + " \\\\\n")

    def compute_correlations(self, x_values, y_values):
        # indices of relevant languages
        lang_indices = [1,2,3,6,7,8,9,11,13,14,17,19,20,21,22,23,24]
        x_values_cor = np.array(x_values)[lang_indices]
        y_values_cor = np.array(y_values)[lang_indices]
        pearson = scipy.stats.pearsonr(x_values_cor, y_values_cor)[0]
        spearman = scipy.stats.spearmanr(x_values_cor, y_values_cor)[0]
        linreg = scipy.stats.linregress(x_values_cor, y_values_cor)

        return pearson, spearman, linreg

    def plot_similarities(self, feature_name,
                           experiment=None,
                           compare_lang=lang2iso['English'],
                           print_distances=False):
        if experiment is None:
            experiment = self.experiments[0]
        ##---------------- Determines the x-values ----------------------------
        x_values = []
        langs = []
        features = l2v.get_features(list(lang2iso.values()), feature_name)

        for lang, iso in lang2iso.items():
            langs.append(lang)
            a = features[compare_lang]
            b = features[iso]
            c, d = [], []
            for x, y in zip(a, b):
                if not (x == "--" or y == "--"):
                    c.append(x)
                    d.append(y)
            distance = 1 - spatial.distance.cosine(c, d)
            x_values.append(round(distance, 3))

        ##---------------- Determines the y-values ----------------------------
        df = pd.read_csv(self.csv_file)
        y_values = df[experiment].tolist()

        if print_distances:
            for idx, lang in enumerate(langs):
                print(lang + '-' + compare_lang + ':', '(' + str(x_values[idx])
                      + ', ' + str(y_values[idx]) + ')')

        ##---------------- Determines the y-values ----------------------------
        #fig=plt.figure()
        #ax=fig.add_axes([0,0,1,1])
        print(len(x_values))
        print(x_values)
        pearson, spearman, linreg = self.compute_correlations(x_values, y_values)
        slope = linreg.slope; intercept = linreg.intercept
        x_dots = np.array(np.linspace(4,10,10)) / 10
        y_dots = slope*x_dots + intercept

        plt.scatter(x_values, y_values)
        plt.plot(x_dots, y_dots, color="green")
        for idx, txt in enumerate(langs):
            plt.annotate(txt, (x_values[idx], y_values[idx]))

        plt.xlabel('{} Relatedness'.format(feature_name))
        plt.ylabel('{} Performance'.format(experiment))
        plt.xlim(0.4,1.0)
        plt.ylim(0.2, 0.9)
        plt.title('Language: {}, Pearson: {}, Spearman: {}'.format(compare_lang, round(pearson,3), round(spearman,3)))
        plt.show()




if __name__=='__main__':



    sim = Similarities(lang2iso, feature_names, expMix)
    #Get the data points that are to be plotted
    #experiment = experiments[0]
    table = sim.create_table()
    #sim.save_tables(table)
    #experiment_vec = np.asarray([value for key, value in experiment.items()])
    #np.save("zerofinetune.npy", experiment_vec)


    #print('experiment:', experiment)
    #sim.plot_similarities(feature_name, experiment, compare_lang, print_distances=True)
