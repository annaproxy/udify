"""
Fits a simple linear model to the data.
Prints all kinds of correlations and plots.
"""

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import numpy as np 
from naming_conventions import languages_readable, languages 
from collections import defaultdict 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt 
from os import listdir
from os.path import isfile, join
import lang2vec.lang2vec as l2v

#print(l2v.FEATURE_SETS)                                            
features = l2v.get_features(['eng'], 'syntax_knn', header=True)                          
THE_SYNTAX_FEATURES = features['CODE']
print(THE_SYNTAX_FEATURES)
#raise ValueError()
onlyfiles = [f for f in listdir("performances") if isfile(join("performances", f))]

train_lans = [0,5,6,11,13,16,17,19]
train_lans_names = np.array(languages_readable)[train_lans]
train_lans_unreadable_names = np.array(languages)[train_lans]

def get_X_given_features(feature_list, exclude_language_list):
    X_dict = defaultdict(lambda:[])
    features = list()
    test_lans = list()
    with open("similarities.txt", "r") as f:
        # eng Arabic syntax_knn 0.6443860762255925
        for line in f:
            xs = line.strip().split()
            train_lan = xs[0]
            test_lan = xs[1]
            feature = xs[2]
            score = float(xs[3])
            if test_lan not in train_lans_names and feature in feature_list and train_lan not in exclude_language_list:
                X_dict[test_lan].append(score)
                feat = train_lan + '_' + feature
                test_lans.append(test_lan)
                if feat not in features:
                    features.append(feat)
    X = np.zeros((17, len(feature_list)*(8-len(exclude_language_list))))
    for i,k in enumerate(sorted(X_dict.keys())):
        X[i,:] = np.array(X_dict[k])
    return X, sorted(X_dict.keys()), test_lans, features

def get_thick_X():
    X = np.zeros((17, 103))
    enum = 0
    test_lans = list()

    with open("fullvectors.txt", "r") as f:
        for line in f:
            xs = line.strip().split(',')
            test_lan = xs[0]
            if test_lan not in train_lans_names:
                vector = np.array([float(b) for b in xs[1:]])
                X[enum,:] = vector 
                enum += 1
    return X, test_lans
            

def fit_model(csv_file, feature_list, exclude_language_list, plot=False):
    y = []
    with open(csv_file, "r") as f:
        first = True 
        for line in f:
            if first:
                name = line.strip().split(',')[1]
                first = False 
                continue 
            xs = line.strip().split(',')
            test_lan = xs[0]
            performance = xs[1]
            if test_lan not in train_lans_unreadable_names:
                y.append(float(performance))
    y = np.array(y)
    X, keys, testlans, features = get_X_given_features(feature_list, exclude_language_list) #[ "syntax_knn", "phonology_knn","inventory_knn"])

    #X, testlans = get_thick_X()
    loo = LeaveOneOut()

    weights = []
    mse = 0.0
    total = 0.0
    for i, (train, test) in enumerate(loo.split(X)):
        model = Ridge()
        model.fit(X[train], y[train])
        yes = model.predict(X[test])
        dif = yes - y[test]
        #print()[i], dif)
        mse += (dif)**2
        weights.append(model.coef_)
        total += 1
    #print(mse/i)
    #print(weights)

    if plot:
        fig, ax = plt.subplots()
        hot1 = np.array(weights).T
        hi = np.expand_dims(np.mean(hot1,axis=1),1)
        hott = np.hstack((hot1, hi))
        #print(hott.shape)
        #print(hott)
        sns.heatmap(hott, annot=True, ax=ax, 
                            cmap='coolwarm_r')
        ax.set_xticklabels(testlans + ["All"], rotation=30)
        ax.set_yticks(np.arange(len(features))+0.5)
        ax.set_yticklabels(features, rotation=1, verticalalignment='center')
        plt.show()
    return name, np.sqrt( mse/ total ), weights, testlans, features



#fit_model("performances/13.csv", ["fam"], [], plot=True)
#fit_model("performances/13.csv", ["syntax_knn",], [], plot=True)
#fit_model("performances/13.csv", ["syntax_knn","phonology_knn", "inventory_knn"], [], plot=True)
#fit_model("performances/30.csv", ["syntax_knn","phonology_knn", "inventory_knn"], [], plot=True)


#raise ValueError()
results = defaultdict(lambda:[])
mean_weights = defaultdict(lambda:[])
for f in onlyfiles: #"ces","hin","arb","ita","kor","nor", "rus"
    name, mse, weights , test_lans , features= fit_model(join("performances",f), [ "syntax_knn" ], [])
    results[name].append(mse)
    mean_weights[name].append(weights)

"""
# TODO: RUN THIS WITH PROPER LABELS , compare to Pearson!!!!
means = np.zeros((103, 8))
#raise ValueError()
for i, l in enumerate(results):
    hott = np.mean(np.mean(np.array(mean_weights[l]), axis=0),axis=0)
    means[:,i] = hott 

print(means.shape)

fig, ax = plt.subplots(figsize=(20,20))

#print(hott.shape)
#print(hott)

sns.heatmap(means, annot=True, ax=ax, 
                    cmap='coolwarm_r')
ax.set_xticklabels( ["english","maml","x-ne","x-maml","zero-eng", "zero-maml", "zero-x-ne", "zero-x-maml"], rotation=30)
ax.set_yticks(np.arange(len(THE_SYNTAX_FEATURES))+0.5)
ax.set_yticklabels(THE_SYNTAX_FEATURES, rotation=1, fontsize=6, verticalalignment='center')
plt.title("Average Weights for Regression Model, per Model")

plt.savefig("test.png")
plt.show()"""

#raise ValueError()
for i, l in enumerate(results):
    a = np.array(results[l])
    print(l, np.mean(a), np.std(a))
    for l2 in results:
        if l != l2:
            a2 = np.array(results[l2])
            p_value = stats.ttest_ind(a,a2 , equal_var=False).pvalue
            print(l,l2, round(np.mean(a),4), round(np.mean(a2),4), p_value)
