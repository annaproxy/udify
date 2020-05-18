import matplotlib.pyplot as plt
import numpy as np 
meta_losses = []
language_losses = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: []
}
validations = np.array([
    0.7517807173747139,
    0.7479648944288985,
    0.7463113711523786,
    0.7522258967183922,
    0.7526074790129738 ,
    0.7452938183668277,
    0.7154032052912744])
languages = ["italian", "norwegian", "czech", "russian", "hindi","korean","arabic"]
#training_tasks.append(get_language_dataset('UD_Italian-ISDT','it_isdt-ud'))
#training_tasks.append(get_language_dataset('UD_Norwegian-Nynorsk','no_nynorsk-ud'))
#training_tasks.append(get_language_dataset('UD_Czech-PDT','cs_pdt-ud'))
#training_tasks.append(get_language_dataset('UD_Russian-SynTagRus','ru_syntagrus-ud'))
#training_tasks.append(get_language_dataset('UD_Hindi-HDTB','hi_hdtb-ud'))
#training_tasks.append(get_language_dataset('UD_Korean-Kaist','ko_kaist-ud'))
#training_tasks.append(get_language_dataset('UD_Arabic-PADT','ar_padt-ud'))
lan = 0 
yes = 0
with open("losses.txt", "r") as f: 
    for line in f:

        if line.startswith("meta"):
            yes += 1
            lan = 0
            meta_losses.append(float(line.split()[1].strip()))
        else:
            language_losses[lan].append(float(line.strip()))
            lan += 1

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(np.arange(yes), meta_losses, label='meta loss')
ax2.plot([10,30,40,50,60,70,80], validations, label='Bulgarian LAS')
ax1.legend()
ax2.legend()
#for i,lan in enumerate(language_losses):
#    print(len(language_losses[lan]))
#    plt.plot(np.arange(yes), language_losses[lan], label=languages[i] )

#plt.show()

fig, (ax1,ax2) = plt.subplots(2,1)


for i,lan in enumerate(language_losses):
    presmooth = np.array(language_losses[lan])
    realplot = []
    s = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
    for z in s:
        hi = np.mean(presmooth[z-5:z])
        print(hi)
        realplot.append(hi)
    if i < 4:
        ax1.plot(np.arange(len(s)), realplot, label=languages[i] )
    else:
        ax2.plot(np.arange(len(s)), realplot, label=languages[i] )
ax1.legend()
ax2.legend()

plt.show()