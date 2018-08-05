#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:27:09 2017

@author: Dohi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

Mode = 1800
width = 100

choice='D'



direc = 'learning'
if choice =='F':
    other = '_STFT_'+str(width)#+'_512'
elif choice=='Fr':
    other = '+refinery_STFT'
elif choice=='D':
    other = '_'+str(width)+'_mu_300_300'
#    other = '_100_mu_300_300'
else:
    other = '+refinery_'+str(width)+'_b_300_500'
print(Mode,other)



filetest = '../'+direc+'/test'+str(Mode)+other+'.csv'
filetrain = '../'+direc+'/train'+str(Mode)+other+'.csv'
fileout = '../'+direc+'/predict_'+str(Mode)+other+'.csv'

if width == 100:
    num = int(3600/width)+1
if width == 500:
    num = int(22000/width)+1
         

col = ["index","decision"]
for i in range(1,num):
    col.append(str(i*width))
df_test = pd.read_csv(filetest)
df_test.columns = col
df_train = pd.read_csv(filetrain)
df_train.columns=col


#scikit-learnに渡すためにvalueだけにする
train_data= df_train.drop(["index","decision"],axis=1).values
teacher_data=df_train.ix[:,"decision"].values

#予測
test_data=df_test.drop(["index","decision"],axis=1).values



def RandomForest(train_data,teacher_data,test_data):
    #学習(決定木の数100)
    forest = RandomForestClassifier(n_estimators = 50,random_state=0,n_jobs=2,verbose=False,criterion="entropy")
    forest = forest.fit(train_data, teacher_data)
    output = forest.predict(test_data)
    return output,forest

def SVM(train_data,teacher_data,test_data):
    svm = SVC(kernel='linear',C=1.0,random_state=0)
    svm.fit()
    

output,forest= RandomForest(train_data,teacher_data,test_data)


#最終的な出力
import csv
with open(fileout, "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["index","decision"])
    for pid, survived in zip(df_test.ix[:len(output),"index"].values.astype(int), output.astype(int)):
        writer.writerow([pid, survived])

data = df_test["decision"]
point = 0
pointf = 0
pointb = 0
for i in range(len(output)):
    if output[i] == data[i]:
        point += 1
        if i< len(output)/2:
            pointf += 1
            
        if i>= len(output)/2:
            pointb += 1
ratio = np.round(point/len(output),3)*100
ratiof = np.round(pointf/(len(output)/2),3)*100
ratiob = np.round(pointb/(len(output)/2),3)*100
print(ratio,"% (all)")
print(ratiof,"%(front)")
print(ratiob,"%(back)")




feat_labels = df_train.columns[2:]

#特徴量の重要度を抽出
importances = forest.feature_importances_

#重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]
indices = np.arange(len(importances))



#重要度の降順で特徴量の名称、重要度を表示
#for f in range(len(col)-2):
#    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[f]))

plt.title('Feature Importances')
plt.bar(range(len(col)-2),importances[indices],color='lightblue', align='center')
#plt.bar(range(len(col)-2),importances[indices],color='lightblue', align='center')
#plt.ylim([0,0.1])
plt.xticks(range(len(col)-2), feat_labels[indices], rotation=90)
plt.xlim([-1, len(col)-2])
plt.axhline(1.0/(len(col)-2),color = 'k',linestyle ='dashed')
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.xlabel('freq[Hz]')
plt.ylabel('amp')
plt.show()



variance = []
trainl = int(len(df_train)/2)

for i in range(len(col)-2):
    df = df_train.iloc[:trainl,i+2]
    stdab = df.mean()
    df = df_train.iloc[trainl:,i+2]
    stdn = df.mean()
    variance.append(stdab-stdn)

    
"""
plt.title('Feature Importances')
plt.bar(range(len(col)-2),variance,color='lightblue', align='center')

plt.xticks(range(len(col)-2), feat_labels[indices], rotation=90)
plt.xlim([-1, len(col)-2])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()"""

        
        
        
        