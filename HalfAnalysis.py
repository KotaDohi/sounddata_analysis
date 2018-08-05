#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:27:09 2017

@author: Dohi
"""
#refinery1とrefinery2のデータを別々に計算する。


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

mode = 1
Mode = 1800

L = 2000
width = 100

choice = 'D'





direc = 'learning'

if choice =='F':
    other = '_STFT'
elif choice=='Fr':
    other = '+refinery_STFT'
elif choice=='D':
    other = '_100_b'
else:
    other = '+refinery_100_b'

print(Mode,other)
num = int(3600/width)+1


filetest = '../'+direc+'/test'+str(Mode)+other+'.csv'
filetrain = '../'+direc+'/train'+str(Mode)+other+'.csv'
fileout = '../'+direc+'/predict_'+str(Mode)+other+'.csv'
   
if choice=='F' or choice=='Fr':
    d = 600
else:
    d = int(2600000/L/2)


col = ["index","decision"]
for i in range(1,num):
    col.append('~'+str(i*100))
    
df_test = pd.read_csv(filetest)
df_test.columns = col
df_train = pd.read_csv(filetrain)
df_train.columns=col




#scikit-learnに渡すためにvalueだけにする
train_data= df_train.drop(["index","decision"],axis=1).values
train_data11 = train_data[0:d,:]
train_data12 = train_data[d:2*d,:]
train_data01 = train_data[2*d:3*d,:]
train_data02 = train_data[3*d:4*d,:]
train_data1 = np.r_[train_data11,train_data01]  #only refinery1
train_data2 = np.r_[train_data12,train_data02]  #only refinery2

                   
teacher_data=df_train.ix[:,"decision"].values
teacher_data11 = teacher_data[0:d]
teacher_data12 = teacher_data[d:2*d]
teacher_data01 = teacher_data[2*d:3*d]
teacher_data02 = teacher_data[3*d:4*d]
teacher_data1 = np.r_[teacher_data11,teacher_data01]
teacher_data2 = np.r_[teacher_data12,teacher_data02]
               
                        

#学習(決定木の数100)
forest = RandomForestClassifier(n_estimators = 100,random_state=1)

if mode ==1:
    forest = forest.fit(train_data1, teacher_data1)
if mode ==2:
    forest = forest.fit(train_data2, teacher_data2)



#予測
test_data=df_test.drop(["index","decision"],axis=1).values
test_data11 = test_data[0:d,:]
test_data12 = test_data[d:2*d,:]
test_data01 = test_data[2*d:3*d,:]
test_data02 = test_data[3*d:4*d,:]
test_data1 = np.r_[test_data11,test_data01]
test_data2 = np.r_[test_data12,test_data02]

if mode ==1:
    output = forest.predict(test_data1)
if mode ==2:
    output = forest.predict(test_data2)





#最終的な出力
import csv
with open(fileout, "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["index","decision"])
    for pid, survived in zip(df_test.ix[:len(output),"index"].values.astype(int), output.astype(int)):
        writer.writerow([pid, survived])

data = df_test["decision"]
data11 = data[0:d]
data12 = data[d:2*d]
data01 = data[2*d:3*d]
data02 = data[3*d:4*d]
data1 = np.r_[data11,data01]
data2 = np.r_[data12,data02]


point = 0
pointf = 0
pointb = 0



for i in range(len(output)):
    if mode ==1:
        if output[i] == data1[i]:
            point += 1
            if i< len(output)/2:
                pointf += 1
            if i>= len(output)/2:
                pointb += 1
    if mode ==2:
        if output[i] == data2[i]:
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
plt.bar(range(len(col)-2),importances[indices],color='lightblue', align='center')

plt.xticks(range(len(col)-2), feat_labels[indices], rotation=90)
plt.xlim([-1, len(col)-2])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()

