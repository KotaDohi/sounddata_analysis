#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:56:00 2017

@author: Dohi
"""

#Code to make Spectrogram

import pandas as pd
import wavio
import numpy as np
import matplotlib.pyplot as plt

def Data_loading(filename):
#    merge=5
    wav = wavio.read(filename)
    data,fs = wav.data/float(2**(8*wav.sampwidth-1)),wav.rate
#    fs = fs/merge
    g,f_list = data[:,0],(np.fft.fftfreq(N, d=1.0/fs))[:int(N/2)]
    print(len(g))
    return g,f_list,fs
def FTFT(n0,N,g):
    window = np.hamming(N)    # ハミング窓    #window = np.hanning(N)    # ハニング窓    #window = np.blackman(N)  # ブラックマン窓    #window = np.bartlett(N)  # バートレット窓
    return (abs(np.fft.fft(window * g[n0:n0+N]))*2.0/N)[:int(N/2)]
def Processing(N,n0,f_list,g,flag):
    return (pd.DataFrame(np.vstack([f_list,FTFT(n0,N,g)]).T,columns=['freq',flag]).sort_values(by="freq",ascending=True))[flag]
#Nを大きくすれば周波数分解能がよくなり、小さくすると時間分解能がよくなる。
filename = 'clack_600rpm_1'
#filename='bomb'

co = 9
start = 0
end = 300

N = 2**co
def STFT(filename):
    g,f_list,fs = Data_loading('../Sounddata/'+filename+'.wav')
#    merge = 5
#    fs = fs/merge

    #moving window
    sub = np.arange(int(start*fs),int(end*fs),N)
    total = Processing(N,sub[0],f_list,g,1)
    count = 0
    for (n0,flag) in zip(sub[1:],range(len(sub))):
        total = pd.concat([total,Processing(N,n0,f_list,g,flag+2)],axis=1)
        count+=1
        print(len(sub))
        print(n0,flag)
    total = np.array(total)
    #適当な長さにカットする
    print(total.shape)
    cuts = 0
    cute = 20
    cut2 = 0
    cut3 = 6000
    cut = 0
    
    x=np.linspace(start,end,len(total.T)+1)
    if cut==1:
        x = x[cut2:cut3]
        y=f_list[cuts:cute]
        total = total[cuts:cute,cut2:cut3]
    if cut==0:
        x = x
        y=f_list
    x,y=np.meshgrid(x,y)
#    plt.figure()
#    plt.pcolor(x,y,total)#,cmap='RdBu')
#    plt.xlabel("Time[s]",fontsize=16)
#    plt.ylabel("Frequency[Hz]",fontsize=16)
#    plt.colorbar()
    print(filename,"done")
    return pd.DataFrame(total),pd.DataFrame(f_list)

total,f_list = STFT(filename)
#total.to_csv('../Sounddata2/'+filename+'.csv')
f_list.to_csv('../Sounddata2/'+filename+'flist'+'.csv')
print(total.shape)

#スペクトログラムに適用してみたら面白いかもしれない。
