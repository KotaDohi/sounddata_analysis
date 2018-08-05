#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:29:52 2017

@author: Dohi
"""
import pandas as pd
import wavio
import numpy as np
import matplotlib.pyplot as plt

def Data_loading(filename):
    wav = wavio.read(filename)
    bit = 8*wav.sampwidth
    data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
    g = data[:,0]
    merge = 1
    g = g[::merge]
    fs = wav.rate
    f_list = np.fft.fftfreq(N, d=1.0/(fs/merge)) 
    return g,f_list

def FTFT(n0,N,g):              
    window = np.hamming(N)    # ハミング窓
    #window = np.hanning(N)    # ハニング窓
    #window = np.blackman(N)  # ブラックマン窓
    #window = np.bartlett(N)  # バートレット窓
    G = np.fft.fft(window * g[n0:n0+N])
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in G]
    return amp

def Processing(N,n0,f_list,g):
    amp = FTFT(n0,N,g)
#    plt.figure()
#    plt.plot(f_list[0:int(N/2)],amp[0:int(N/2)])
#    plt.xlim([0,4000])
#    plt.ylim([0,2])
#    plt.xlim([0,25])
    
    amp = pd.Series(amp)
    f = pd.Series(f_list)
    
    #making DataFrame
    total = pd.concat([f,amp],axis=1)
    total = pd.DataFrame(total)
    total.columns=["freq","amp"]
    total = total.sort_values(by="freq",ascending=True)
    total = total[int(len(total)/2):]
    return total

N = 2048
Mode = 1800
steps = 1200
normal = 0
sample = N/44100



#    filename(input)
#N個の行と、steps個の列を持った行列を生成
def STFT(filename):
    g,f_list = Data_loading('../data/'+filename+'.wav')

    sub = np.arange(0,N*steps,N)
    j = 0
    for n0 in sub:
        if n0 == 0:
            total = Processing(N,n0,f_list,g)
            total.columns = ['freq',j+1]
            total = total[j+1]
        else:
            present = Processing(N,n0,f_list,g)
            present.columns=['freq',j+1]
            present = present[j+1]
            total = pd.concat([total,present],axis=1)
        print(n0/N)
        j+=1
        
    print(filename,"done")
    total.index = [f_list[:int(len(f_list)/2)]]
    total.to_csv('../STFTdata/'+filename+'.csv',header ='None')
    #print(total)
    

#STFT('normal_1800rpm+refinery_1')
#STFT('normal_1800rpm+refinery_2')
#STFT('clack_1800rpm+refinery_1')
#STFT('clack_1800rpm+refinery_2')

#STFT('normal_1200rpm+refinery_1')
#STFT('normal_1200rpm+refinery_2')
#STFT('clack_1200rpm+refinery_1')
#STFT('clack_1200rpm+refinery_2')

STFT('normal_1800rpm_1')
STFT('normal_1800rpm_2')
STFT('clack_1800rpm_1')
STFT('clack_1800rpm_2')

