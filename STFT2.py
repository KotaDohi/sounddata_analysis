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

def Data_loaming(filename):
    wav = wavio.read(filename)
    bit = 8 * wav.sampwidth
    data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
    g = data[:,0]
    merge = 5
    g = g[::merge]
    fs = wav.rate
    return g,fs
    


def Processing(filename,g,fs,N,n0,width):
    merge = 5
    f_list = np.fft.fftfreq(N, d=1.0/(fs/merge)) 

#    G = np.fft.fft(g[n0:n0+N])                      # 高速フーリエ変換
#    amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in G]       # 振幅スペクトル
#    phase = [np.arctan2(int(c.imag), int(c.real)) for c in G]   # 位相スペクトル
#    f_list = np.fft.fftfreq(N, d=1.0/fs)             # 周波数リスト
    #window = np.hamming(N)    # ハミング窓
    #window = np.hanning(N)    # ハニング窓
    #window = np.blackman(N)  # ブラックマン窓
    window = np.bartlett(N)  # バートレット窓
    G = np.fft.fft(window * g[n0:n0+N])
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2)*2.0/N for c in G]
    
#    plt.figure()
#    plt.plot(f[0:int(N/2)],amp[0:int(N/2)])
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
        
        
    #Introduction to RF
    points = np.arange(0,3600,width)
    con = []
    if filename[8] == 'n':
        con.append(0)
    else:
        con.append(1)
        
    for i in points:
        totalcut = total[total['freq']>=i]
        totalcut = totalcut[totalcut['freq']<=(i+width)]
        con.append(sum(totalcut['amp']))
    return con    

n0 = 2000
N = 256
Mode = 1800
width = 100
ldata = 1200
coef = 0.6


#    filename(input)
g1,fs1 = Data_loaming('../data/normal_'+str(Mode)+'rpm_1.wav')
g2,fs2 = Data_loaming('../data/normal_'+str(Mode)+'rpm_2.wav')
g3,fs3 = Data_loaming('../data/clack_'+str(Mode)+'rpm_1.wav')
g4,fs4 = Data_loaming('../data/clack_'+str(Mode)+'rpm_2.wav')


sub = np.arange(0,N*ldata*coef,N)
lsub =len(sub)
Con = np.zeros([4*lsub,int(3600/width+1)])
j = 0
for n0 in sub:
    con1 = Processing('../data/normal_'+str(Mode)+'rpm_1.wav',g1,fs1,N,n0,width)
    con2 = Processing('../data/normal_'+str(Mode)+'rpm_2.wav',g2,fs2,N,n0,width)
    con3 = Processing('../data/clack_'+str(Mode)+'rpm_1.wav',g3,fs3,N,n0,width)
    con4 = Processing('../data/clack_'+str(Mode)+'rpm_2.wav',g4,fs4,N,n0,width)
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"train")

Con = pd.DataFrame(Con)
Con.to_csv('../learning/train'+str(Mode)+'_STFT_100_256.csv')

sub = np.arange(N*ldata*coef+1,N*ldata,N)
lsub = len(sub)
Con = np.zeros([4*lsub,int(3600/width+1)])
j = 0
for n0 in sub:
    con1 = Processing('../data/normal_'+str(Mode)+'rpm_1.wav',g1,fs1,N,n0,width)
    con2 = Processing('../data/normal_'+str(Mode)+'rpm_2.wav',g2,fs2,N,n0,width)
    con3 = Processing('../data/clack_'+str(Mode)+'rpm_1.wav',g3,fs3,N,n0,width)
    con4 = Processing('../data/clack_'+str(Mode)+'rpm_2.wav',g4,fs4,N,n0,width)
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"test")

Con = pd.DataFrame(Con)
Con.to_csv('../learning/test'+str(Mode)+'_STFT_100_256.csv')