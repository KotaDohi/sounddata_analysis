#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:26:13 2017

@author: Dohi
"""

import pandas as pd
import wavio
import numpy as np
import matplotlib.pyplot as plt

def FTFT(filename,n0,N):              
    wav = wavio.read(filename)
    bit = 8 * wav.sampwidth
    data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
    g = data[:,0]
    g = g[::5]
    fs = wav.rate

#    G = np.fft.fft(g[n0:n0+N])                      # 高速フーリエ変換
#    amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in G]       # 振幅スペクトル
#    phase = [np.arctan2(int(c.imag), int(c.real)) for c in G]   # 位相スペクトル
    f_list = np.fft.fftfreq(N, d=1.0/fs)             # 周波数リスト
    #window = np.hamming(N)    # ハミング窓
    #window = np.hanning(N)    # ハニング窓
    #window = np.blackman(N)  # ブラックマン窓
    window = np.bartlett(N)  # バートレット窓
    G2 = np.fft.fft(window * g[n0:n0+N])
    amp2 = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in G2]
    return f_list,amp2

def Processing(filename):
    
    n0 = 10000
    N = 2048
    f,amp = FTFT(filename,n0,N)
    
    plt.figure()
    plt.plot(f[0:int(N/2)],amp[0:int(N/2)])
#    plt.xlim([0,4000])
#    plt.ylim([0,2])

Mode = 600
con1 = Processing('../data/clack_'+str(Mode)+'rpm+refinery_2.wav')