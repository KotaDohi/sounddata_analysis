#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:29:51 2017

@author: Dohi
"""

import numpy as np
import matplotlib.pyplot as plt
import wavio
import pandas as pd
from numpy import dot,diag,multiply,power,pi
from numpy.linalg import inv,pinv
from scipy.linalg import svd,eig
from termcolor import cprint


#making data
filename = str(900)+"Hz.wav"

stop = 3.0 #time(sec)
Srate = 8000 #sampling frequency

freq1 = 1000
rad1 = np.linspace(0, 2 * np.pi * freq1 * stop, Srate * stop)
freq2 = 2000
rad2 = np.linspace(0, 2 * np.pi * freq2 * stop, Srate * stop)
freq3 = 3000
rad3 = np.linspace(0, 2 * np.pi * freq3 * stop, Srate * stop)

rad4 = np.zeros(int(Srate*stop+1))
for i in range(len(rad1)):
    rad4[i] = int(np.random.randn())

drate= 300
rad5 = []
for i in range(len(rad1)):
    if i < len(rad1):
        rad5.append(drate*i/Srate)
    else:
        rad5.append(0)

rad6 = np.linspace(0, 2 * np.pi * freq4 * stop, Srate)


                   
y1 = np.zeros(len(rad1))
for i in range(len(rad1)):
    y1[i] = np.sin(rad1[i])+np.sin(rad2[i])+np.sin(rad3[i])+rad4[i]

#これさ、減衰のレートにノイズ掛け合わせると、結果面白い
t = np.linspace(0,stop,stop*Srate+1)
#xclean = 14*np.sin(7*2*pi*t)+5*np.sin(13*2*pi*t)
#x1 = xclean+rad4
wavio.write(filename,y1,Srate,sampwidth=2)


#Data_loaming
def Data_loaming(filename):
    wav = wavio.read(filename)
    bit = 8 * wav.sampwidth
    data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
    g = data[:,0]
    fs = wav.rate
    g = y1
    return g,fs
l

#FTFT
def FTFT(filename,n0,N):     
    g,fs = Data_loaming(filename)
    f_list = np.fft.fftfreq(N, d=1.0/fs)             # 周波数リスト
    print(f_list)
    #窓関数を用いないとDMDとSTFTで近くなる
#    window = np.hamming(N)    # ハミング窓
    window = np.hanning(N)    # ハニング窓
#    window = np.blackman(N)  # ブラックマン窓
#    window = np.bartlett(N)  # バートレット窓
    G = np.fft.fft(window*g[n0:n0+N])
    amp = [np.sqrt(c.real ** 2 + c.imag ** 2)*2.0/N for c in G]
    
    return f_list,amp


#DMD
def DMD(Data,dt,r):
    X = Data[:,:-1]
    Y = Data[:,1:]
    U,Sig,Vh = svd(X, False)
    #truncation
    U = U[:,:r]
    Sig = diag(Sig)[:r,:r]
    V = Vh.conj().T[:,:r]
    # build A tilde
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig))
    mu,W= eig(Atil,right=True)

    #cal omega
    omega = np.arange(len(mu))
    for i in range(len(mu)):
        omega[i] = abs((np.log(mu[i])/(2.0*np.pi*dt)).imag) 
    
    #cal Phi
    Phi = dot(dot(dot(Y, V), inv(Sig)), W)
    b = dot(pinv(Phi), X[:,0])

    plt.figure()
    plt.scatter(mu[:].real,mu[:].imag)
#    for i in range(len(mu)):
#        print (mu.real**2+mu.imag**2)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    return omega,mu,b,Sig




#Normal DMD
def Data_DMD(filename,L,snaps,r):
    x,fs = Data_loaming(filename)

    dt = 1.0/fs
    t = np.linspace(0,dt*(L+snaps),(L+snaps))
    
    plt.figure()
#    plt.plot(t[:100],x[:100])
    plt.xlabel("Time [s]")
    plt.ylabel("State")
    
    Data = np.zeros((L,snaps))
    for i in range(snaps):
        Data[:,i] = x[i:L+i]
    
    omega,mu,b,Sig = DMD(Data,dt,r)
    Sig = np.diag(Sig)
    con = np.zeros(len(Sig))
    for i in range(len(Sig)):
        con[i] = Sig[i]/sum(Sig)

    for i in range(len(b)):
        b[i] = abs(b[i])*2.0/np.sqrt(snaps)#*con[i]
    return omega,b


#DMD section
L = 500
snaps = 500
r = 500
omega, b = Data_DMD(filename,L,snaps,r)


#FTFT section
N = 1024
n0 = 0
f,amp = FTFT(filename,n0,N)
ticks = np.linspace(0,10,11)
plt.figure(figsize=(12,4))
plt.subplots_adjust(wspace=0.4,hspace=0.2)
plt.subplot(1,2,1)
plt.title("STFT")
plt.scatter(f[0:int(N/2)],amp[0:int(N/2)])
plt.xlabel("Frequency[Hz]")
#plt.ylabel("State")
plt.grid()
#plt.xticks(ticks)
#plt.ylim([0,0.001])
plt.xlim([-100,4100])



#DMD plot 
plt.subplot(1,2,2)
plt.title("DMD")
plt.scatter(omega,b)
plt.xlabel("Frequency[Hz]")
#plt.ylabel("Srate")
#plt.xticks(ticks)
#plt.ylim([0,1])
plt.xlim([-100,4100])
plt.grid()

#あとはSTFTでは検知できないような非定常性をどうやって説明するかとかそういう話になる。


