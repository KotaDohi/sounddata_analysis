#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:54:01 2017

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


class Analysis:
    
    def __init__(self):
        self.filename = ""
        
    def Rawdata_loading(self,filename,N):
        wav = wavio.read(filename)
        bit = 8*wav.sampwidth
        data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0(normalize)
        g = data[:,0]
        print(len(g))
        fs = wav.rate
        return g,fs
    

    def DMD(self,g,fs,L,snaps,r,sub):
        Data = np.zeros((L,snaps))
        for i in range(snaps):
            Data[:,i] = g[sub+i:sub+L+i]
            
        X = Data[:,:-1]
        Y = Data[:,1:]
        
        U,Sig,Vh = svd(X, False)
        
        #truncation
        U = U[:,:r]
        Sig = diag(Sig)[:r,:r]
        V = Vh.conj().T[:,:r]
    
        
        # build A tilde
        Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig))
        mu,Wr,Wl= eig(Atil,right=True,left=True)
        
        
        #cal omega
        omega = np.arange(len(mu))
        for i in range(len(mu)):
            omega[i] = abs((np.log(mu[i])/(2.0*np.pi*1.0/fs)).imag)

        #cal Phi
        t = np.linspace(0,1.0/fs*len(Data.T),len(Data.T))
        Phi = dot(dot(dot(Y, V), inv(Sig)), Wr)
        b = dot(pinv(Phi), X[:,0])
        dt = 1/fs
        Psi = np.zeros([r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        
        plt.figure()
        for i in range(r):
            plt.plot(t,Psi[i,:])
        return omega,mu,b,Sig,Data,Phi
    

    def main(self,g,fs,L,snaps,r,sub):
        omega,mu,b,Sig,Data,Phi = self.DMD(g,fs,L,snaps,r,sub)
        
        for i in range(len(b)):
            b[i] = np.sqrt(b.real[i]**2+b.imag[i]**2)
    
        b = pd.Series(b.real)
        mu = pd.Series((mu[:].real**2+mu[:].imag**2)**0.5)
        omega = pd.Series(omega)
        Sig = pd.Series(np.diag(Sig)[:])
        cont = np.zeros(len(Sig))
        contri = np.zeros(len(Sig))
        for i in range(len(Sig)):
            cont[i] = float(Sig[i]/sum(Sig)*100*b[i])
        for i in range(len(Sig)):
            contri[i] = float(Sig[i]/sum(Sig)*100)
        cont = pd.Series(cont)
        contri = pd.Series(contri)

        #making DataFrame
        total = pd.concat([omega,b,mu,cont],axis=1)
        total = pd.DataFrame(total)
        total.columns=['freq','b','eigen','contribution']
        total = total.sort_values(by="freq",ascending=True)
        return total
        
    
#parameter adjusting
#co = 

L = 2000
snaps = 20
r = 19
start = 0.0
end = 0.5

cut = int(r)/2



#    filename(input)
DMDA = Analysis()
g,fs = DMDA.Rawdata_loading('../Sounddata/bomb.wav',L)


#DMD窓の動かし方を決定する
#今回はsnapごとに動かしていく
#snapを大きくすればdtが大きくなる。
sub = np.arange(int(start*fs),int(end*fs),(L+snaps))



#dataframeをつくる
total = DMDA.main(g,fs,L,snaps,r,sub[0])
total.columns = ['freq',1,'eigen','contribution']
fmax = np.max(total['freq'])
freq = total['freq']
total = total[1]




j = 1
for n0 in sub[1:]:
    present = DMDA.main(g,fs,L,snaps,r,n0)
    present.columns=['freq',j+1,'eigen','contribution']
    if np.max(present['freq'])>fmax:
        fmax = np.max(present['freq'])
    present = present[j+1]
    total = pd.concat([total,present],axis=1)
    j+=1


totallog = total
totallog.index = freq
total = np.array(total)


x=np.linspace(start,end,len(total.T)+1)
y=np.array(freq)
y=y[:cut]
total = total[:cut,:]
x,y=np.meshgrid(x,y)
plt.pcolor(x,y,total)#,cmap='RdBu')
plt.xlabel("t",fontsize=16)
plt.ylabel("freq",fontsize=16)
plt.colorbar()
print("done")

#totallog.to_csv('../Sound/Hello2000.csv')




    