#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:52:16 2017

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
    
    def _init_(self):
        self.filename = ""

    def DMD(self,Data,dt,r):
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
            omega[i] = abs((np.log(mu[i])/(2.0*np.pi*dt)).imag)
        
    
    #    modes = np.arange(r/2)
    #    plt.figure()
    #    plt.plot(modes,omega)
        
        
        #cal Phi
    #    t = np.linspace(0,dt*len(Data.T),len(Data.T))
        Phi = dot(dot(dot(Y, V), inv(Sig)), Wr)
        b = dot(pinv(Phi), X[:,0])
        return omega,mu,b,Sig,Data,Phi
    
    
    
    
    
    #Normal DMD
    
    
    def Data_DMD(self,filename,L,snaps,r):
        wav = wavio.read(filename)
        bit = 8 * wav.sampwidth
        data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
        x = data[:,0]
        fs = wav.rate
        

        merge = int(44100/8000)
        
        dt = 1.0/fs*merge
        
        if merge !=1:
            x = x[::merge]
        Data = np.zeros((L,snaps))
        
        for i in range(snaps):
            Data[:,i] = x[sub+i:sub+L+i]
        
        omega,mu,b,Sig,Data,Phi = self.DMD(Data,dt,r)
        
        for i in range(len(b)):
            b[i] = np.sqrt(b.real[i]**2+b.imag[i]**2)
    
        b = pd.Series(b.real)
        mu = pd.Series((mu[:].real**2+mu[:].imag**2)**0.5)
        omega = pd.Series(omega)
        Sig = pd.Series(np.diag(Sig)[:])
        cont = np.zeros(len(Sig))
        for i in range(len(Sig)):
            cont[i] = float(Sig[i]/sum(Sig))*100
        cont = pd.Series(cont)
        index = pd.Series(np.arange(0,len(b)))
    
        
    
        #making DataFrame
        total = pd.concat([index,mu,omega,b,cont],axis=1)
        total = pd.DataFrame(total)
        total.columns=["index","eigen","freq","b","contribution"]
        total = total.sort_values(by="freq",ascending=True)
        return total
        

#parameter adjusting
L = 10000
snaps = 500
r = 100
Mode = 1800
width = 100
sub = np.arange(0,1300000,L)


#filename(input)
DMDA = Analysis()
total = DMDA.Data_DMD('../data/clack_1800rpm+refinery_1.wav',L,snaps,r)
total.to_csv('../DMDperformed/clack_1800rpm+refinery')





    
