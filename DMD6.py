#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:52:16 2017

@author: Dohi
"""

#Perform DMD on STFTdata 

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
        #cal Phi
        t = np.linspace(0,dt*len(Data.T),len(Data.T))
        Phi = dot(dot(dot(Y, V), inv(Sig)), Wr)
        b = dot(pinv(Phi), X[:,0])
#        b = np.diag(dot(Phi.conj().T,Phi))
        Psi = np.zeros([r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        plt.figure()
        for i in range(r):
            plt.plot(t,Psi[i,:],label="Mode_"+str(i+1))
        plt.legend(loc="upper center", bbox_to_anchor=(1.11,1.03), ncol=1)
        mx = 5
        my = 6
        
        plt.figure()
        Mode=np.zeros([my,mx])
        for i in range(0,my):
            for j in range(0,mx):
                Mode[i,j]=abs(Phi[j+i*mx,0])
        Z=np.arange(mx*my).reshape((my,mx))
        Z=Mode
        x=np.linspace(-mx/2,mx/2,int(mx)+1)
        y=np.linspace(-my/2,my/2,int(my)+1)
        x,y=np.meshgrid(x,y)
        plt.pcolor(x,y,Z,cmap='RdBu')
        plt.colorbar()
        return omega,mu,b,Sig,Data,Phi
    
    

    #Normal DMD
    def Data_DMD(self,Data,sub,snaps,r,width):

        fs =  44100/1024
        dt = 1.0/fs

        omega,mu,b,Sig,Data,Phi = self.DMD(Data,dt,r)
        for i in range(len(b)):
            b[i] = np.sqrt(b.real[i]**2+b.imag[i]**2)
#        b = np.diag(dot(Phi.conj().T,Phi))
        b = pd.Series(b)
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


Mode = 1800
sub = 0
snaps = 10
r = 9
width = 200

filename = '../STFTdata/clack_'+str(Mode)+'rpm_1.csv'
#filename = '/Users/Dohi/Desktop/DMD/DMDprogram/Sounddata/bomb.csv'
fftdata = pd.read_csv(filename,index_col=0)
print(filename)
Data = np.array(fftdata)
print(Data.shape)
Data = Data[:,:snaps]
print(Data.shape)
DMDA = Analysis()
total= DMDA.Data_DMD(Data,sub,snaps,r,width)


    
