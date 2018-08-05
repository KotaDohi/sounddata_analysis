#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:52:16 2017

@author: Dohi
"""

#最新バージョンです(11/4)
#merge=5にしてあります

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
        
    def Data_loaming(self,filename):
        wav = wavio.read(filename)
        bit = 8 * wav.sampwidth
        data = wav.data / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
        x = data[:,0]
        fs = wav.rate
        
#        merge = int(44100/8000)
        merge=1
        dt = 1.0/fs*merge
        
        if merge !=1:
            x = x[::merge]
        return x,dt
        

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
        
    
        Phi = dot(dot(dot(Y, V), inv(Sig)), Wr)
        b = dot(pinv(Phi), X[:,0])
        return omega,mu,b,Sig,Data,Phi
    
    
    
    
    
    #Normal DMD
    
    
    def Data_DMD(self,sub,L,snaps,r,width,x,dt,filename):
        Data = np.zeros((L,snaps))
        for i in range(snaps):
            Data[:,i] = x[sub+i:sub+L+i]
        
        omega,mu,b,Sig,Data,Phi = self.DMD(Data,dt,r)
        
        for i in range(len(b)):
            b[i] = abs(b[i])*2.0/np.sqrt(snaps)
    
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
        #print(total)
        
        
        
        #Introduction to RF
        points = np.arange(0,22000,width)
        con = []
        if filename[8] == 'n':
            con.append(0)
        else:
            con.append(1)
            
        for i in points:
            totalcut = total[total['freq']>=i]
            totalcut = totalcut[totalcut['freq']<=(i+width)]
#            con.append(sum(totalcut['b']))
            
            if len(totalcut['eigen'])>0:
                con.append(sum(totalcut['eigen'])/len(totalcut['eigen']))
            else:
                con.append(sum (totalcut['eigen']))
        
    
    
    
    
        #make figures
        """
        plt.figure(figsize=(18,4))
        plt.title(filename)
        plt.subplots_adjust(wspace=0.4,hspace=0.6)
                   
        plt.subplot(1,3,1)
        plt.scatter(omega,b,s=10,c='r')
        plt.title("freq-b")
        plt.ylim(0,0.1)
        
        plt.subplot(1,3,2)
        plt.scatter(omega,cont,s=10,c='r')
        plt.title("freq-contribution")
        #plt.ylim(0,10)
        
        plt.subplot(1,3,3)
        plt.scatter(b,cont,s=10,c='r')
        plt.title("b-contribution")
        plt.ylim(0,10)
        
        plt.figure()
        plt.title(filename,fontsize=16)
        plt.scatter(omega,cont,s=10,c='r')
        #plt.ylim(0,15)
        plt.xlabel("freq[Hz]",fontsize=14)
        plt.ylabel("contribution[%]",fontsize=14)"""
    
        return con
        
#parameter adjusting
L = 300
snaps = 300
r = 299
Mode = 600
width = 500

#    filename(input)
#ファイルの読みこみ
DMDA = Analysis()
x1,dt1 = DMDA.Data_loaming('../data/clack_'+str(Mode)+'rpm_1.wav')
x2,dt2 = DMDA.Data_loaming('../data/clack_'+str(Mode)+'rpm_2.wav')
x3,dt3 = DMDA.Data_loaming('../data/normal_'+str(Mode)+'rpm_1.wav')
x4,dt4 = DMDA.Data_loaming('../data/normal_'+str(Mode)+'rpm_2.wav')


ldata = 2600000*0.03  #length of the sound data
coef = 0.6       #ratio of trainning data

sub = np.arange(0,ldata*0.6,L)
lsub =len(sub)
Con = np.zeros([4*lsub,int(22000/width+1)])
j = 0

for i in sub:
    con1 = DMDA.Data_DMD(i,L,snaps,r,width,x1,dt1,'../data/clack_'+str(Mode)+'rpm_1.wav')
    con2 = DMDA.Data_DMD(i,L,snaps,r,width,x2,dt2,'../data/clack_'+str(Mode)+'rpm_2.wav')
    con3 = DMDA.Data_DMD(i,L,snaps,r,width,x3,dt3,'../data/normal_'+str(Mode)+'rpm_1.wav')
    con4 = DMDA.Data_DMD(i,L,snaps,r,width,x4,dt4,'../data/normal_'+str(Mode)+'rpm_2.wav')
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"train")
    
Con = pd.DataFrame(Con)
Con.to_csv('../learning/train'+str(Mode)+'_500_mu_300_300.csv')




sub = np.arange(ldata*0.6+1,ldata,L)
lsub = len(sub)
Con = np.zeros([4*lsub,int(22000/width+1)])
j = 0
for i in sub:
    con1 = DMDA.Data_DMD(i,L,snaps,r,width,x1,dt1,'../data/clack_'+str(Mode)+'rpm_1.wav')
    con2 = DMDA.Data_DMD(i,L,snaps,r,width,x2,dt2,'../data/clack_'+str(Mode)+'rpm_2.wav')
    con3 = DMDA.Data_DMD(i,L,snaps,r,width,x3,dt3,'../data/normal_'+str(Mode)+'rpm_1.wav')
    con4 = DMDA.Data_DMD(i,L,snaps,r,width,x4,dt4,'../data/normal_'+str(Mode)+'rpm_2.wav')
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"test")


Con = pd.DataFrame(Con)
Con.to_csv('../learning/test'+str(Mode)+'_500_mu_300_300.csv')


#refineryがうまくいかないときはrefineryの部分を捨てたらいいのかなー
#すなわち、とるrankの範囲を帰るとか
#もしかしたらrefineryはbでやるといいかもよ
#refinery1は1どうし、2は2どうしで考えたらいいかもしれない。
#再構成誤差を作り出すためのシステム行列の決め方を考えておいたほうがいいかも

#他の部分のノイズのデータを使って異常を検知するというよりは、
#ノイズの成分をきちんと割り出して考えられたほうがいいよね

#パラメータはランクの数、freqのwidth、bかcontributionか、ぐらい


    
