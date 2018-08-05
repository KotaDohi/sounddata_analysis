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
        """
        Psi = np.zeros([r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        
        
        plt.plot(t,Psi[94,:].real)
        print(Psi[42,:])
        print(Psi[42,:].real**2+Psi[42,:].imag**2)
        print(mu[42])
        print(b[42])
        plt.figure()
        for i in range(r):
            plt.plot(t,Psi[i,:])"""
        
        """
        mu1 = diag(mu)
        B = dot(dot(Y,V),inv(Sig))
        A = dot(B,U.conj().T)
        vl = dot(dot(mu,Wl),pinv(B))
        vr = dot(dot(B,Wr),inv(mu))
        plt.figure()
        plt.scatter(mu[:].real,mu[:].imag,label=name)
        plt.legend(loc="upper right")"""
        return omega,mu,b,Sig,Data,Phi
    
    
    
    
    
    #Normal DMD
    
    
    def Data_DMD(self,filename,sub,L,snaps,r):
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
        """return omega,mu,b,Sig,Data,x,Phi,wav
    
    
    
    
    


    
    
    def Dataframes(self):
        #prmameter processing
        omega,mu,b,Sig,Data,x,Phi,wav = self.Data_DMD(self,filename+".wav")
        """
        
        for i in range(len(b)):
            b[i] = max(abs(b.real[i]),abs(b.imag[i]))
    
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
        width = 200
        points = np.arange(0,1400,width)
        con = []
        if filename[8] == 'n':
            con.append(0)
        else:
            con.append(1)
            
        for i in points:
            totalcut = total[total['freq']>=i]
            totalcut = totalcut[totalcut['freq']<=(i+width)]
            con.append(sum(totalcut['contribution']))
        
    
    
    
    
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
        
                  
        """
        #making wave data
        Sound = np.zeros(len(Data[:,0]))
        
        erase = 0
        
        Phi_new = np.zeros([len(Phi),len(Phi.T)],dtype=np.complex)
        
        if erase==0:
            start = 0
            to = 0
            Phi_new = Phi
            Srate = 44100
            Sound = dot(Phi_new,b)
            wavio.write("audio2/"+filename+'_from'+str(start)+'_to'+str(to)+".wav",Sound,Srate,sampwidth=3)
            Phi_new = np.zeros([len(Phi),len(Phi.T)],dtype=np.complex)
        
        
        if erase==1:
            for i in range(len(b)-1):
                start = i
                to = i+1
                survive = np.arange(start,to+1)
            
            
                for j in survive:
                    Phi_new[:,j] = Phi[:,(total[j:(j+1)]["index"])]
                freq = int(total[i:(i+1)]["freq"])
        #        print(total[i:(i+1)]["index"]) 
            
            
                Sound = dot(Phi_new,b)
                Srate = 44100
                wavio.write("audio2/normal_1800rpm_2_1000/"+filename+'_from'+str(start)+'_to'+str(to)+'_freq'+str(freq)+".wav",Sound,Srate,sampwidth=3)
                Phi_new = np.zeros([len(Phi),len(Phi.T)],dtype=np.complex)"""
        
        return con
        
#parameter adjusting
L = 10000
snaps = 500
r = 100
Mode = 1800

#    filename(input)
DMDA = Analysis()



sub = np.arange(0,1300000,L)
lsub =len(sub)
Con = np.zeros([4*lsub,8])
j = 0
for i in sub:
    con1 = DMDA.Data_DMD('../data/clack_1800rpm_1.wav',i,L,snaps,r)
    con2 = DMDA.Data_DMD('../data/clack_1800rpm_2.wav',i,L,snaps,r)
    con3 = DMDA.Data_DMD('../data/normal_1800rpm_1.wav',i,L,snaps,r)
    con4 = DMDA.Data_DMD('../data/normal_1800rpm_2.wav',i,L,snaps,r)
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"train")
    
Con = pd.DataFrame(Con)
Con.to_csv('../learnings/train'+str(Mode)+'2.csv')




sub = np.arange(1300001,2600000,L)
lsub =len(sub)
Con = np.zeros([4*lsub,8])
j = 0
for i in sub:
    con1 = DMDA.Data_DMD('../data/clack_1800rpm+refinery_1.wav',i,L,snaps,r)
    con2 = DMDA.Data_DMD('../data/clack_1800rpm+refinery_2.wav',i,L,snaps,r)
    con3 = DMDA.Data_DMD('../data/normal_1800rpm+refinery_1.wav',i,L,snaps,r)
    con4 = DMDA.Data_DMD('../data/normal_1800rpm+refinery_2.wav',i,L,snaps,r)
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"test")


Con = pd.DataFrame(Con)
Con.to_csv('../learnings/test'+str(Mode)+'2.csv')


#refineryがうまくいかないときはrefineryの部分を捨てたらいいのかなー
#すなわち、とるrankの範囲を帰るとか
#もしかしたらrefineryはbでやるといいかもよ
#refinery1は1どうし、2は2どうしで考えたらいいかもしれない。
#再構成誤差を作り出すためのシステム行列の決め方を考えておいたほうがいいかも

#他の部分のノイズのデータを使って異常を検知するというよりは、
#ノイズの成分をきちんと割り出して考えられたほうがいいよね


    
