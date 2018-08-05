#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:43 2017

@author: Dohi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import dot,multiply,power
from numpy.linalg import inv,pinv
from scipy.linalg import svd,eig


class Analysis:
    def DMD(self,Data,fs,r,j):
        dt = 1/fs
        X,Y= Data[:,:-1],Data[:,1:]
        U,Sig,Vh = svd(X, False)
        U,Sig,V = U[:,:r],np.diag(Sig)[:r,:r],Vh.conj().T[:,:r]
        # freq
        mu,W = eig(dot(dot(dot(U.conj().T, Y), V), inv(Sig)))
        freq = abs((np.log(mu)/(2.0*np.pi*1.0/fs)).imag).real
        eta  = np.log(abs(mu))*fs
        #Psi
        t = np.linspace(0,1.0/fs*len(Data.T),len(Data.T))
        Phi = dot(dot(dot(Y, V), inv(Sig)), W)
        b = dot(pinv(Phi), X[:,0])
        Psi = np.zeros([r, len(t)], dtype='complex')
        for i,_t in enumerate(t):
            Psi[:,i] = multiply(power(mu, _t/dt), b)
        #Psiplot
        
        #if j<1:
            #plt.figure(),plt.xlabel("Time[s]",fontsize=16),plt.ylabel("State",fontsize=16)
#            for i in range(r):
#                if i==0 or i==1 or i==3 or i==5:
#                    plt.plot(t,Psi[i,:],label='Mode_'+str(i+1),linestyle='solid')
#                if i==2 or i==4:
#                    plt.plot(t,Psi[i,:],label='Mode_'+str(i+1),linestyle='dotted')
#            plt.plot(t,Psi[0,:],label='Mode_'+str(1),linestyle='solid')
#            plt.plot(t,Psi[1,:],label='Mode_'+str(2),linestyle='solid',color='navy')
#            plt.plot(t,Psi[2,:],label='Mode_'+str(3),linestyle='dashed',color='gold')
#            plt.plot(t,Psi[3,:],label='Mode_'+str(4),linestyle='solid',color='lawngreen')
#            plt.plot(t,Psi[4,:],label='Mode_'+str(5),linestyle='dashed',color='crimson')
#            plt.plot(t,Psi[5,:],label='Mode_'+str(6),linestyle='solid',color='tan')

            #plt.plot(t,Psi[0,:],label='Mode_'+str(1),linestyle='solid')
            #plt.plot(t,Psi[1,:],label='Mode_'+str(2),linestyle='dashed',color='orange')
            #plt.plot(t,Psi[2,:],label='Mode_'+str(3),linestyle='solid',color='navy')
            #plt.plot(t,Psi[3,:],label='Mode_'+str(4),linestyle='dashed',color='gold')
            #plt.plot(t,Psi[4,:],label='Mode_'+str(5),linestyle='solid',color='lawngreen')
            #plt.plot(t,Psi[5,:],label='Mode_'+str(6),linestyle='dashed',color='crimson')
            
            #plt.legend(bbox_to_anchor=(1.0,1.02), ncol=1)
        return freq,mu,b,Sig,Phi,eta

    def main(self,Data,fs,r,mx,my,f_list,j,jlist):
        freq,mu,b,Sig,Phi,eta = self.DMD(Data,fs,r,j)
        Sig,b,mu,eta = np.diag(Sig),abs(b).real,abs(mu).real,eta.real
        contri=Sig/sum(Sig)*100
        total = pd.DataFrame(np.vstack([freq,b,mu,contri,eta]).T,columns=['freq','b','eig','contribution','eta']).sort_values(by='freq',ascending=True)     
#        print(total['eta'][0])
        #jlist.append(total['eta'][0])
        if total['freq'][0]==0:
            j+=1
            jlist.append(abs(total['eta'][0]))
#            print("NO!")
        
#        print(total)
        freq,eig,contri,b,eta = np.array(total['freq']),np.array(total['eig']),np.array(total['contribution']),np.array(total['b']),np.array(total['eta'])
        x,y=np.meshgrid(np.linspace(0,mx,mx+1),np.linspace(0,my,my+1))
        #see above(eig,contri,b,freq,Phi[:,0])
        """
        plt.figure(),plt.pcolor(x,y,abs(Phi[:,0]).real.reshape(my,mx),cmap='gray'),plt.tick_params(labelbottom='off',labelleft='off'),plt.gca().yaxis.set_ticks_position('none'),plt.gca().xaxis.set_ticks_position('none'),plt.colorbar() 
        for i in range(mx):
            for j in range(my):
                x = 1.5/mx+i
                y = 1/my+j
                plt.annotate(int(f_list[mx*j+i]),xy=(x,y),size=10) """
#        print(Data.shape)
        return total,j,jlist


#    filename(input)
DMDA = Analysis()
filename = '../Sounddata2/normal_1200rpm_1'
filename1 = filename+'.csv'
#filename = '/Users/Dohi/Desktop/DMD/DMDprogram/Sounddata/bomb.csv'
Datam= np.array(pd.read_csv(filename1,index_col=0))
print(Datam.shape) 

filename2 = filename+'flist.csv'
f_list= (np.array(pd.read_csv(filename2,index_col=0)))[:len(Datam)]



#パラメータ(T:窓幅、T>rこれはTの間隔でデータを切り出してDMDを適用している。）
T = 10
d = len(Datam.T)/T
print(d)
r = int(len(Datam.T)/d-1)
mx = 8
L = len(Datam)

my = int(len(Datam)/mx)
fs = 44100/1024




#実行
#LでDMD窓幅を変更
j=0
jlist = []
sub = np.linspace(0,len(Datam.T),int(d+1))

if len(sub)>2:
    sub1,sub2 = sub[0:-1],sub[1:]
    for (n0,n1) in zip(sub1,sub2):
        n0 = int(n0)
        n1 = int(n1)
        total,j,jlist = DMDA.main(Datam[:,n0:n1],fs,r,mx,my,f_list,j,jlist)
else:
    total,j,jlist = DMDA.main(Datam[:,:sub[1]],fs,r,mx,my,f_list,j,jlist)


print("時間")
print(np.round(1/fs*len(Datam.T)/d,2),"s")
j=j/len(sub)
print("全体のサンプル数:",len(sub))
print("減衰成分が支配的な割合")
print(np.round(j*100,2),"%")
    

    
    
#異常音の方が明らかに減衰率大のモードの寄与が大きい。


print("減衰率の平均")
print(np.average(jlist))
print("減衰率の分散")
print(np.std(jlist))
#plt.figure()
#plt.hist(jlist,bins=50,range=(0,5),rwidth=0.8)
