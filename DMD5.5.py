#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:23:48 2017

@author: Dohi
"""

        #Introduction to RF
        points = np.arange(0,1400,width)
        con = []
        if filename[8] == 'n':
            con.append(0)
        else:
            con.append(1)
            
        for i in points:
            totalcut = total[total['freq']>=i]
            totalcut = totalcut[totalcut['freq']<=(i+width)]
            con.append(sum(totalcut['b']))
            
sub = np.arange(0,1300000,L)
lsub =len(sub)
Con = np.zeros([4*lsub,int(1400/width+1)])
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
    print(j,"train")
    
Con = pd.DataFrame(Con)
Con.to_csv('../learnings/train'+str(Mode)+'+refinery_b_100.csv')




sub = np.arange(1300001,2600000,L)
lsub =len(sub)
Con = np.zeros([4*lsub,int(1400/width+1)])
j = 0
for i in sub:
    con1 = DMDA.Data_DMD('../data/clack_1800rpm+refinery_1.wav',i,L,snaps,r,width)
    con2 = DMDA.Data_DMD('../data/clack_1800rpm+refinery_2.wav',i,L,snaps,r,width)
    con3 = DMDA.Data_DMD('../data/normal_1800rpm+refinery_1.wav',i,L,snaps,r,width)
    con4 = DMDA.Data_DMD('../data/normal_1800rpm+refinery_2.wav',i,L,snaps,r,width)
    Con[j,:] = con1
    Con[j+lsub,:] = con2
    Con[j+2*lsub,:] = con3
    Con[j+3*lsub,:] = con4
    j+=1
    print(j,"test")


Con = pd.DataFrame(Con)
Con.to_csv('../learnings/test'+str(Mode)+'+refinery_b_100.csv')


#refineryがうまくいかないときはrefineryの部分を捨てたらいいのかなー
#すなわち、とるrankの範囲を帰るとか
#もしかしたらrefineryはbでやるといいかもよ
#refinery1は1どうし、2は2どうしで考えたらいいかもしれない。
#再構成誤差を作り出すためのシステム行列の決め方を考えておいたほうがいいかも

#他の部分のノイズのデータを使って異常を検知するというよりは、
#ノイズの成分をきちんと割り出して考えられたほうがいいよね

#パラメータはランクの数、freqのwidth、bかcontributionか、ぐらい