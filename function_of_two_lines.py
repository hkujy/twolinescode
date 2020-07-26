# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 23:46:46 2020
半成品

@author: dell
"""
import random
import numpy as np
import pandas as pd

#import sys
#sys.path.append(r'D:\【数据】公交IC卡数据')
#import a,b

def getdepart():
    departtable1 = []
    for i in np.arange(14):
        if i <= 3:
            departtable1.append(90+i*10)
        elif i <= 13:
            departtable1.append(120+(i-3)*15)
    departtable2 = []
    for i in np.arange(38):
        if i <= 3:
            departtable2.append(i*20)
        elif i <= 21:
            departtable2.append(60+(i-3)*10)
        elif i <= 31:
            departtable2.append(240+(i-21)*12)
        elif i <= 37:
            departtable2.append(360+(i-31)*10)
        departtable = [departtable1,departtable2]
    return departtable
departtable = getdepart()

#flag=0 小网络；flag=1 数值实验； flag=2 case
a_alight = 40                                                                  #下车（人/分钟）
b_board = 30                                                                   #上车（人/分钟）
# Cap = 200                                                                        #承载力（人）
ep = 0.1                                                                       #最小车头时距（分钟）

lam_all = 5
Am = 1
disbus = []
disstop = []
# if (flag == 0)|(flag == 1): disbus=[50],disstop=[0]
# elif (flag == 2): disbus=[4],disstop=[0]

#########                     车辆数          flag==2时需要departtable
def get_M(flag):
    if (flag == 0)|(flag==1):
        M_1 = 100
        M_2 = 100    
    elif flag == 2:
        M_1 = len(departtable[0])
        M_2 = len(departtable[1])
    return M_1,M_2


###########                  设置线路结构车站数
def get_OMEGA(flag):
    if flag == 1:
        Omega_1 = np.arange(1 , 6)
        Omega_2 = np.arange(6, 11)
        Omega_3 = np.arange(11, 16)
        Omega_4 = np.arange(16, 21)
        Omega_5 = np.arange(21, 26)    
    elif flag == 0:
        Omega_1 = np.arange(1 , 3)
        Omega_2 = np.arange(3, 5)
        Omega_3 = np.arange(5, 7)
        Omega_4 = np.arange(7, 9)
        Omega_5 = np.arange(9, 11)  
    elif flag == 2:
        Omega_1 = np.arange(1, 2)
        Omega_2 = np.arange(2, 9)
        Omega_3 = np.arange(9, 13)
        Omega_4 = np.arange(13, 20)
        Omega_5 = np.arange(20, 21)    
        # Omega_1 = np.arange(1 , 3)
        # Omega_2 = np.arange(3, 5)
        # Omega_3 = np.arange(5, 7)
        # Omega_4 = np.arange(7, 17)
        # Omega_5 = np.arange(17, 27)              
        # Omega_1 = np.arange(1 , 2)
        # Omega_2 = np.arange(2, 7)
        # Omega_3 = np.arange(7 , 8)
        # Omega_4 = np.arange(8, 9)
        # Omega_5 = np.arange(9 , 17)       
    return Omega_1,Omega_2,Omega_3,Omega_4,Omega_5


#################          给站台标号
def get_N(Omega_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N_1 = len(Omega_1)
    N_2 = len(Omega_2)
    N_3 = len(Omega_3)
    N_4 = len(Omega_4)
    N_5 = len(Omega_5)

    N1=N_1
    N2=N1+N_2
    N3=N2+N_3
    N4=N3+N_4
    N5=N4+N_5
    return N_1,N_2,N_3,N_4,N_5,N1,N2,N3,N4,N5

###################      组成线路
def get_S(Omega_list,flag):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    if (flag == 0) | (flag ==1) | (flag ==2):
        S_1 = list(np.append(np.append(Omega_1,Omega_3),Omega_4))
        S_2 = list(np.append(np.append(Omega_2,Omega_3),Omega_5))
    elif flag==3:
        S_1 = list(np.append(np.append(np.append(Omega_1,Omega_2),Omega_3),Omega_4))
        S_2 = list(np.append(np.append(Omega_1,Omega_3),Omega_5))
    return S_1,S_2


################     行驶时间（分钟）  有手动内容
def get_T(S_list,flag):
    S_1,S_2=S_list[0],S_list[1]
    if (flag == 0):
        T_1 = np.ones(len(S_1)-1) * 3
        T_2 = np.ones(len(S_2)-1) * 3
        T_n_1 = pd.DataFrame({'beta':S_1[1:],'T':T_1,'n':S_1[:-1]})  
        T_n_2 = pd.DataFrame({'beta':S_2[1:],'T':T_2,'n':S_2[:-1]})  
    elif (flag == 1):
        T_1 = np.ones(len(S_1)-1) * 6
        T_2 = np.ones(len(S_2)-1) * 6
        T_n_1 = pd.DataFrame({'beta':S_1[1:],'T':T_1,'n':S_1[:-1]})  
        T_n_2 = pd.DataFrame({'beta':S_2[1:],'T':T_2,'n':S_2[:-1]})  
    elif flag == 2:
        T_1_rtol=[2,1,2,1,4,5,1,1,1,2,2]
        T_2_rtol=[5,4,4,4,4,2,1,1,2,1,8]
        T_n_1 = pd.DataFrame({'beta':S_1[1:],'T':T_1_rtol,'n':S_1[:-1]})  
        T_n_2 = pd.DataFrame({'beta':S_2[1:],'T':T_2_rtol,'n':S_2[:-1]})      
    elif flag==3:
        T_1_rtol=[6,2,4,4,1,4,40]
        T_2_rtol=[20,7,5,2,4,2,7,7,45]
        T_n_1 = pd.DataFrame({'beta':S_1[1:],'T':T_1_rtol,'n':S_1[:-1]})  
        T_n_2 = pd.DataFrame({'beta':S_2[1:],'T':T_2_rtol,'n':S_2[:-1]})              
    return T_n_1,T_n_2


##############   生成对于行驶时间的随机扰动，设置成0的时候相当于行驶时间稳定不变                                                                
def randtime(Am1,Am2):
    T_ep = random.uniform(Am1, Am2)
    return T_ep

##########       两线路首车发车时刻表差（分钟），后续车辆的发车间隔，有手动内容   如果flag==2不需要用H
def get_H(flag): ##########flag==2时H不呢个乱用，用departtable代替
    if flag==1:
#        h_1 = 0
#        h_2 = 1.5
#        H_1 = 3
#        H_2 = 3     
        h_1 = 0
        h_2 = 3
        H_1 = 6
        H_2 = 6  
    elif flag==0:
        h_1 = 0
        h_2 = 3
        H_1 = 6
        H_2 = 6   
    elif flag==2:
        h_1 = departtable[0][0]   
        h_2 = departtable[1][0]
        H_1 = departtable[0][1]-h_1
        H_2 = departtable[1][1]-h_2
    elif flag==3:
        h_1 = 0
        h_2 = 0
        H_1 = 5
        H_2 = 7
    H_c = (1/H_1+1/H_2)**(-1)
    return h_1,h_2,H_1,H_2,H_c


#############  

def initial_some_nparray(M_list,N_list,S_list,flag):####何时用到该函数？返回值的不同
    M_1, M_2=M_list[0],M_list[1]
    N_1,N_3,N_4,N5 = N_list[0],N_list[2],N_list[3],N_list[9]
    S_1,S_2 = S_list[0],S_list[1]
    a_mn_compare=np.zeros([2,M_1+M_2,N5])        #arrival time
    d_mn_compare=np.zeros([2,M_1+M_2,N5])        #depart time
    xy_compare_1=np.zeros([2,len(S_1)*2,M_1])    #trajectories of line 1 (arrival time and depart time)
    xy_compare_2=np.zeros([2,len(S_2)*2,M_2])    #trajectories of line 2 (arrival time and depart time)
    w_mn_compare=np.zeros([2,M_1+M_2,N5])        #waiting time * waiting passengers
    B_a_mn_compare=np.zeros([2,M_1+M_2,N5])      #waiting passengers
    if (flag==0)|(flag==1):
        xy_compare = np.zeros([2,len(S_2)*2,M_1+M_2])
        xy_local_compare=np.zeros([2,(N_1+N_3+N_4)*2,M_1-50+M_2-50])    
        return a_mn_compare,d_mn_compare,xy_compare_1,xy_compare_2,xy_compare,xy_local_compare,w_mn_compare,B_a_mn_compare
    elif flag==2:
        return a_mn_compare,d_mn_compare,xy_compare_1,xy_compare_2,w_mn_compare,B_a_mn_compare

########## service bus fleet
def getBusFleetofCommonSet(n,a_mn,M_list):
    M_1,M_2=M_list[0],M_list[1]
    mf = np.array((np.arange(0,M_1+M_2),a_mn[:,n]))
    mf = mf[:,mf[1].argsort()]
    mf = np.array(mf[0],dtype=int)
    return mf

################# get mlast depart bus's depart time
def get_d_last(m,n,M,a_mn,d_mn,sorc,H_list,M_list):
    H_1,H_2,H_c=H_list[2],H_list[3],H_list[4]
    M_1=M_list[0]
    if sorc == 0:
        if m == M[0] :
            mlast = m
            d_last = a_mn[m,n]-H_c
        else:
            mlast =M[M.index(m)-1]
            d_last = d_mn[mlast,n]
    else:
        if m==0:
            mlast = m
            d_last = a_mn[m,n]-H_1
        elif m==M_1:
            mlast = m
            d_last = a_mn[m,n]-H_2 
        else:
            mlast = m-1
            d_last = d_mn[mlast,n]                                    
    return d_last,mlast

#score是判断是1的dedicated/2的dedicated/common, k是集合的编号
def SetofCommonStops(sorc,k,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    M_1, M_2=M_list[0],M_list[1]
    T_n_1,T_n_2=T_list[0],T_list[1]
    if k == 1:
        N = (Omega_1-1).tolist()
    elif k == 2:
        N = (Omega_2-1).tolist()
    elif k == 3:
        N = (Omega_3-1).tolist()
    elif k == 4:
        N = (Omega_4-1).tolist()[:-1]
    elif k == 5:
        N = (Omega_5-1).tolist()[:-1]

    for n in N:
        if sorc == 1:
            M = np.arange(M_1)
        elif sorc == 2:
            M = np.arange(M_1,M_1+M_2)
        elif sorc == 0:
            M = getBusFleetofCommonSet(n,a_mn,M_list).tolist()
        
        for m in M:
            d_last,mlast = get_d_last(m,n,M,a_mn,d_mn,sorc,H_list,M_list)
            
            if a_mn[m,n]<d_last+ep:
                a_mn[m,n] = d_last+ep
            
            if (flag==0)|(flag==1):
                Lam_nn = getlambda0(ratio,N_list,Omega_list)
            elif (flag==2):
                Lam_nn = getlambda1(a_mn[m,n],a,flag)

            lam_1 = getlam_1(n,Lam_nn,flag,Omega_list,N_list)
            lam_2 = getlam_2(n,Lam_nn,flag,Omega_list,N_list)
            lam_12 = getlam_12(n,Lam_nn,flag,Omega_list,N_list)

            inte_12 = a_mn[m,n] - d_last

            lastleft_1 = getleft_1(mlast,n,L_mnn,flag,Omega_list,N_list)
            lastleft_2 = getleft_2(mlast,n,L_mnn,flag,Omega_list,N_list)
            lastleft_12 = getleft_12(mlast,n,L_mnn,flag,Omega_list,N_list)

            if m < M_1:
                lam_r = lam_1
                left_r = lastleft_1
            elif m < M_1+M_2:
                lam_r = lam_2
                left_r = lastleft_2            
            
            Wb_mn[m,n]=(inte_12*(lam_r+lam_12)+left_r+lastleft_12)/(b_board-lam_r-lam_12)

            A_mn[m,n] = getA(m,n,p_mnn,flag,Omega_3,M_list,N_list)
            C_mn[m,n] = getCava(P_mn[m,n],A_mn[m,n],Cap)
            Wa_mn[m,n]=A_mn[m,n]/a_alight     
            Wc_mn[m,n]=C_mn[m,n]/b_board
       
            W_mn[m,n]=max(min(Wb_mn[m,n],Wc_mn[m,n]),Wa_mn[m,n])
            
            d_mn[m,n]=a_mn[m,n]+W_mn[m,n]
            
            n_next,Traveltime = getnextn(m,n,T_list,M_list)
            if (m in disbus) & (n in disstop):
                Traveltime=Traveltime + delay  
            Traveltime=Traveltime + randtime(0,Traveltime*0.5)*Am*delay       
            a_mn[m,n_next]=d_mn[m,n]+Traveltime
            
            # if (m in disbus) & (n in disstop):
            #     a_mn[m,n_next]=a_mn[m,n_next] + delay            
            # if (flag == 0)|(flag == 1):
            #     if (m in [50]) & (n in [0]):
            #         a_mn[m,n_next]=a_mn[m,n_next] + delay
            # elif (flag == 2):
            #     if (m in [4]) & (n in [0]):
            #         a_mn[m,n_next]=a_mn[m,n_next] + delay
   
            delta_12 = d_mn[m,n]-d_last

            B_mn[m,n]=delta_12*(lam_12+lam_r)+left_r+lastleft_12
            B_max_mn[m,n] = min(C_mn[m,n], B_mn[m,n])

            w_mn[m,n]=0.5*delta_12*(lam_r+lam_12)*delta_12+delta_12*(left_r+lastleft_12)

            p_mnn,P_mn,L_mnn = get_onleft(Lam_nn,m,n,mlast,n_next,B_mn[m,n],B_max_mn[m,n],p_mnn,P_mn,L_mnn,delta_12,flag,Omega_list,M_list,N_list)

def getlambda1(a_t,a,flag):
    if flag==2:##############建议以5点为0
        if a_t<60:
            Lam_nn = np.array(a[0])
        elif a_t<120:
            Lam_nn = np.array(a[1])
        elif a_t<180:
            Lam_nn = np.array(a[2])
        elif a_t<240:
            Lam_nn = np.array(a[3])
        elif a_t<300:
            Lam_nn = np.array(a[4])     
        elif a_t<360:
            Lam_nn = np.array(a[5])
        else:
            Lam_nn = np.array(a[6])      
    elif flag==3:       
        if a_t<60:
            Lam_nn = np.array(a[0])
        elif a_t<120:
            Lam_nn = np.array(a[1])
        elif a_t<180:
            Lam_nn = np.array(a[2])
        elif a_t<240:
            Lam_nn = np.array(a[3])
        elif a_t<300:
            Lam_nn = np.array(a[4])     
        elif a_t<360:
            Lam_nn = np.array(a[5])
        else:
            Lam_nn = np.array(a[6])              
    return Lam_nn

def getlambda0(ratio,N_list,Omega_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N_1,N_2,N_3,N_4,N_5,N4,N5=N_list[0],N_list[1],N_list[2],N_list[3],N_list[4],N_list[8],N_list[9]
    Lam_nn = np.zeros([N5,N5])
    lam_ij = np.nditer(Lam_nn, flags = ['multi_index'], op_flags = ['readwrite'])

    while not lam_ij.finished:                                                  #多换乘乘客
        i,j = lam_ij.multi_index[0]+1,lam_ij.multi_index[1]+1
        if i in Omega_1 and ((j in Omega_1 and i < j) or (j in Omega_3) or (j in Omega_4) or (j in Omega_5)):
            if (j in Omega_1 and i < j) or (j in Omega_3) or (j in Omega_4):
                lam_ij[0] = lam_all/(N_1 + N_3 + N_4 + ratio*N_5- i)
            else:
                lam_ij[0] = ratio*lam_all/(N_1 + N_3 + N_4 + ratio*N_5- i)
        elif i in Omega_2 and ((j in Omega_2 and i < j) or (j in Omega_3) or (j in Omega_4) or (j in Omega_5)):
            if (j in Omega_2 and i < j) or (j in Omega_3) or (j in Omega_5):
                lam_ij[0] = lam_all/(N_2 + N_3 + ratio*N_4 + N_5- (i-N_1))
            else:
                lam_ij[0] = ratio*lam_all/(N_2 + N_3 + ratio*N_4 + N_5- (i-N_1))
            
        elif i in Omega_3 and ((j in Omega_3 and i < j) or (j in Omega_4) or (j in Omega_5)):
            lam_ij[0] = lam_all/(N5-i)
        elif i in Omega_4 and j in Omega_4 and i < j:
            lam_ij[0] = lam_all/(N4-i)
        elif i in Omega_5 and j in Omega_5 and i < j:
            lam_ij[0] = lam_all/(N5-i)
        lam_ij.iternext()

    return Lam_nn

def getlam_1(n,Lam_nn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N1,N2,N3,N4=N_list[5],N_list[6],N_list[7],N_list[8]
    if (flag == 0) | (flag == 1)| (flag == 2):
        if n+1 in Omega_1:
            lam_1 = Lam_nn[n,n:].sum()
        elif n+1 in Omega_3:
            lam_1 = Lam_nn[n,N3:N4].sum()
        elif n+1 in Omega_4:
            lam_1 = Lam_nn[n,n:N4].sum()
        else:
            lam_1 = 0
    elif flag==3:
        if n+1 in Omega_1:
            lam_1 = Lam_nn[n,N1:N2].sum()+Lam_nn[n,N3:N4].sum()
        elif n+1 in Omega_2:
            lam_1 = Lam_nn[n,n:].sum()
        elif n+1 in Omega_3:
            lam_1 = Lam_nn[n,N3:N4].sum()
        else:
            lam_1 = 0

    return lam_1
    
def getlam_2(n,Lam_nn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N4,N5=N_list[8],N_list[9]
    if flag == 3:
        if n+1 in Omega_1:
            lam_2 = Lam_nn[n,N4:N5].sum()
        elif n+1 in Omega_3:
            lam_2 = Lam_nn[n,N4:N5].sum()
        elif n+1 in Omega_5:
            lam_2 = Lam_nn[n,n:N5].sum()
        else:
            lam_2 = 0
    elif (flag == 0) | (flag == 2)| (flag == 1):
        if n+1 in Omega_2:
            lam_2 = Lam_nn[n,n:].sum()
        elif n+1 in Omega_3:
            lam_2 = Lam_nn[n,N4:N5].sum()
        elif n+1 in Omega_5:
            lam_2 = Lam_nn[n,n:].sum()
        else:
            lam_2 = 0

    return lam_2
                
def getlam_12(n,Lam_nn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N2,N3=N_list[6],N_list[7]
    if flag == 3:
        if n+1 in Omega_1:
            lam_12 = Lam_nn[n,N2]
        else:
            lam_12 = 0
    elif (flag == 0) | (flag == 2)| (flag == 1):
        if n+1 in Omega_3:
            lam_12 = Lam_nn[n,n:N3].sum()
        else:
            lam_12 = 0
        
    return lam_12

def getleft_1(mlast,n,L_mnn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N1,N2,N3,N4,N5=N_list[5],N_list[6],N_list[7],N_list[8],N_list[9]
    if flag==3:
        if n+1 in Omega_1:
            lastleft_1 = L_mnn[mlast,n,N1:N2].sum()+L_mnn[mlast,n,N3:N4].sum()
        elif n+1 in Omega_2:
            lastleft_1 = L_mnn[mlast,n,n:].sum()
        elif n+1 in Omega_3:
            lastleft_1 = L_mnn[mlast,n,N3:N4].sum()
        else:
            lastleft_1 = 0
    
    elif (flag==0)|(flag==2)|(flag==1):
        if n+1 in Omega_1:
            lastleft_1 = L_mnn[mlast,n,n:].sum()
        elif n+1 in Omega_3:
            lastleft_1 = L_mnn[mlast,n,N3:N4].sum()
        elif n+1 in Omega_4:
            lastleft_1 = L_mnn[mlast,n,n:N4].sum()
        else:
            lastleft_1 = 0

    return lastleft_1

def getleft_2(mlast,n,L_mnn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N4,N5=N_list[8],N_list[9]
    if flag == 3:
        if n+1 in Omega_1:
            lastleft_2 = L_mnn[mlast,n,N4:N5].sum()
        elif n+1 in Omega_3:
            lastleft_2 = L_mnn[mlast,n,N4:N5].sum()
        elif n+1 in Omega_5:
            lastleft_2 = L_mnn[mlast,n,n:N5].sum()
        else:
            lastleft_2 = 0
    elif (flag == 0)|(flag==2)|(flag==1):
        if n+1 in Omega_2:
            lastleft_2 = L_mnn[mlast,n,n:].sum()
        elif n+1 in Omega_3:
            lastleft_2 = L_mnn[mlast,n,N4:N5].sum()
        elif n+1 in Omega_5:
            lastleft_2 = L_mnn[mlast,n,n:N5].sum()
        else:
            lastleft_2 = 0

    return lastleft_2

def getleft_12(mlast,n,L_mnn,flag,Omega_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    N2,N3=N_list[6],N_list[7]
    if flag == 3:
        if n+1 in Omega_1:
            lastleft_12 = L_mnn[mlast,n,N2]
        else:
            lastleft_12 = 0
    elif (flag == 0)|(flag==2)|(flag==1):
        if n+1 in Omega_3:
            lastleft_12 = L_mnn[mlast,n,n:N3].sum()
        else:
            lastleft_12 = 0

    return lastleft_12

def getA(m,n,p_mnn,flag,Omega_3,M_list,N_list):
    M_1,M_2=M_list[0],M_list[1]
    N3,N4,N5=N_list[7],N_list[8],N_list[9]
    if flag == 3:
        if n+1 in Omega_3:
            rou = get_rou(n,N3)
            if m < M_1:
                A = rou * p_mnn[m,n,N4:N5].sum() + p_mnn[m,n,n].sum()
            elif m < M_1+M_2:
                A = rou * p_mnn[m,n,N3:N4].sum() + p_mnn[m,n,n].sum()
        else:
            A = p_mnn[m,n,n].sum()
    
    elif (flag == 0)|(flag==2)|(flag==1):
        if n+1 in Omega_3:
            rou = get_rou(n,N3)
            if m < M_1:
                A = rou * p_mnn[m,n,N4:N5].sum() + p_mnn[m,n,n].sum()
            elif m < M_1+M_2:
                A = rou * p_mnn[m,n,N3:N4].sum() + p_mnn[m,n,n].sum()
            
        else:
            A = p_mnn[m,n,n]

    return A        

def get_rou(n,N3):
    rou = 1/(N3-n)
    return rou

def getCava(P,A,Cap):
    Cava = Cap - P + A
    return Cava

def getnextn(m,n,T_list,M_list):
    T_n_1,T_n_2=T_list[0],T_list[1]
    M_1,M_2=M_list[0],M_list[1]
    if m < M_1:
        nextn = int(T_n_1.loc[T_n_1['n']==n+1]['beta']-1)
        Traveltime = T_n_1.loc[T_n_1['n']==n+1]['T'].tolist()[0]
    elif m < M_1+M_2:
        nextn = int(T_n_2.loc[T_n_2['n']==n+1]['beta']-1)
        Traveltime = T_n_2.loc[T_n_2['n']==n+1]['T'].tolist()[0]
    return nextn,Traveltime

def get_onleft(Lam_nn,m,n,mlast,n_next,B,B_max,p_mnn,P_mn,L_mnn,delta_12,flag,Omega_list,M_list,N_list):
    Omega_1,Omega_2,Omega_3,Omega_4,Omega_5=Omega_list[0],Omega_list[1],Omega_list[2],Omega_list[3],Omega_list[4]
    M_1,M_2=M_list[0],M_list[1]
    N_1,N_3,N_4,N1,N2,N3,N4,N5 = N_list[0],N_list[2],N_list[3],N_list[5],N_list[6],N_list[7],N_list[8],N_list[9]
    if flag == 3:
        if n+1 in Omega_1:
            if B == 0:
                if m < M_1:        
                    p_mnn[m,n_next,N1:N4] = p_mnn[m,n,N1:N4]
                    L_mnn[m,n,N1:N4] = 0
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,N2:N3] = p_mnn[m,n,N2:N3]
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5]               
                    L_mnn[m,n,N2:N3] = 0
                    L_mnn[m,n,N4:N5] = 0
            else:
                if m < M_1:
                    p_mnn[m,n_next,N1:N4] = p_mnn[m,n,N1:N4] + (delta_12*Lam_nn[n,N1:N4] + L_mnn[mlast,n,N1:N4])*(B_max)/B
                    L_mnn[m,n,N1:N4] = (delta_12*Lam_nn[n,N1:N4] + L_mnn[mlast,n,N1:N4])*(B-B_max)/B
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,N2:N3] = p_mnn[m,n,N2:N3] + (delta_12*Lam_nn[n,N2:N3] + L_mnn[mlast,n,N2:N3])*(B_max)/B
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] + (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B_max)/B
                    L_mnn[m,n,N2:N3] = (delta_12*Lam_nn[n,N2:N3] + L_mnn[mlast,n,N2:N3])*(B-B_max)/B
                    L_mnn[m,n,N4:N5] = (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B-B_max)/B

        elif n+1 in Omega_2:
            if B == 0:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:]
                L_mnn[m,n,n_next:] = 0
            else:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:] + (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B_max)/B
                L_mnn[m,n,n_next:] = (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B-B_max)/B

        elif n+1 in Omega_3:
            if B==0:
                if m < M_1:
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4]
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] * (1-get_rou(n,N3))
                    L_mnn[m,n,N3:N4] = 0
                    L_mnn[m,n,N4:N5] = delta_12*Lam_nn[n,N4:N5] +L_mnn[mlast,n,N4:N5] + p_mnn[m,n,N4:N5] * get_rou(n,N3)
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4] * (1-get_rou(n,N3))
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5]
                    L_mnn[m,n,N3:N4] = delta_12*Lam_nn[n,N3:N4] +L_mnn[mlast,n,N3:N4] + p_mnn[m,n,N3:N4] * get_rou(n,N3)
                    L_mnn[m,n,N4:N5] = 0
            else:
                if m < M_1:
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4] + (delta_12*Lam_nn[n,N3:N4] + L_mnn[mlast,n,N3:N4])*(B_max)/B
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] * (1-get_rou(n,N3))
                    L_mnn[m,n,N3:N4] = (delta_12*Lam_nn[n,N3:N4] + L_mnn[mlast,n,N3:N4])*(B-B_max)/B
                    L_mnn[m,n,N4:N5] = delta_12*Lam_nn[n,N4:N5] +L_mnn[mlast,n,N4:N5] + p_mnn[m,n,N4:N5] * get_rou(n,N3)
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4] * (1-get_rou(n,N3))
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] + (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B_max)/B
                    L_mnn[m,n,N3:N4] = delta_12*Lam_nn[n,N3:N4] +L_mnn[mlast,n,N3:N4] + p_mnn[m,n,N3:N4] * get_rou(n,N3)
                    L_mnn[m,n,N4:N5] = (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B-B_max)/B
        
        elif n+1 in Omega_5:
            if B==0:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:]
                L_mnn[m,n,n_next:] = 0
            else:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:] + (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B_max)/B
                L_mnn[m,n,n_next:] = (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B-B_max)/B

    elif (flag == 0)|(flag==2)|(flag==1):
        if ((n+1 in Omega_1) | (n+1 in Omega_2) | (n+1 in Omega_4) | (n+1 in Omega_5)):
            if B==0:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:]
                L_mnn[m,n,n_next:] = 0
            else:
                p_mnn[m,n_next,n_next:] = p_mnn[m,n,n_next:] + (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B_max)/B
                L_mnn[m,n,n_next:] = (delta_12*Lam_nn[n,n_next:] + L_mnn[mlast,n,n_next:])*(B-B_max)/B      
        elif n+1 in Omega_3:
            if B==0:
                if m < M_1:
                    p_mnn[m,n_next,n_next:N4] = p_mnn[m,n,n_next:N4] 
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] * (1-get_rou(n,N3))
                    L_mnn[m,n,n_next:N4] = 0
                    L_mnn[m,n,N4:N5] = delta_12*Lam_nn[n,N4:N5] +L_mnn[mlast,n,N4:N5] + p_mnn[m,n,N4:N5] * get_rou(n,N3)
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,n_next:N3] = p_mnn[m,n,n_next:N3] 
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4] * (1-get_rou(n,N3))
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] 
                    L_mnn[m,n,n_next:N3] = 0
                    L_mnn[m,n,N3:N4] = delta_12*Lam_nn[n,N3:N4] +L_mnn[mlast,n,N3:N4] + p_mnn[m,n,N3:N4] * get_rou(n,N3)                    
                    L_mnn[m,n,N4:N5] = 0

            else:
                if m < M_1:
                    p_mnn[m,n_next,n_next:N4] = p_mnn[m,n,n_next:N4] + (delta_12*Lam_nn[n,n_next:N4] + L_mnn[mlast,n,n_next:N4])*(B_max)/B
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] * (1-get_rou(n,N3))
                    L_mnn[m,n,n_next:N4] = (delta_12*Lam_nn[n,n_next:N4] + L_mnn[mlast,n,n_next:N4])*(B-B_max)/B
                    L_mnn[m,n,N4:N5] = delta_12*Lam_nn[n,N4:N5] +L_mnn[mlast,n,N4:N5] + p_mnn[m,n,N4:N5] * get_rou(n,N3)
                elif m < M_1 + M_2:
                    p_mnn[m,n_next,n_next:N3] = p_mnn[m,n,n_next:N3] + (delta_12*Lam_nn[n,n_next:N3] + L_mnn[mlast,n,n_next:N3])*(B_max)/B
                    p_mnn[m,n_next,N3:N4] = p_mnn[m,n,N3:N4] * (1-get_rou(n,N3))
                    p_mnn[m,n_next,N4:N5] = p_mnn[m,n,N4:N5] + (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B_max)/B
                    L_mnn[m,n,n_next:N3] = (delta_12*Lam_nn[n,n_next:N3] + L_mnn[mlast,n,n_next:N3])*(B-B_max)/B
                    L_mnn[m,n,N3:N4] = delta_12*Lam_nn[n,N3:N4] +L_mnn[mlast,n,N3:N4] + p_mnn[m,n,N3:N4] * get_rou(n,N3)   
                    L_mnn[m,n,N4:N5] = (delta_12*Lam_nn[n,N4:N5] + L_mnn[mlast,n,N4:N5])*(B-B_max)/B                 
    p_mnn[m,n_next,:n_next] = 0
    P_mn[m,n_next] = p_mnn[m,n_next,n:].sum()

    return p_mnn,P_mn,L_mnn                                  

def initial_littlearray(M_list,N_list,flag,H_list):
    M_1,M_2=M_list[0],M_list[1]
    N_1,N_3,N_4,N1,N2,N3,N4,N5 = N_list[0],N_list[2],N_list[3],N_list[5],N_list[6],N_list[7],N_list[8],N_list[9]
    h_1,h_2,H_1,H_2,H_c=H_list[0],H_list[1],H_list[2],H_list[3],H_list[4]
    a_mn = np.zeros([M_1 + M_2,N5])
    Wb_mn = np.zeros([M_1 + M_2,N5])
    Wc_mn = np.zeros([M_1 + M_2,N5])
    Wa_mn = np.zeros([M_1 + M_2,N5])
    W_mn = np.zeros([M_1 + M_2,N5])
    d_mn = np.zeros([M_1 + M_2,N5])
    B_mn = np.zeros([M_1 + M_2,N5])
    B_max_mn = np.zeros([M_1 + M_2,N5])
    A_mn = np.zeros([M_1 + M_2,N5])
    C_mn = np.zeros([M_1 + M_2,N5])
    p_mnn = np.zeros([M_1 + M_2,N5,N5])
    P_mn = np.zeros([M_1 + M_2,N5])
    L_mnn = np.zeros([M_1 + M_2,N5,N5])
    w_mn = np.zeros([M_1 + M_2,N5])  

    if (flag == 0)|(flag==1):
        a_mn[0:M_1,0] = np.arange(0,M_1) * H_1 + np.ones(M_2) * h_1
        a_mn[M_1:M_1+M_2,N_1] = np.arange(0,M_2) * H_2 + np.ones(M_2) * h_2
    elif flag == 2:
        a_mn[0:M_1,0] = departtable[0]
        a_mn[M_1:M_1+M_2,N_1] = departtable[1]
    return a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,w_mn

def gettrajectories(flag,M_list,N_list,a_mn,d_mn):
    M_1,M_2=M_list[0],M_list[1]
    N_1,N_2,N_3,N_4,N_5,N1,N2,N3,N4,N5 = N_list[0],N_list[1],N_list[2],N_list[3],N_list[4],N_list[5],N_list[6],N_list[7],N_list[8],N_list[9]
    if (flag==0)|(flag==2)|(flag==1):
        xy_1 = np.zeros([M_1,2*(N_1+N_3+N_4)])
        #1号线每辆车在每个站的到达时间
        xy_1[:,:N_1]=a_mn[:M_1,:N1]                                            
        xy_1[:,N_1:N_1+N_3+N_4]=a_mn[:M_1,N2:N4]
        #1号线每辆车在每个站的离开时间
        xy_1[:,N_1+N_3+N_4:N_1+N_3+N_4+N_1]=d_mn[:M_1,:N1]
        xy_1[:,N_1+N_3+N_4+N_1:2*(N_1+N_3+N_4)]=d_mn[:M_1,N2:N4]
    
        xy_2 =np.zeros([M_2,2*(N_2+N_3+N_5)])
        #2号线每辆车在每个站的到达时间
        xy_2[:,:N_2+N_3]=a_mn[M_1:,N1:N3]
        xy_2[:,N_2+N_3:N_2+N_3+N_5]=a_mn[M_1:,N4:N5]
        #2号线每辆车在每个站的离开时间
        xy_2[:,N_1+N_3+N_4:N_1+N_3+N_4+N_2+N_3]=d_mn[M_1:,N1:N3]
        xy_2[:,N_1+N_3+N_4+N_2+N_3:2*(N_2+N_3+N_5)]=d_mn[M_1:,N4:N5]
    
        #转置，行是站，列是车
        xy_1=xy_1.transpose() 
        xy_2=xy_2.transpose()
    
        #从小到达排序
        np.arange(N_1+N_3+N_4)
        xy_1 = xy_1[np.hstack((np.arange(0,(N_1+N_3+N_4)*2,2), np.arange(1,(N_1+N_3+N_4)*2+1,2))).argsort()]
        xy_2 = xy_2[np.hstack((np.arange(0,(N_2+N_3+N_5)*2,2), np.arange(1,(N_2+N_3+N_5)*2+1,2))).argsort()]

        if (flag == 0)|(flag==1):
            xy = np.zeros([(N_1+N_3+N_4)*2,M_1+M_2])#注意这样画图只能在两线路总站台数相同时完成
            xy[:,:M_1]=xy_1
            xy[:,M_1:]=xy_2
            return xy,xy_1,xy_2
            #去掉初始化
            xy_local=np.zeros([(N_1+N_3+N_4)*2,M_1-50+M_2-50])
            xy_local[:,:M_1-50]=xy_1[:,50:]
            xy_local[:,M_1-50:]=xy_2[:,50:]
            
        elif flag==2:
            return xy_1,xy_2 


def getrealarrivaltime(a_mn_compare,M_list,N_list,T_list,d_mn_compare):
    M_1,M_2 = M_list[0], M_list[1]
    N1,N2,N3,N4,N5 = N_list[5],N_list[6],N_list[7],N_list[8],N_list[9]
    a_mn_real_compare = np.zeros([2,M_1+M_2,N5])
    T_mn = np.zeros([M_1+M_2,N5])
    for n in np.arange(0,T_mn.shape[1]-1):
        if (n < N1):
            for m in np.arange(M_1):
                nextn,Traveltime = getnextn(m,n,T_list,M_list)
                T_mn[m,nextn] = Traveltime
                a_mn_real_compare[0,m,nextn]=d_mn_compare[0,m,n]+Traveltime
                a_mn_real_compare[1,m,nextn]=d_mn_compare[1,m,n]+Traveltime                
        elif n < N2:
            for m in np.arange(M_1,M_1+M_2):
                nextn,Traveltime = getnextn(m,n,T_list,M_list)
                T_mn[m,nextn] = Traveltime
                a_mn_real_compare[0,m,nextn]=d_mn_compare[0,m,n]+Traveltime
                a_mn_real_compare[1,m,nextn]=d_mn_compare[1,m,n]+Traveltime  
        elif n < N3:
            for m in np.arange(M_1+M_2):
                nextn,Traveltime = getnextn(m,n,T_list,M_list)
                T_mn[m,nextn] = Traveltime               
                a_mn_real_compare[0,m,nextn]=d_mn_compare[0,m,n]+Traveltime
                a_mn_real_compare[1,m,nextn]=d_mn_compare[1,m,n]+Traveltime  
        elif n < N4-1:
            for m in np.arange(M_1):
                nextn,Traveltime = getnextn(m,n,T_list,M_list)
                T_mn[m,nextn] = Traveltime   
                a_mn_real_compare[0,m,nextn]=d_mn_compare[0,m,n]+Traveltime
                a_mn_real_compare[1,m,nextn]=d_mn_compare[1,m,n]+Traveltime  
        elif (n < N5-1) & (n > N4-1):
            for m in np.arange(M_1,M_1+M_2):
                nextn,Traveltime = getnextn(m,n,T_list,M_list)
                T_mn[m,nextn] = Traveltime
                a_mn_real_compare[0,m,nextn]=d_mn_compare[0,m,n]+Traveltime
                a_mn_real_compare[1,m,nextn]=d_mn_compare[1,m,n]+Traveltime  
    a_mn_real_compare[0,:,0] = a_mn_compare[0,:,0]
    a_mn_real_compare[0,:,N1] = a_mn_compare[0,:,N1]
    a_mn_real_compare[1,:,0] = a_mn_compare[1,:,0]
    a_mn_real_compare[1,:,N1] = a_mn_compare[1,:,N1]
    return a_mn_real_compare
