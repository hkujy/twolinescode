
#import random
import numpy as np
import pandas as pd

##################主程序

from function_of_two_bunching import (get_M,get_OMEGA,get_N,get_S,get_T,get_H,
initial_some_nparray,getBusFleetofCommonSet,get_d_last,SetofCommonStops,
getlambda1,getlambda0,getlam_1,getlam_2,getlam_12,getleft_1,getleft_2,getleft_12,
getA,get_rou,getCava,getnextn,get_onleft,initial_littlearray,gettrajectories,getrealarrivaltime)

def area(a_mn_compare,d_mn_compare):
    a_area=a_mn_compare[1]-a_mn_compare[0]
    a_area[a_area>0.016]=1
    a_area[a_area<-0.016]=1
    a_area[a_area!=1]=0
    d_area=d_mn_compare[1]-d_mn_compare[0]
    d_area[d_area>0.016]=1
    d_area[d_area<-0.016]=1
    d_area[d_area!=1]=0
    
    bunching_area=a_area+d_area
    bunching_area[bunching_area!=0]=1#受串车影响范围
    
    return bunching_area

def w_time(w_mn,B_a_mn,bunching_area):
    w_1=(w_mn[:M_1,:]*bunching_area[:M_1,:]).sum(axis=1).sum(axis=0)/(B_a_mn[:M_1,:]*bunching_area[:M_1,:]).sum(axis=1).sum(axis=0)
    w_2=(w_mn[M_1:,:]*bunching_area[M_1:,:]).sum(axis=1).sum(axis=0)/(B_a_mn[M_1:,:]*bunching_area[M_1:,:]).sum(axis=1).sum(axis=0)
    w=((w_mn[:,:]*bunching_area[:,:]).sum(axis=1).sum(axis=0))/((B_a_mn[:,:]*bunching_area[:,:]).sum(axis=1).sum(axis=0))

    return w_1, w_2, w

def headway(d_mn, bunching_area ): #求一种情况的
    Headway_1=np.zeros([M_1,N_1+N_3+N_4])   #线路1的车头时距
    Headway_2=np.zeros([M_2,N_2+N_3+N_5])   #线路2的车头时距
    #bunching区域的车头时距
    Headway_1[1:,:N_1]=(d_mn[1:M_1,:N_1]-d_mn[0:M_1-1,:N_1])*bunching_area[1:M_1,:N_1]
    Headway_1[1:,N_1:]=(d_mn[1:M_1,N2:N2+N_3+N_4]-d_mn[0:M_1-1,N2:N2+N_3+N_4])*bunching_area[1:M_1,N2:N2+N_3+N_4]
    Headway_2[1:,:N_2+N_3]=(d_mn[M_1+1:,N1:N3]-d_mn[M_1:M_1+M_2-1,N1:N3])*bunching_area[M_1+1:,N1:N3]
    Headway_2[1:,N_2+N_3:]=(d_mn[M_1+1:,N_4+N3:]-d_mn[M_1:M_1+M_2-1,N_4+N3:])*bunching_area[M_1+1:,N_4+N3:]
    Headway_1_in=Headway_1[Headway_1!=0]
    Headway_2_in=Headway_2[Headway_2!=0]
    Sigma_1=np.std(Headway_1_in[:])
    Sigma_2=np.std(Headway_2_in[:])

    
    return Sigma_1, Sigma_2

def result(w,B,d,area):
    w_10,w_20,w0 = w_time(w[0],B[0],area)
    sig_10,sig_20 = headway(d[0], area)[0], headway(d[0], area)[1]
    w_11,w_21,w1 = w_time(w[1],B[1],area)
    sig_11,sig_21 = headway(d[1], area)[0],headway(d[1], area)[1]        
    rise_w = (w1-w0)/w0
    rise_w_1 = (w_11-w_10)/w_10
    rise_w_2 = (w_21-w_20)/w_20
    rise_sig_1 = (sig_11-sig_10)
    rise_sig_2 = (sig_21-sig_20)
    return w0, w1, rise_w, w_10, w_11, rise_w_1, w_20, w_21, rise_w_2, sig_10, sig_11, rise_sig_1, sig_20, sig_21, rise_sig_2

flag = 2

M_1, M_2 = get_M(flag)
M_list=[M_1,M_2]

Omega_1,Omega_2,Omega_3,Omega_4,Omega_5 = get_OMEGA(flag)
Omega_list = [Omega_1,Omega_2,Omega_3,Omega_4,Omega_5]

N_1,N_2,N_3,N_4,N_5,N1,N2,N3,N4,N5 = get_N(Omega_list)
N_list = [N_1,N_2,N_3,N_4,N_5,N1,N2,N3,N4,N5]

S_1,S_2 = get_S(Omega_list,flag)
S_list = [S_1,S_2]

T_n_1,T_n_2 = get_T(S_list,flag)
T_list = [T_n_1,T_n_2]

h_1,h_2,H_1,H_2,H_c = get_H(flag)
H_list = [h_1,h_2,H_1,H_2,H_c]

if (flag==0)|(flag==1):
    a_mn_compare,d_mn_compare,xy_compare_1,xy_compare_2,xy_compare,xy_local_compare,w_mn_compare,B_a_mn_compare = initial_some_nparray(M_list,N_list,S_list,flag)
elif flag==2:
    a_mn_compare,d_mn_compare,xy_compare_1,xy_compare_2,w_mn_compare,B_a_mn_compare = initial_some_nparray(M_list,N_list,S_list,flag)
#if (flag == 0): operationlist = [0,1] 
#if (flag == 2): operationlist = [0,1] 
operationlist = [0,1] 
needlist = []

experimentlist = np.arange(0,1)
#experimentlist = np.arange(120,210,10)
# experimentlist = np.arange(0,2,0.2)
#'''
#experimentlist = np.arange(120,130,10)
for i in experimentlist:
    Cap = 70
    ratio = 0
    if flag == 2:
        Cap = 100
        ratio = 0
    for dis in operationlist:
    
        a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,w_mn = initial_littlearray(M_list,N_list,flag,H_list)
    
        delay = dis*1
        SetofCommonStops(1,1,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a)
        SetofCommonStops(2,2,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a)
        SetofCommonStops(0,3,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a)
        SetofCommonStops(1,4,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a)
        SetofCommonStops(2,5,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio,a)
        d_mn = a_mn + W_mn
        a_mn_compare[dis],d_mn_compare[dis]=a_mn,d_mn
        w_mn_compare[dis]=w_mn
        B_a_mn_compare[dis]=B_max_mn
        
        if (flag==0):
            xy_compare[dis],xy_compare_1[dis],xy_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn,d_mn)
        elif flag==2:
            xy_compare_1[dis],xy_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn,d_mn)

    a_mn_real_compare=getrealarrivaltime(a_mn_compare,M_list,N_list,T_list,d_mn_compare)
    xy_real_compare_1 = np.zeros([2,len(S_1)*2,M_1])
    xy_real_compare_2 = np.zeros([2,len(S_2)*2,M_2])
    xy_real_compare = np.zeros([2,len(S_2)*2,M_1+M_2])
    for dis in operationlist:
        if (flag==0):
            xy_real_compare[dis],xy_real_compare_1[dis],xy_real_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn_real_compare[dis],d_mn_compare[dis])
        elif flag==2:
            xy_real_compare_1[dis],xy_real_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn_real_compare[dis],d_mn_compare[dis]) 

    bunching_area = area(a_mn_compare,d_mn_compare)
    
    need = result(w_mn_compare,B_a_mn_compare,d_mn_compare,bunching_area)
    needlist.append(need)
#'''
'''
Cap = 130
for dis in operationlist:

    a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,w_mn = initial_littlearray(M_list,N_list,flag,H_list)

    delay = dis*1
    SetofCommonStops(1,1,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio)
    SetofCommonStops(2,2,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio)
    SetofCommonStops(0,3,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio)
    SetofCommonStops(1,4,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio)
    SetofCommonStops(2,5,a_mn,Wb_mn,Wc_mn,Wa_mn,W_mn,d_mn,w_mn,B_mn,B_max_mn,A_mn,C_mn,p_mnn,P_mn,L_mnn,flag,Omega_list,M_list,N_list,T_list,delay,H_list,Cap,ratio)
    d_mn = a_mn + W_mn
    a_mn_compare[dis],d_mn_compare[dis]=a_mn,d_mn
    w_mn_compare[dis]=w_mn
    B_a_mn_compare[dis]=B_max_mn
    
    if flag==0:
        xy_compare[dis],xy_compare_1[dis],xy_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn,d_mn)
    elif flag==2:
        xy_compare_1[dis],xy_compare_2[dis] = gettrajectories(flag,M_list,N_list,a_mn,d_mn)
    
bunching_area = area(a_mn_compare,d_mn_compare)

need = result(w_mn_compare,B_a_mn_compare,d_mn_compare,bunching_area)
'''
