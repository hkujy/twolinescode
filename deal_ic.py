# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:21:26 2020
导入4月1日的刷卡数据： ic_0401
找出814和882的有效刷卡数据：df_ic
找到两线路运行的车辆：
814：bus1_ic
882：bus2_ic

@author: dell
"""
#import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def getSandEtime(s,e):
    peak_s_t = datetime.datetime.strptime(s, '%Y%m%d%H')
    peak_e_t = datetime.datetime.strptime(e, '%Y%m%d%H')
    return peak_s_t,peak_e_t

def getIC(ic_0401,l_c_1,l_c_2):
    df_ic = (ic_0401
            .loc[(ic_0401['LINE_CODE']==l_c_1) | (ic_0401['LINE_CODE']==l_c_2)]
            .loc[ic_0401['DEAL_TYPE']==6]
            [['DEAL_TYPE','GRANT_CARD_CODE','VEHICLE_CODE','LINE_CODE','ON_STATION','UP_TIME','OFF_STATION','DEAL_TIME']]) 
    
    df_ic['计算上车时间'] = pd.to_datetime(df_ic['UP_TIME'])
    df_ic['计算下车时间'] = pd.to_datetime(df_ic['DEAL_TIME'])
        
    '''乘客的行程时间'''
    df_ic['Travel_second'] = pd.to_datetime(df_ic['DEAL_TIME'])-pd.to_datetime(df_ic['UP_TIME'])    
    
    return df_ic
'''源文件，2018年4月1日，起始时间'''
ic_0401 = pd.read_csv('D:/bus_IC_data/20180402-bus.csv')
peak_s_t, peak_e_t =  getSandEtime('2018040205','2018040212')
starttimestr = '20180402 05:00'

'''各列的类型'''
t_ic_0401 = ic_0401.dtypes

'''把目标线路挑出来'''
l_c_1 = 24027
l_c_2 = 43017

'''得到两线路的正常刷卡数据'''
df_ic = getIC(ic_0401,l_c_1,l_c_2)

#########################以下是为了得到发车时间和发车间隔##########################

def gettrajectory(num,df_ic):
    swipeboardingtime = []
    droplist_board = []
    swipealightingtime = []
    droplist_alight = []
    for i in np.arange(1,num):
        boardpas = df_ic.loc[(df_ic['ON_STATION']==i)]#所有在i站上车的乘客
        alightpas = df_ic.loc[(df_ic['OFF_STATION']==i)]#所有在i站上车的乘客
        dt_1_board, droplist_1_board = getfirstdepart(l_c_1,24,'30Min',boardpas)#线路1同一车辆，最早、晚上车的乘客
        dt_2_board, droplist_2_board = getfirstdepart(l_c_2,24,'30Min',boardpas)#线路2同一车辆，最早、晚上车的乘客
        dt_1_alight, droplist_1_alight = getfirstdepart(l_c_1,24,'30Min',alightpas)#线路1同一车辆，最早、晚下车的乘客
        dt_2_alight, droplist_2_alight = getfirstdepart(l_c_2,24,'30Min',alightpas)#线路2同一车辆，最早、晚下车的乘客
        swipeboardingtime.append([dt_1_board,dt_2_board])
        swipealightingtime.append([dt_1_alight,dt_2_alight])
        droplist_board.append([droplist_1_board,droplist_2_board])
        droplist_alight.append([dt_1_alight,dt_2_alight])
    return swipeboardingtime,droplist_board,swipealightingtime,droplist_alight


'''在始发站的刷卡数据'''
#depart = df_ic.loc[(df_ic['ON_STATION']==1)]

'''整理发车车辆和发车时间'''
def pickoutre(dt):
    length = len(dt)
    index = np.argsort(dt, 0)[:, 1]
    dt_sort = []
    for i in index:
        dt_sort.append(dt[i])
    droplist = []
    for i in np.arange(1,length):
        if dt_sort[i][0]==dt_sort[i-1][0]:
            droplist.append(i-1)

    return dt_sort, droplist

'''得到发车车辆，时间，与前车间隔 '''
def getfirstdepart(linecode,periods,freq,depart):
    '''指定线路'''
    depart_l = depart.loc[depart['LINE_CODE']==linecode]
    '''把一天等分成多个时段'''
    timelist = pd.date_range('20180402 00:00',periods=periods,freq=freq).tolist()
    dt = []
    for i in np.arange(periods-1):
        start=timelist[i]
        end=timelist[i+1]
        timeperiod = depart_l.loc[(depart_l['计算上车时间']>=start)&(depart_l['计算上车时间']<end)]
        car = timeperiod['VEHICLE_CODE']
        car = car.value_counts().reset_index()
        car_list = car['index'].tolist()
        '''找到一个时段内的所有车辆'''
        for m in car_list:
            '''第一个上车时间当作到站时间'''
            depart_time = timeperiod.loc[timeperiod['VEHICLE_CODE']==m]['计算上车时间'].min()
            depart_time_1 = timeperiod.loc[timeperiod['VEHICLE_CODE']==m]['计算上车时间'].max()
            dt.append([m,depart_time,depart_time_1])
    '''整理发车车辆和发车时间'''
    dt_sort, droplist = pickoutre(dt)
    
    dt_sort_array = np.array(dt_sort)
    interval = (dt_sort_array[1:,1]-dt_sort_array[:-1,1]).tolist()
    dt_df = pd.DataFrame(dt_sort,columns = ['VEHICLE_CODE','DEPART_TIME','Depart_TIME_1'])
    dt_df['INTERVAL']=pd.DataFrame(interval)

    Itimeminute = []
    Dtimeminute = []
    s_h=5
    s_m=16
    for i in np.arange(len(dt_df)-1):
        intervalstr = str(dt_df.loc[i]['INTERVAL'])
        intervalmin=(int(intervalstr[7:9]))*60+int(intervalstr[10:12])
        Itimeminute.append(intervalmin)
    for i in np.arange(len(dt_df)):
        departstr = str(dt_df.loc[i]['DEPART_TIME'])
        departmin=(int(departstr[11:13])-s_h)*60+int(departstr[14:16])-s_m
        Dtimeminute.append(departmin)
    Itimeminute= np.array(Itimeminute)
    Dtimeminute= np.array(Dtimeminute)
    dt_df['INTERVALmin']=pd.DataFrame(Itimeminute)
    dt_df['DEPART_TIMEmin']=pd.DataFrame(Dtimeminute)
        
    
    return dt_df, droplist        

#dt_sort, droplist = getfirstdepart(l_c_1,24,'30Min')
#dt_sort_882, droplist_882 = getfirstdepart(l_c_2,24,'30Min')

#dt_sort_pick = dt_sort.drop([9,27,42])
#dt_sort_882_pick = dt_sort_882.drop([12])

##################################################以上是为了得到发车时间和发车间隔#######################


'''两条线路分别涉及到哪些车'''
def getcar(linecode,df_ic):
    bus1_ic = df_ic[df_ic['LINE_CODE']==linecode]['VEHICLE_CODE'].value_counts().reset_index()
    bus1_ic.columns = ['涉及车辆','出现次数']
    return bus1_ic

bus1_ic = getcar(l_c_1,df_ic)
bus2_ic = getcar(l_c_2,df_ic)
vehicle_1 = bus1_ic['涉及车辆'].tolist()
vehicle_2 = bus2_ic['涉及车辆'].tolist()
vehicle_1_str = []
vehicle_2_str = []
for i in np.arange(len(vehicle_1)):
#    vehicle_1.append(str(vehicle_1[i]))
    vehicle_1_str.append(str(vehicle_1[i]))
for i in np.arange(len(vehicle_2)):
#    vehicle_2.append(str(vehicle_2[i]))
    vehicle_2_str.append(str(vehicle_2[i]))

'''挑一辆，比如72375，绘制刷卡时间分布图，结合GPS数据判断'''
'''线路1的32401，线路2的19580'''
def plotabus(vehicle_code):
    y1 = df_ic.loc[df_ic['VEHICLE_CODE']==vehicle_code]['ON_STATION'].tolist()
    x1 = df_ic.loc[df_ic['VEHICLE_CODE']==vehicle_code]['计算上车时间'].tolist()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x1, y1, s=20, c='k', marker='.')
    #plt.xlim(xmax=10, xmin=0)
    plt.show()



#################################以下是得到OD表
'''得到一个线路的OD表'''
def getOD(code,s_t,e_t):
    '''有哪些上车站台'''
    b_stop = df_ic[df_ic['LINE_CODE']==code]['ON_STATION'].value_counts().index.tolist()
    '''排序'''
    b_stop.sort()
    
    '''有哪些下车站台'''
    a_stop = df_ic[df_ic['LINE_CODE']==code]['OFF_STATION'].value_counts().index.tolist()
    '''排序'''
    a_stop.sort()   
    
    q_ba = pd.DataFrame(np.zeros([b_stop[-1],a_stop[-1]]), columns=np.arange(1,a_stop[-1]+1),index = np.arange(1,b_stop[-1]+1))
    for i in df_ic.loc[(df_ic['LINE_CODE']==code)&(df_ic['计算上车时间']>=s_t)&(df_ic['计算上车时间']<=e_t)].index.tolist():
        b = df_ic.loc[i,'ON_STATION']
        a = df_ic.loc[i,'OFF_STATION']
        count = q_ba.loc[b,a] + 1
        q_ba.loc[b,a] = count    
    
    return q_ba

'''得到两个时刻之间的乘客量''' 
#q_ba_814 = getOD(24027,peak_s_t,peak_e_t)    
#q_ba_882 = getOD(43017,peak_s_t,peak_e_t)    


#################################将两个OD表合起来

'''这个计算是否有问题'''
#S_1_index = [1,0,0,0,0,0,0,0,2,3,4,5,6,7,8,9,10,11,12,0]#和最终OD表一样长
#S_2_index = [0,1,2,3,4,5,6,7,8,9,10,11,0,0,0,0,0,0,0,12]
def combinetwoOD_new(q_ba_814,q_ba_882,S_1,S_2):
    timelen = 60
    com = [9,10,11,12]
    S_1_index = [1,0,0,0,0,0,0,0,2,3,4,5,6,7,8,9,10,11,12,0]#和最终OD表一样长
    S_2_index = [0,1,2,3,4,5,6,7,8,9,10,11,0,0,0,0,0,0,0,12]
    twoOD = pd.DataFrame(np.zeros([len(S_1_index),len(S_1_index)]),columns = list(np.arange(0,20)))
    for i in np.arange(0,20):
        for j in np.arange(i+1,20):
            if i+1 in S_1:#O是线路1的
                if j+1 in S_1:#D是线路1的
                    if (i+1 in com) & (j+1 in com):#O，D都是com
                        twoOD.loc[i,j] = (q_ba_814.loc[S_1_index[i],S_1_index[j]]+q_ba_882.loc[S_2_index[i],S_2_index[j]])/timelen
                    else:
                        twoOD.loc[i,j] =  q_ba_814.loc[S_1_index[i],S_1_index[j]]/timelen
                elif (i+1 in S_2) & (j+1 in S_2):
                    twoOD.loc[i,j] =  q_ba_882.loc[S_2_index[i],S_2_index[j]]/timelen
            elif i+1 in S_2:
                if j+1 in S_2:
                    twoOD.loc[i,j] =  q_ba_882.loc[S_2_index[i],S_2_index[j]]/timelen
                elif (i+1 in S_1) & (j+1 in S_1):
                    twoOD.loc[i,j] =  q_ba_814.loc[S_1_index[i],S_1_index[j]]/timelen
    return twoOD

###############################分段的OD并且combine

'''得到分段的ODλ，子函数 getOD,combinetwoOD_new'''
def getOD_new_seg(periods):
    N_1 = np.arange(1, 2)
    N_2 = np.arange(2, 9)
    N_3 = np.arange(9, 13)
    N_4 = np.arange(13, 20)
    N_5 = np.arange(20, 21)    
    S_1 = list(np.append(np.append(N_1,N_3),N_4))
    S_2 = list(np.append(np.append(N_2,N_3),N_5))
    
    timelist = pd.date_range(starttimestr,periods=periods,freq='60Min').tolist()
    OD=[]
    q_ba=[]
    for i in np.arange(len(timelist)-1):
        q_ba_814 = getOD(l_c_1,timelist[i],timelist[i+1])
        q_ba_882 = getOD(l_c_2,timelist[i],timelist[i+1])
        twoOD_rtol = combinetwoOD_new(q_ba_814,q_ba_882,S_1,S_2)
        OD.append(twoOD_rtol)
        q_ba.append([q_ba_814,q_ba_882,timelist[i]])
    
    return OD,q_ba
a,b = getOD_new_seg(8)

#bef_s_t = datetime.datetime.strptime('2018040205', '%Y%m%d%H')
#bef_e_t = datetime.datetime.strptime('2018040206', '%Y%m%d%H')
#
#bef6 = getOD(814,s_t,e_t)


'''flag==2 的用法，导入数据，函数getOD，combinetwoOD_new，运行getOD_new_seg'''


def get_pas(linecode,df_ic):
    pas1_ic = df_ic[df_ic['LINE_CODE']==linecode]['GRANT_CARD_CODE'].value_counts().reset_index()
    pas1_ic.columns = ['乘客','出现次数']
    return pas1_ic

pas1_ic = get_pas(l_c_1,df_ic)
pas2_ic = get_pas(l_c_2,df_ic)



