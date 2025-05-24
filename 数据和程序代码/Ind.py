# -*- coding: utf-8 -*-
import pandas as pd
#移动平均线指标的计算
def MA(data,N1,N2,N3):
   MAN1=pd.rolling_mean(data['Clsprc'].values,N1) 
   MAN2=pd.rolling_mean(data['Clsprc'].values,N2) 
   MAN3=pd.rolling_mean(data['Clsprc'].values,N3) 
   return (MAN1,MAN2,MAN3)
#指数平滑异同平均线的计算
def MACD(data): 
    import numpy as np 
    EMA12 = pd.ewma(data['Clsprc'].values, 12)
    EMA26 = pd.ewma(data['Clsprc'].values, 26)
    DIF=EMA12- EMA26
    DEA=np.zeros((len(DIF)))
    MACD=np.zeros((len(DIF)))
    for t in range(len(DIF)):
        if t==0:
             DEA[t]= DIF[t]
        if t>0:
             DEA[t]=(2*DIF[t]+8*DEA[t-1])/10
        MACD[t]=2*(DIF[t]-DEA[t])
    return MACD
#随机指标KDJ的计算
def KDJ(data,N):
    import numpy as np 
    Lmin=pd.rolling_min(data['Loprc'].values,N)
    Lmax=pd.rolling_max(data['Hiprc'].values,N)
    RSV=(data['Clsprc'].values-Lmin)/(Lmax-Lmin)
    K=np.zeros((len(RSV)))
    D=np.zeros((len(RSV)))
    J=np.zeros((len(RSV)))
    for t in range(N,len(data)):
        if t==0:
            K[t]=RSV[t]
            D[t]=RSV[t]
        if t>0:
            K[t]=2/3*K[t-1]+1/3*RSV[t]
            D[t]=2/3*D[t-1]+1/3*K[t]
        J[t]=3*D[t]-2*K[t]
    return (K,D,J)
#相对强弱指标的计算
def RSI(data,N):
    import numpy as np
    z=np.zeros(len(data)-1) 
    z[data.iloc[1:,2].values-data.iloc[0:-1,2].values>=0]=1
    z[data.iloc[1:,2].values-data.iloc[0:-1,2].values<0]=-1
    z1=pd.rolling_sum(z==1,N)
    z2=pd.rolling_sum(z==-1,N)
    rsi=np.zeros((len(data)))
    for t in range(N-1,len(data)-1):
        rsi[t]=z1[t]/(z1[t]+z2[t])
    return rsi
#乖离率指标
def BIAS(data,N):
    import numpy as np
    bias=np.zeros((len(data)))
    man=pd.rolling_mean(data.iloc[:,2].values,N)
    for t in range(N-1,len(data)):
        bias[t]=(data.iloc[t,2]-man[t])/man[t]
    return bias
#能量潮指标的计算
def OBV(data):
    import numpy as np
    obv=np.zeros((len(data)))
    for t in range(len(data)):
        if t==0:
            obv[t]=data['Dnshrtrd'].values[t]
        if t>0:
            if data['Clsprc'].values[t]>=data['Clsprc'].values[t-1]:
                obv[t]=obv[t-1]+data['Dnshrtrd'].values[t]
            if data['Clsprc'].values[t]<data['Clsprc'].values[t-1]:
                obv[t]=obv[t-1]-data['Dnshrtrd'].values[t]
    return obv
#涨跌趋势（分类指标/预测指标）的计算
def cla(data):
    import numpy as np
    y=np.zeros(len(data)) 
    z=np.zeros(len(y)-1)
    for i in range(len(z)):
        z[data.iloc[1:,2].values-data.iloc[0:-1,2].values>0]=1
        z[data.iloc[1:,2].values-data.iloc[0:-1,2].values==0]=0
        z[data.iloc[1:,2].values-data.iloc[0:-1,2].values<0]=-1
        y[i]=z[i]  
    return y

