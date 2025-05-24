# -*- coding: utf-8 -*-
def Re(data,year):
   #data--表示某只股票代码的交易数据
   #n--表示选用的方法，0--神经网络，1--支持向量机，2--逻辑回归
   #返回结果为一个元组(R,r_list,r_total,sc)
   #其中R--表示预测准确率，r_list--表示策略每次投资收益
   # r_total--表示总收益，sc--表示模型准确率

   ##1.计算指标
   import pandas as pd
   import Ind
   import numpy as np
   MA= Ind.MA(data,5,10,20) 
   macd=Ind.MACD(data)
   kdj=Ind.KDJ(data,9)
   rsi6=Ind.RSI(data,6)
   rsi12=Ind.RSI(data,12)
   rsi24=Ind.RSI(data,24)
   bias5=Ind.BIAS(data,5)
   bias10=Ind.BIAS(data,10)
   bias20=Ind.BIAS(data,20)
   obv=Ind.OBV(data) 
   y=Ind.cla(data)
   pm={'交易日期':data['Trddt'].values}
   PM=pd.DataFrame(pm)
   DF={'MA5':MA[0],'MA10':MA[1],'MA20':MA[2],'MACD':macd,
    'K':kdj[0],'D':kdj[1],'J':kdj[2],'RSI6':rsi6,'RSI12':rsi12,
    'RSI24':rsi24,'BIAS5':bias5,'BIAS10':bias10,'BIAS20':bias20,'OBV':obv}
   DF=pd.DataFrame(DF)
   s1=PM.join(DF)
   y1={'涨跌趋势':y}
   ZZ=pd.DataFrame(y1)
   s2=s1.join(ZZ)
   ss=s2.dropna()
   Data=ss[ss.iloc[:,6].values!=0]
   ##2.训练样本与测试样本的划分
   x1=Data['交易日期']>=year+'-01-01'
   x2=Data['交易日期']<=year+'-11-30'
   xx=x1&x2
   index=xx.values==True
   index1=xx.values==False
   x_train=Data.iloc[index,1:15]  
   y_train=Data.iloc[index,[15]]
   x_test=Data.iloc[index1,1:15]
   y_test=Data.iloc[index1,[15]]
   ##3.数据标准化处理
   from sklearn.preprocessing import StandardScaler  
   scaler = StandardScaler()
   scaler.fit(x_train) 
   x_train=scaler.transform(x_train)
   x_test=scaler.transform(x_test)     
   #逻辑回归模型
   from sklearn.linear_model import LogisticRegression as LR
   clf = LR()
   clf.fit(x_train, y_train) 
   result=clf.predict(x_test)
   sc=clf.score(x_train, y_train)
   sc=np.round(sc,3)
   result=pd.DataFrame(result)
   ff=Data.iloc[index1,0]
   #将预测结果与实践结果整合在一起，进行比较
   pm1={'交易日期':ff.values,'预测结果':result.iloc[:,0].values,
        '实际结果':y_test.iloc[:,0].values}
   result1=pd.DataFrame(pm1)
   z=result1['预测结果'].values-result1['实际结果'].values
   R=len(z[z==0])/len(z)
   R=np.round(R,3)   
   ##5.收益率计算
   r_list=[]
   for t in range(len(result1)-1):
      if result1['预测结果'].values[t]==1:
         p1=data.loc[data['Trddt'].values== result1['交易日期'].values[t],'Clsprc'].values
         dt=data.loc[data['Trddt'].values>result1['交易日期'].values[t],['Trddt','Clsprc']]
         dt=dt.sort_values('Trddt')
         p2=dt['Clsprc'].values[0]
         r=(p2-p1)/p1
         r_list.append(r)
   r_total=sum(r_list)
   r_total=np.round(r_total,3)
   return (R,r_list,r_total,sc)


