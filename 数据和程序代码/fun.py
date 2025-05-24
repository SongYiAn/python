def Fr(data):
   import pandas as pd
   #data=pd.read_excel('Data.xlsx')
   data=data[data>0]
   data=data.dropna()
   from sklearn.preprocessing import StandardScaler  
   X=data.iloc[:,1:]
   scaler = StandardScaler()
   scaler.fit(X) 
   X=scaler.transform(X)  
   from sklearn.decomposition import PCA 
   pca=PCA(n_components=0.95)      #累计贡献率为95%
   Y=pca.fit_transform(X)            #满足累计贡献率为95%的主成分数据
   gxl=pca.explained_variance_ratio_   #贡献率
   import numpy as np
   F=np.zeros((len(Y)))
   for i in range(len(gxl)):
       f=Y[:,i]*gxl[i]
       F=F+f
    
   fs1=pd.Series(F,index=data['ts_code'].values)
   Fscore1=fs1.sort_values(ascending=False)   #降序，True为升序
   
   co=pd.read_excel('stkcode.xlsx')
   #Co=pd.Series(co['name'].values,index=co['ts_code'].values)
   #Co1=Co[data['ts_code'].values]
   #fs2=pd.Series(F,index=Co1.values)
   dict1={'ts_code':data['ts_code'].values,'F':F}
   df1=pd.DataFrame(dict1)
   df2=pd.merge(df1,co.iloc[:,[0,2]],how='inner',on='ts_code')
   fs2=pd.Series(df2['F'].values,index=df2['name'].values)
   Fscore2=fs2.sort_values(ascending=False)   #降序，True为升序
   return (Fscore1,Fscore2)




