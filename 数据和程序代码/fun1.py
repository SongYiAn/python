# -*- coding: utf-8 -*-
def Fr1(data, year):
    import pandas as pd
    # 读取所需年份数据，其中第0列为标识列（股票代码）
    data2 = data.iloc[data['Accper'].values == str(year) + '-12-31', [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    # 筛选指标值大于0的数据以及去掉NAN值
    data2 = data2[data2 > 0]
    data2 = data2.dropna()
    # 数据标准化，注意标准化的数据需要去掉第0列（股票代码，标识列），这里数据标准化方法采用均值-方差法。
    from sklearn.preprocessing import StandardScaler
    X = data2.iloc[:, 1:]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # 主成分分析
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)  # 累计贡献率为95%
    Y = pca.fit_transform(X)  # 满足累计贡献率为95%的主成分数据
    gxl = pca.explained_variance_ratio_  # 贡献率
    import numpy as np
    F = np.zeros((len(Y)))
    for i in range(len(gxl)):
        f = Y[:, i] * gxl[i]
        F = F + f
    Fscore1 = pd.DataFrame(data2['Stkcd'].values, columns=['Stkcd'])
    Fscore1['zhdf'] = F
    Fscore1 = Fscore1.sort_values('zhdf', ascending=False)  # 降序，True为升序

    co = pd.read_excel('上市公司信息表.xlsx')
    Co = pd.Series(co['Stknme'].values, index=co['Stkcd'].values)
    Co1 = Co[data2['Stkcd'].values]
    Fscore2 = pd.DataFrame(Co1.values, columns=['Stknme'])
    Fscore2['zhdf'] = F
    Fscore2 = Fscore2.sort_values('zhdf', ascending=False)  # 降序，True为升序
    return (Fscore1, Fscore2)


