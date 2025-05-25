# -*- coding: utf-8 -*-
"""
量化投资分析系统 - 基于财务与交易数据
整合版本，包含主成分分析、技术指标计算、逻辑回归预测等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import sys
import os
import warnings
warnings.filterwarnings('ignore')

class QuantitativeAnalysisSystem:
    def __init__(self):
        self.sw_data = None  # 申银万国行业分类数据
        self.company_data = None  # 上市公司数据
        self.stock_info = None  # 股票信息
        self.analysis_results = {}

    
    def load_basic_data(self):
        """加载基础数据"""
        try:
            self.sw_data = pd.read_excel('sw.xlsx')
            self.company_data = pd.read_excel('上市公司总体规模与投资效率指标.xlsx')
            self.stock_info = pd.read_excel('上市公司信息表.xlsx')
            print("基础数据加载成功")
            return True
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            return False
    
    def financial_analysis(self, data_year='2014'):
        """财务数据主成分分析 """
        print(f"开始进行{data_year}年财务数据分析...")
        
        try:
            # 读取指定年份数据
            data = pd.read_excel(f'Data{data_year}.xlsx')
            print(f"成功读取{len(data)}条{data_year}年数据")
            
            # 数据预处理：去除负值和缺失值
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_clean = data.copy()

            # 仅对数值列进行大于0的筛选
            for col in numeric_cols:
                data_clean = data_clean[data_clean[col] > 0]

            data_clean = data_clean.dropna()
            
            if len(data_clean) == 0:
                print("警告：没有有效数据进行分析")
                return None
            
            # 数据标准化
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            X = data_clean[numeric_cols]  # 只选择数值列进行标准化
            X_scaled = scaler.fit_transform(X)
            
            # 主成分分析
            pca = PCA(n_components=0.95)  # 累计贡献率为95%
            Y = pca.fit_transform(X_scaled)
            contribution_rates = pca.explained_variance_ratio_
            
            print(f"主成分数量: {len(contribution_rates)}")
            print(f"累计贡献率: {sum(contribution_rates):.4f}")
            
            # 计算综合得分
            F = np.zeros(len(Y))
            for i in range(len(contribution_rates)):
                f = Y[:, i] * contribution_rates[i]
                F = F + f
            
            # 确定股票代码列
            stock_code_col = None
            possible_cols = ['ts_code', 'stock_code', 'Stkcd', '股票代码']
            for col in possible_cols:
                if col in data_clean.columns:
                    stock_code_col = col
                    break

            if stock_code_col is None:
                print("找不到股票代码列")
                return None

            # 按股票代码排序
            fs1 = pd.Series(F, index=data_clean[stock_code_col].values)
            Fscore1 = fs1.sort_values(ascending=False)
            
            # 按股票名称排序
            try:
                # 读取股票信息
                co = pd.read_excel('stkcode.xlsx')

                # 找出stkcode表中对应的代码列
                co_code_col = None
                for col in possible_cols:
                    if col in co.columns:
                        co_code_col = col
                        break

                if co_code_col is None:
                    print("在stkcode表中找不到股票代码列")
                    return (Fscore1, None)

                # 确定名称列
                name_col = None
                for col in ['name', 'Stknme', '名称']:
                    if col in co.columns:
                        name_col = col
                        break

                if name_col is None:
                    print("在stkcode表中找不到名称列")
                    return (Fscore1, None)

                # 创建临时DataFrame
                df1 = pd.DataFrame({stock_code_col: data_clean[stock_code_col].values, 'F': F})

                # 执行合并
                df2 = pd.merge(df1, co[[co_code_col, name_col]], how='inner', left_on=stock_code_col, right_on=co_code_col)
                fs2 = pd.Series(df2['F'].values, index=df2[name_col].values)
                Fscore2 = fs2.sort_values(ascending=False)
                
                self.analysis_results[f'financial_{data_year}'] = {
                    'by_code': Fscore1,
                    'by_name': Fscore2,
                    'pca_components': len(contribution_rates),
                    'total_variance': sum(contribution_rates)
                }
                
                print(f"\n{data_year}年财务分析完成")
                print("综合得分排名前10（按股票名称）:")
                print(Fscore2.head(10))
                
                return (Fscore1, Fscore2)
                
            except Exception as e:
                print(f"合并股票信息时出错: {e}")
                import traceback
                traceback.print_exc()
                return (Fscore1, None)
                
        except Exception as e:
            print(f"财务分析出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def investment_efficiency_analysis(self, analysis_year='2014'):
        """投资效率分析 """
        print(f"开始进行{analysis_year}年投资效率分析...")
        
        try:
            # 读取指定年份数据
            data2 = self.company_data[self.company_data['Accper'] == f'{analysis_year}-12-31'].copy()

            # 确定股票代码列名
            stock_code_col = None
            possible_cols = ['ts_code', 'Stkcd', 'stock_code', '股票代码', '证券代码']
            for col in possible_cols:
                if col in data2.columns:
                    stock_code_col = col
                    break

            if stock_code_col is None:
                print(f"警告：在公司数据中找不到股票代码列")
                # 假设第一列是股票代码
                stock_code_col = data2.columns[0]
                print(f"使用第一列 '{stock_code_col}' 作为股票代码列")

            # 选择相关列：股票代码列和数值列
            numeric_cols = data2.select_dtypes(include=[np.number]).columns.tolist()

            # 移除重复的股票代码列
            numeric_cols = [col for col in numeric_cols if col != stock_code_col]

            # 创建新的数据框，避免重复列名
            cols_to_use = [stock_code_col] + numeric_cols
            data_clean = data2[cols_to_use].copy()

            print(f"数据列: {data_clean.columns.tolist()}")

            # 数据清洗：只对数值列进行过滤
            for col in numeric_cols:
                data_clean = data_clean[data_clean[col] > 0]

            data_clean = data_clean.dropna()

            if len(data_clean) == 0:
                print("警告：没有有效数据进行投资效率分析")
                return None
            
            # 数据标准化
            scaler = StandardScaler()
            X = data_clean[numeric_cols]  # 只选择数值列
            X_scaled = scaler.fit_transform(X)
            
            # 主成分分析
            pca = PCA(n_components=0.95)
            Y = pca.fit_transform(X_scaled)
            contribution_rates = pca.explained_variance_ratio_
            
            # 计算综合得分
            F = np.zeros(len(Y))
            for i in range(len(contribution_rates)):
                f = Y[:, i] * contribution_rates[i]
                F = F + f
            
            # 创建结果DataFrame
            Fscore1 = pd.DataFrame({
                stock_code_col: data_clean[stock_code_col].values,
                'zhdf': F
            }).sort_values('zhdf', ascending=False)
            
            # 合并股票名称
            try:
                # 确定stock_info中的股票代码列
                info_code_col = None
                for col in possible_cols:
                    if col in self.stock_info.columns:
                        info_code_col = col
                        break

                if info_code_col is None:
                    print(f"警告：在股票信息表中找不到对应的股票代码列")
                    return (Fscore1, None)

                # 确定股票名称列
                name_col = None
                for possible_name in ['name', 'Stknme', '名称', '股票名称']:
                    if possible_name in self.stock_info.columns:
                        name_col = possible_name
                        break

                if name_col is None:
                    print(f"警告：在股票信息表中找不到名称列")
                    return (Fscore1, None)

                # 确保代码类型一致（转为字符串）
                Fscore1[stock_code_col] = Fscore1[stock_code_col].astype(str)
                self.stock_info[info_code_col] = self.stock_info[info_code_col].astype(str)

                # 执行合并
                Fscore2 = pd.merge(Fscore1, self.stock_info[[info_code_col, name_col]], how='inner', left_on=stock_code_col, right_on=info_code_col)

                if len(Fscore2) == 0:
                    print(f"警告：合并后结果为空，检查代码格式是否匹配")
                    print(f"Fscore1代码示例: {Fscore1[stock_code_col].head(3).tolist()}")
                    print(f"股票信息代码示例: {self.stock_info[info_code_col].head(3).tolist()}")
                    return (Fscore1, None)

                Fscore2 = Fscore2.sort_values('zhdf', ascending=False)
                
                self.analysis_results[f'investment_{analysis_year}'] = {
                    'by_code': Fscore1,
                    'by_name': Fscore2,
                    'pca_components': len(contribution_rates)
                }
                
                print(f"\n{analysis_year}年投资效率分析完成")
                print("投资效率排名前10:")
                print(Fscore2.head(10)[[name_col, 'zhdf']])

                return (Fscore1, Fscore2)
                
            except Exception as e:
                print(f"合并股票名称时出错: {e}")
                import traceback
                traceback.print_exc()
                return (Fscore1, None)
                
        except Exception as e:
            print(f"投资效率分析出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def match_industry(self, stock_code):
        """匹配股票所属行业"""
        if self.sw_data is None:
            return "其他"
        
        industry_match = self.sw_data[self.sw_data['Stkcd'] == stock_code]
        if len(industry_match) > 0:
            return industry_match.iloc[0]['行业名称']
        else:
            return "其他"
    
    def technical_indicators(self, data):
        """计算技术指标 """
        # 移动平均线
        MA5 = data['Clsprc'].rolling(5).mean()
        MA10 = data['Clsprc'].rolling(10).mean()
        MA20 = data['Clsprc'].rolling(20).mean()
        
        # MACD指标
        EMA12 = data['Clsprc'].ewm(span=12).mean()
        EMA26 = data['Clsprc'].ewm(span=26).mean()
        DIF = EMA12 - EMA26
        DEA = DIF.ewm(span=9).mean()
        MACD = 2 * (DIF - DEA)
        
        # KDJ指标
        Lmin = data['Loprc'].rolling(9).min()
        Lmax = data['Hiprc'].rolling(9).max()
        RSV = (data['Clsprc'] - Lmin) / (Lmax - Lmin)
        K = RSV.ewm(alpha=1/3).mean()
        D = K.ewm(alpha=1/3).mean()
        J = 3 * D - 2 * K
        
        # RSI指标
        delta = data['Clsprc'].diff()
        gain = delta.where(delta > 0, 0).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        RSI6 = 100 - (100 / (1 + rs))
        
        # BIAS乖离率
        BIAS5 = (data['Clsprc'] - MA5) / MA5
        BIAS10 = (data['Clsprc'] - MA10) / MA10
        BIAS20 = (data['Clsprc'] - MA20) / MA20
        
        # OBV能量潮
        price_change = data['Clsprc'].diff()
        obv = np.zeros(len(data))
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv[i] = obv[i-1] + data['Dnshrtrd'].iloc[i]
            elif price_change.iloc[i] < 0:
                obv[i] = obv[i-1] - data['Dnshrtrd'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        # 涨跌趋势
        trend = np.zeros(len(data))
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                trend[i] = 1
            elif price_change.iloc[i] < 0:
                trend[i] = -1
            else:
                trend[i] = 0
        
        indicators = pd.DataFrame({
            'MA5': MA5, 'MA10': MA10, 'MA20': MA20, 'MACD': MACD,
            'K': K, 'D': D, 'J': J, 'RSI6': RSI6,
            'BIAS5': BIAS5, 'BIAS10': BIAS10, 'BIAS20': BIAS20,
            'OBV': obv, '涨跌趋势': trend
        })
        
        return indicators
    
    def strategy_backtest(self, stock_code, base_year='2014'):
        """策略回测 - 简化版本"""
        try:
            stock_code = str(stock_code).strip()
            next_year = str(int(base_year) + 1)
            trading_file = f'{next_year}年所有上市股票交易数据.xlsx'

            # 检查文件是否存在
            if not os.path.exists(trading_file):
                print(f"文件不存在: {trading_file}")
                return None

            try:
                # 读取数据
                all_trading = pd.read_excel(trading_file)
                all_trading['Stkcd'] = all_trading['Stkcd'].astype(str)

                # 尝试多种匹配方式获取股票数据
                stock_data = all_trading[all_trading['Stkcd'] == stock_code].copy()
                if len(stock_data) == 0:
                    # 尝试只保留数字部分
                    numeric_code = ''.join(c for c in stock_code if c.isdigit())
                    stock_data = all_trading[all_trading['Stkcd'].str.contains(numeric_code)].copy()

                if len(stock_data) < 20:
                    print(f"股票{stock_code}交易记录不足")
                    return None
            except Exception as e:
                print(f"读取数据失败: {e}")
                return None

            # 只保留必要列并排序
            required_cols = ['Stkcd', 'Trddt', 'Clsprc', 'Hiprc', 'Loprc', 'Dnshrtrd']
            available_cols = [col for col in required_cols if col in stock_data.columns]

            if 'Trddt' not in available_cols or 'Clsprc' not in available_cols:
                print(f"缺少必要的列")
                return None

            stock_data = stock_data[available_cols].sort_values('Trddt').reset_index(drop=True)
            stock_data['Trddt'] = pd.to_datetime(stock_data['Trddt'])
            
            # 计算技术指标
            try:
                indicators = self._optimized_technical_indicators(stock_data)
            except:
                try:
                    indicators = self.technical_indicators(stock_data)
                except:
                    print("技术指标计算失败")
                    return None

            # 合并数据
            analysis_data = pd.concat([stock_data, indicators], axis=1)
            analysis_data = analysis_data.dropna()
            
            if len(analysis_data) < 20:
                return None
            
            # 划分训练和测试集
            train_end = f'{next_year}-11-30'
            train_mask = analysis_data['Trddt'] <= train_end
            
            # 如果训练数据太少，使用前70%作为训练集
            if sum(train_mask) < 15:
                train_size = int(len(analysis_data) * 0.7)
                train_mask = np.zeros(len(analysis_data), dtype=bool)
                train_mask[:train_size] = True

            # 选择特征
            feature_cols = ['MA5', 'MA10', 'MA20', 'MACD', 'K', 'D', 'J', 'RSI6', 'BIAS5', 'BIAS10', 'BIAS20', 'OBV']
            available_features = [col for col in feature_cols if col in analysis_data.columns]

            if len(available_features) < 5:
                return None
            
            # 提取特征和目标
            X_train = analysis_data.loc[train_mask, available_features].values
            y_train = analysis_data.loc[train_mask, '涨跌趋势'].values
            X_test = analysis_data.loc[~train_mask, available_features].values
            y_test = analysis_data.loc[~train_mask, '涨跌趋势'].values

            if len(X_train) < 10 or len(X_test) < 5:
                return None

            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 逻辑回归预测
            clf = LogisticRegression(max_iter=500, solver='liblinear')
            clf.fit(X_train_scaled, y_train)
            predictions = clf.predict(X_test_scaled)
            model_score = clf.score(X_train_scaled, y_train)
            accuracy = (predictions == y_test).mean()
            
            # 计算收益率
            test_data = analysis_data.loc[~train_mask].copy()
            test_data['prediction'] = predictions
            
            buy_signals = test_data['prediction'] == 1
            if buy_signals.sum() > 0:
                # 移除最后一个买入信号
                if buy_signals.iloc[-1]:
                    buy_signals.iloc[-1] = False

                if buy_signals.sum() > 0:
                    buy_prices = test_data.loc[buy_signals, 'Clsprc'].values
                    next_day_indices = np.where(buy_signals)[0] + 1
                    next_day_indices = next_day_indices[next_day_indices < len(test_data)]

                    if len(next_day_indices) > 0:
                        next_day_prices = test_data.iloc[next_day_indices]['Clsprc'].values
                        min_len = min(len(buy_prices), len(next_day_prices))
                        returns = (next_day_prices[:min_len] - buy_prices[:min_len]) / buy_prices[:min_len]
                        total_return = np.sum(returns)
                    else:
                        returns = []
                        total_return = 0.0
                else:
                    returns = []
                    total_return = 0.0
            else:
                returns = []
                total_return = 0.0

            return {
                'accuracy': round(accuracy, 3),
                'model_score': round(model_score, 3),
                'total_return': round(total_return, 3),
                'trade_count': len(returns)
            }
            
        except Exception as e:
            print(f"策略回测出错: {e}")
            return None
    
    def _optimized_technical_indicators(self, data):
        """优化版的技术指标计算，使用向量化操作替代循环"""
        # 移动平均线
        MA5 = data['Clsprc'].rolling(5).mean()
        MA10 = data['Clsprc'].rolling(10).mean()
        MA20 = data['Clsprc'].rolling(20).mean()

        # MACD指标
        EMA12 = data['Clsprc'].ewm(span=12).mean()
        EMA26 = data['Clsprc'].ewm(span=26).mean()
        DIF = EMA12 - EMA26
        DEA = DIF.ewm(span=9).mean()
        MACD = 2 * (DIF - DEA)

        # KDJ指标
        Lmin = data['Loprc'].rolling(9).min()
        Lmax = data['Hiprc'].rolling(9).max()
        RSV = (data['Clsprc'] - Lmin) / (Lmax - Lmin)
        K = RSV.ewm(alpha=1/3).mean()
        D = K.ewm(alpha=1/3).mean()
        J = 3 * D - 2 * K

        # RSI指标
        delta = data['Clsprc'].diff()
        gain = delta.where(delta > 0, 0).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss
        RSI6 = 100 - (100 / (1 + rs))

        # BIAS乖离率
        BIAS5 = (data['Clsprc'] - MA5) / MA5
        BIAS10 = (data['Clsprc'] - MA10) / MA10
        BIAS20 = (data['Clsprc'] - MA20) / MA20

        # OBV能量潮 - 向量化实现
        price_change = data['Clsprc'].diff()
        volume = data['Dnshrtrd'].values
        obv = np.zeros(len(data))

        # 设置第一个非NaN位置的OBV为0
        first_valid = 1
        obv[first_valid] = 0

        # 使用numpy向量化操作计算OBV
        price_change_np = price_change.values[first_valid+1:]
        volume_slice = volume[first_valid+1:]

        # 预分配数组
        signs = np.zeros_like(price_change_np)
        signs[price_change_np > 0] = 1
        signs[price_change_np < 0] = -1

        # 计算OBV变化
        obv_changes = signs * volume_slice

        # 累积求和
        obv_values = np.cumsum(obv_changes)
        obv[first_valid+1:] = obv_values

        # 涨跌趋势 - 向量化实现
        trend = np.zeros(len(data))
        trend[1:] = np.sign(price_change.values[1:])  # 使用sign函数将正值转为1，负值转为-1，0保持0

        indicators = pd.DataFrame({
            'MA5': MA5, 'MA10': MA10, 'MA20': MA20, 'MACD': MACD,
            'K': K, 'D': D, 'J': J, 'RSI6': RSI6,
            'BIAS5': BIAS5, 'BIAS10': BIAS10, 'BIAS20': BIAS20,
            'OBV': obv, '涨跌趋势': trend
        })

        return indicators

    def industry_analysis(self, industry_name, analysis_year='2014'):
        """行业分析"""
        print(f"开始进行{industry_name}行业{analysis_year}年分析...")
        
        # 获取该行业的股票代码
        # 检查sw_data中的列名
        if self.sw_data is None or len(self.sw_data) == 0:
            print(f"申万行业分类数据为空")
            return None

        # 检查行业名称列
        industry_col = None
        for col in self.sw_data.columns:
            if '行业' in col:
                industry_col = col
                break

        if industry_col is None:
            print(f"在申万行业分类数据中未找到行业名称列")
            # 假设第一列是行业名称
            industry_col = self.sw_data.columns[0]
            print(f"使用第一列 '{industry_col}' 作为行业名称列")

        # 检查股票代码列
        stock_code_col = None
        possible_cols = ['Stkcd', 'stock_code', '股票代码', '证券代码', 'code']
        for col_name in possible_cols:
            if col_name in self.sw_data.columns:
                stock_code_col = col_name
                break

        if stock_code_col is None:
            print(f"在申万行业分类数据中未找到股票代码列")
            # 假设第二列是股票代码
            if len(self.sw_data.columns) > 1:
                stock_code_col = self.sw_data.columns[1]
                print(f"使用第二列 '{stock_code_col}' 作为股票代码列")
            else:
                print(f"无法确定股票代码列，行业分析无法继续")
                return None

        # 获取该行业的股票代码
        try:
            industry_stocks = self.sw_data[self.sw_data[industry_col] == industry_name][stock_code_col].astype(str).tolist()
            print(f"找到{len(industry_stocks)}只{industry_name}行业股票")
            if len(industry_stocks) > 0:
                print(f"示例股票代码: {industry_stocks[:3]}")
        except Exception as e:
            print(f"获取行业股票代码时出错: {e}")
            return None

        if not industry_stocks:
            print(f"未找到{industry_name}行业的股票")
            return None
        
        # 进行投资效率分析
        investment_result = self.investment_efficiency_analysis(analysis_year)
        if investment_result is None:
            return None
        
        fscore_df = investment_result[0]  # 按代码排序的结果
        
        # 确定fscore_df中的股票代码列
        if 'Stkcd' in fscore_df.columns:
            fscore_stock_col = 'Stkcd'
        elif 'stock_code' in fscore_df.columns:
            fscore_stock_col = 'stock_code'
        else:
            # 尝试查找包含'code'或'代码'的列
            for col in fscore_df.columns:
                if 'code' in col.lower() or '代码' in col:
                    fscore_stock_col = col
                    break
            else:
                # 如果找不到，使用第一列
                fscore_stock_col = fscore_df.columns[0]

        # 重要：确保股票代码格式一致（转为字符串）
        fscore_df[fscore_stock_col] = fscore_df[fscore_stock_col].astype(str)

        # 打印两边代码格式进行调试
        print(f"行业股票代码格式: {type(industry_stocks[0]) if industry_stocks else 'N/A'}")
        print(f"分析结果股票代码格式: {type(fscore_df[fscore_stock_col].iloc[0]) if len(fscore_df) > 0 else 'N/A'}")

        # 检查代码格式，确保匹配（去除可能的前缀）
        cleaned_industry_stocks = []
        for stock in industry_stocks:
            # 保留数字部分
            cleaned_code = ''.join(c for c in stock if c.isdigit())
            if cleaned_code:
                cleaned_industry_stocks.append(cleaned_code)

        # 对分析结果中的代码也做同样处理
        fscore_df['cleaned_code'] = fscore_df[fscore_stock_col].apply(lambda x: ''.join(c for c in str(x) if c.isdigit()))

        # 使用清理后的代码进行匹配
        industry_scores = fscore_df[fscore_df['cleaned_code'].isin(cleaned_industry_stocks)].head(10)

        if industry_scores.empty:
            print(f"未找到{industry_name}行业的投资效率数据")
            # 尝试直接使用原始代码再次匹配
            industry_scores = fscore_df[fscore_df[fscore_stock_col].isin(industry_stocks)].head(10)
            if industry_scores.empty:
                print("使用原始代码匹配也未找到结果")
                return None

        print(f"\n{industry_name}行业投资效率排名前10:")

        # 确保stock_info中的股票代码列
        if 'Stkcd' in self.stock_info.columns:
            info_stock_col = 'Stkcd'
        elif 'stock_code' in self.stock_info.columns:
            info_stock_col = 'stock_code'
        else:
            for col in self.stock_info.columns:
                if 'code' in col.lower() or '代码' in col:
                    info_stock_col = col
                    break
            else:
                info_stock_col = self.stock_info.columns[0]

        # 确保stock_info中的股票名称列
        if 'Stknme' in self.stock_info.columns:
            info_name_col = 'Stknme'
        elif 'name' in self.stock_info.columns:
            info_name_col = 'name'
        else:
            for col in self.stock_info.columns:
                if 'name' in col.lower() or '名称' in col:
                    info_name_col = col
                    break
            else:
                if len(self.stock_info.columns) > 1:
                    info_name_col = self.stock_info.columns[1]
                else:
                    info_name_col = None

        for _, row in industry_scores.iterrows():
            stock_code = row[fscore_stock_col]
            if info_name_col is not None:
                # 转换为相同格式进行匹配
                self.stock_info[info_stock_col] = self.stock_info[info_stock_col].astype(str)
                stock_name_match = self.stock_info[self.stock_info[info_stock_col] == stock_code]
                stock_name = stock_name_match[info_name_col].iloc[0] if len(stock_name_match) > 0 else '未知'
            else:
                stock_name = '未知'
            print(f"{stock_code} ({stock_name}): {row['zhdf']:.4f}")

        # 进行策略回测
        print(f"\n开始对{industry_name}行业前5只股票进行策略回测...")
        backtest_results = []
        
        for _, row in industry_scores.head(5).iterrows():
            result = self.strategy_backtest(row[fscore_stock_col], analysis_year)
            if result:
                result['stock_code'] = row[fscore_stock_col]
                result['investment_score'] = row['zhdf']
                backtest_results.append(result)
        
        if backtest_results:
            print(f"\n{industry_name}行业策略回测结果:")
            for result in backtest_results:
                print(f"股票 {result['stock_code']}: 收益率 {result['total_return']:.4f}, "
                      f"准确率 {result['accuracy']:.3f}, 交易次数 {result['trade_count']}")
        
        return {
            'industry_scores': industry_scores,
            'backtest_results': backtest_results
        }
    
    def comprehensive_analysis(self):
        """综合分析报告"""
        print("\n" + "="*60)
        print("量化投资综合分析报告")
        print("="*60)
        
        # 1. 进行多年财务分析
        years = ['2014', '2015', '2016']
        for year in years:
            self.financial_analysis(year)
        
        # 2. 进行投资效率分析
        for year in years:
            self.investment_efficiency_analysis(year)
        
        # 3. 行业分析
        industries = self.sw_data['行业名称'].unique()[:5]  # 分析前5个行业
        industry_results = {}
        
        for industry in industries:
            result = self.industry_analysis(industry, '2014')
            if result:
                industry_results[industry] = result
        
        # 4. 生成投资建议
        print(f"\n{'='*20} 投资建议 {'='*20}")
        
        if industry_results:
            print("\n各行业推荐股票:")
            for industry, result in industry_results.items():
                if result['backtest_results']:
                    best_stock = max(result['backtest_results'], key=lambda x: x['total_return'])
                    print(f"{industry}: {best_stock['stock_code']} "
                          f"(预期收益: {best_stock['total_return']:.4f})")

        
        return industry_results
    
    def run_system(self):
        """运行整个系统"""
        print("欢迎使用量化投资分析系统！")
        print("系统功能：财务分析、投资效率分析、技术指标分析、策略回测")
        
        # 检查数据文件是否存在
        if not self.load_basic_data():
            print("数据加载失败，程序退出")
            return
        
        while True:
            print("\n" + "-"*40)
            print("请选择功能:")
            print("1. 财务数据分析")
            print("2. 投资效率分析") 
            print("3. 行业分析")
            print("4. 综合分析报告")
            print("5. 退出系统")
            
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                year = input("请输入分析年份 (2014/2015/2016): ").strip() or '2014'
                self.financial_analysis(year)
                
            elif choice == '2':
                year = input("请输入分析年份 (2014/2015/2016): ").strip() or '2014'
                self.investment_efficiency_analysis(year)
                
            elif choice == '3':
                print("可选行业:", ', '.join(self.sw_data['行业名称'].unique()))
                industry = input("请输入行业名称: ").strip()
                year = input("请输入分析年份 (2014/2015/2016): ").strip() or '2014'
                if industry:
                    self.industry_analysis(industry, year)
                
            elif choice == '4':
                self.comprehensive_analysis()
                
            elif choice == '5':
                print("感谢使用量化投资分析系统！")
                break
                
            else:
                print("无效选择，请重新输入")

def main():
    """主函数"""
    system = QuantitativeAnalysisSystem()
    system.run_system()

if __name__ == "__main__":
    main()

