import pandas as pd
import statsmodels.api as sm

data = pd.read_excel('data.xlsx')
data.columns = ['y', 'x1', 'x2', 'x3', 'x4']
# 生成自变量
x = sm.add_constant(data.iloc[:, 1:])
# 生成因变量
y = data['y']
# 生成模型
model = sm.OLS(y, x)
# 模型拟合
result = model.fit()
# 模型描述
print(result.summary())
