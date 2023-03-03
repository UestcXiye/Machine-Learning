import matplotlib.pyplot as plt
import numpy as np
import xlrd as xd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def LoadData(trainpath, testpath):
    file_train_path = trainpath
    file_train_xlsx = xd.open_workbook(file_train_path)
    file_train_sheet = file_train_xlsx.sheet_by_name('Sheet1')
    x_train = []
    y_train = []
    for row in range(file_train_sheet.nrows):
        x_data = []
        for col in range(file_train_sheet.ncols):
            if col < file_train_sheet.ncols - 1:
                x_data.append(file_train_sheet.cell_value(row, col))
            else:
                if file_train_sheet.cell_value(row, col) == 'tested_negative':
                    y_train.append(0)
                else:
                    y_train.append(1)
        x_train.append(list(x_data))

    file_test_path = testpath
    file_test_xlsx = xd.open_workbook(file_test_path)
    file_test_sheet = file_test_xlsx.sheet_by_name('Sheet1')
    x_test = []
    y_test = []
    for row in range(file_test_sheet.nrows):
        x_data = []
        for col in range(file_test_sheet.ncols):
            if col < file_test_sheet.ncols - 1:
                x_data.append(file_test_sheet.cell_value(row, col))
            else:
                if file_test_sheet.cell_value(row, col) == 'tested_negative':
                    y_test.append(0)
                else:
                    y_test.append(1)

        x_test.append(list(x_data))

    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)
    return x_train, y_train, x_test, y_test


# 数据集路径
train_path = 'data/diabetes_train.xlsx'
test_path = 'data/diabetes_test.xlsx'
# 加载数据
x_train, y_train, x_test, y_test = LoadData(train_path, test_path)
# 分别初始化对特征值和目标值的标准化器
ss_x = StandardScaler()
ss_y = StandardScaler()
# 训练数据都是数值型，所以要标准化处理
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.fit_transform(x_test)
# 目标数据也是数值型，所以也要标准化处理
y_train = ss_y.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = ss_y.fit_transform(np.array(y_test).reshape(-1, 1))
# 使用LinearSVC分类训练
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train.astype('int'))
linear_svc_predict = linear_svc.predict(x_test)
# 绘图
l1, = plt.plot(y_test, color='b', linewidth=2)
l2, = plt.plot(linear_svc_predict, color='r', linewidth=2)
plt.legend([l1, l2], ['y_test', 'linear_svc_predict'], loc=2)
plt.savefig('支持向量机回归预测.jpg')
plt.show()
# 性能评估
print('The Accuracy of Linear SVC is', linear_svc.score(x_test, y_test.astype(np.int64)))
