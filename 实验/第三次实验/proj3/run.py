import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xlrd as xd
from sklearn.preprocessing import Normalizer

from BP import BP


def LoadData(trainpath, testpath):
    file_train_path = trainpath
    file_train_xlsx = xd.open_workbook(file_train_path)
    file_train_sheet = file_train_xlsx.sheet_by_name('Sheet1')
    x_train = []
    y_train = []
    for row in range(file_train_sheet.nrows):
        x_data = []
        for col in range(file_train_sheet.ncols):
            if col == 0:
                if file_train_sheet.cell_value(row, col) == 'M':
                    y_train.append(1)
                elif file_train_sheet.cell_value(row, col) == 'F':
                    y_train.append(-1)
                else:
                    y_train.append(0)
            else:
                x_data.append(file_train_sheet.cell_value(row, col))

        x_train.append(list(x_data))

    file_test_path = testpath
    file_test_xlsx = xd.open_workbook(file_test_path)
    file_test_sheet = file_test_xlsx.sheet_by_name('Sheet1')
    x_test = []
    y_test = []
    for row in range(file_test_sheet.nrows):
        x_data = []
        for col in range(file_test_sheet.ncols):
            if col == 0:
                if file_test_sheet.cell_value(row, col) == 'M':
                    y_test.append(1)
                elif file_test_sheet.cell_value(row, col) == 'F':
                    y_test.append(-1)
                else:
                    y_test.append(0)
            else:
                x_data.append(file_test_sheet.cell_value(row, col))

        x_test.append(list(x_data))

    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    # 将特征值的类型转换为tensor类型，避免后面的矩阵乘法报错
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    return x_train, x_test, y_train, y_test


def run_main():
    """
       这是主函数
    """
    # 导入数据
    trainpath = 'data/abalone_train.xlsx'
    testpath = 'data/abalone_test.xlsx'
    Train_Data, Test_Data, Train_Label, Test_Label = LoadData(trainpath, testpath)
    Train_Data = Normalizer().fit_transform(Train_Data)
    Test_Data = Normalizer().fit_transform(Test_Data)

    # 设置网络参数
    input_n = np.shape(Train_Data)[1] + np.shape(Test_Data)[1]
    output_n = np.shape(Train_Label)[1] + np.shape(Test_Label)[1]
    hidden_n = int(np.sqrt(input_n * output_n))
    lambd = 0.001
    batch_size = 64
    learn_rate = 0.01
    epoch = 1000
    iteration = 10000

    # 训练并测试网络
    bp = BP(input_n, hidden_n, output_n, lambd)
    train_loss, test_loss, test_accuracy = bp.train_test(Train_Data, Train_Label, Test_Data, Test_Label, learn_rate,
                                                         epoch, iteration, batch_size)

    # 解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 结果可视化
    col = ['Train_Loss', 'Test_Loss']
    epoch = np.arange(epoch)
    plt.plot(epoch, train_loss, 'r')
    plt.plot(epoch, test_loss, 'b-.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(labels=col, loc='best')
    plt.savefig('./训练与测试损失.jpg')
    plt.show()
    plt.close()

    plt.plot(epoch, test_accuracy, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig('./测试精度.jpg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    run_main()
