import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd as xd

# 数据预处理
# 训练集
file_train_path = 'data/iris_train.xlsx'
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
            if file_train_sheet.cell_value(row, col) == 'Iris-setosa':
                y_train.append(0)
            elif file_train_sheet.cell_value(row, col) == 'Iris-versicolor':
                y_train.append(1)
            else:
                y_train.append(2)
    x_train.append(list(x_data))
# 测试集
file_test_path = 'data/iris_test.xlsx'
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
            if file_test_sheet.cell_value(row, col) == 'Iris-setosa':
                y_test.append(0)
            elif file_test_sheet.cell_value(row, col) == 'Iris-versicolor':
                y_test.append(1)
            else:
                y_test.append(2)
    x_test.append(list(x_data))
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# 将特征值的类型转换为tensor类型，避免后面的矩阵乘法报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 将特征值和目标值一一配对 并且每4组数据为一个batch，喂入神经网络的数据以batch为单位
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(4)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(4)
# print(test_data)

# 初始化梯度和偏置
# 由于输入的特征为4个，目标值是三分类，所以我们将 梯度 随机初始化为 四行三列的tensor
w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
# 同理，我们的目标值是一维数据，所以将 偏置 初始化为随机的1维tensor
b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

# 设置学习率
learn_rate = 0.1
# 梯度下降次数
epoch = 500
# 每轮分4个step，loss_all记录四个step生成的4个loss的和
loss_all = 0
# 绘图相关数据
train_loss = []
test_acc = []

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w) + b
            # 使输出符合概率分布
            y = tf.nn.softmax(y)
            # 将目标值转换为独热编码，方便计算loss和acc
            y_true = tf.one_hot(y_train, depth=3)
            # 回归性能评估采用MSE
            loss = tf.reduce_mean(tf.square(y_true - y))
            # print(loss)
            loss_all += loss.numpy()
        # 对每个梯度和偏置求偏导
        grads = tape.gradient(loss, [w, b])
        # 梯度自更新，这两行代码相当于:
        # w = w - lr * w_grads
        # b = b - lr * b_grads
        w.assign_sub(learn_rate * grads[0])
        b.assign_sub(learn_rate * grads[1])
    print(f"第{epoch}轮,损失是:{loss_all / 4}")
    train_loss.append(loss_all / 4)
    # loss_all归零，为记录下一个epoch的loss做准备
    loss_all = 0

    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_data:
        y = tf.matmul(x_test, w) + b
        y = tf.nn.softmax(y)
        # 返回最大值所在的索引，即预测的分类
        y_pred = tf.argmax(y, axis=1)
        # print(y_pred)
        y_pred = tf.cast(y_pred, dtype=y_test.dtype)
        # 预测正确为1，错误为0
        correct = tf.cast(tf.equal(y_pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    accuracy_rate = total_correct / total_number
    test_acc.append(accuracy_rate)
    print("测试集的准确率为:", accuracy_rate)
    print("---------------------------------")

# 绘制图像
# 训练集上的损失
plt.figure(figsize=(6, 8))
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.plot(train_loss, marker='.', color='r', linestyle='--', label="loss")
plt.legend(loc="best")
plt.show()
# 测试集的准确率
plt.figure(figsize=(10, 8))
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.plot(test_acc, marker='.', color='r', linestyle='--', label="accuracy")
plt.legend(loc="best")
plt.show()
