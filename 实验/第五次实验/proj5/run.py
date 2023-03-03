import random

import matplotlib.pyplot as plt
import numpy as np

# 参数设置
# 簇数
k = 5
# 迭代次数
round = 0
# 最大迭代次数
ROUND_LIMIT = 50
# 均值向量变化量下界
THRESHOLD = 1e-10
# 顶点列表
points = []
# 簇列表
clusters = []
# 读取数据
f = open('data/K-means.txt', 'r')
for line in f:
    points.append(np.array(line.split(','), dtype=np.string_).astype(np.float64))
# print(points)

# 随机取k个不重复的顶点作为初始质心
mean_vectors = random.sample(points, k)
# K-means算法
while True:
    # 迭代次数自增
    round += 1
    # 初始化均值向量变化量
    change = 0
    # 清空对簇的划分
    clusters = []
    for i in range(k):
        clusters.append([])

    for point in points:
        '''
        argmin函数找出容器中最小的下标，在这里这个目标容器是
        list(map(lambda vec: np.linalg.norm(melon - vec, ord = 2), mean_vectors)),
        它表示melon与mean_vectors中所有向量的距离列表。
        (numpy.linalg.norm计算向量的范数,ord = 2即欧几里得范数，或模长)
        '''
        c = np.argmin(
            list(map(lambda vec: np.linalg.norm(point - vec, ord=2), mean_vectors))
        )
        clusters[c].append(point)

    for i in range(k):
        # 求每个簇的新均值向量
        new_vector = np.zeros((1, 2))
        for point in clusters[i]:
            new_vector += point
        new_vector /= len(clusters[i])
        # 累加改变幅度并更新均值向量
        change += np.linalg.norm(mean_vectors[i] - new_vector, ord=2)
        mean_vectors[i] = new_vector
    # 当均值向量变化量足够小或达到最大迭代量时，退出K-means算法
    if round > ROUND_LIMIT or change < THRESHOLD:
        break

# 打印结果
print('共迭代%d轮。' % round)

for i, cluster in zip(range(k), clusters):
    # 求各簇的质心
    centroid = []
    x_mean = 0.0
    y_mean = 0.0
    for point in cluster:
        x_mean += point[0]
        y_mean += point[1]
    x_mean /= len(cluster)
    y_mean /= len(cluster)
    centroid.append(x_mean)
    centroid.append(y_mean)
    print('簇%d的质心为：' % (i + 1), centroid)
# 绘图
colors = ['red', 'green', 'blue', 'yellow', 'black']
# 每个簇换一下颜色，同时迭代簇和颜色两个列表
for i, color in zip(range(k), colors):
    for point in clusters[i]:
        # 绘制散点图
        plt.scatter(point[0], point[1], color=color)
plt.savefig('聚类结果.jpg')
plt.show()
