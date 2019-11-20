# -*-coding: utf-8 -*-
"""
    @Project: 聚类算法：最大值-最小值距离
    @Author : Alex
    @Date   : 2019-11-20 11:24:30
"""
import math
import numpy as np


def calcuDistance(data1, data2):
    '''
    计算两个模式样本之间的欧式距离
    :param data1:
    :param data2:
    :return:
    '''
    distance = 0
    for i in range(len(data1)):
        distance += pow((data1[i] - data2[i]), 2)
    return math.sqrt(distance)


def maxmin_distance_cluster(data, Theta):
    '''
    :param data: 输入样本数据,每行一个特征
    :param Theta:阈值，一般设置为0.5，阈值越小聚类中心越多
    :return:样本分类，聚类中心
    '''
    maxDistance = 0
    start = 0  # 初始选一个中心点
    index = start  # 相当于指针指示新中心点的位置
    k = 0  # 中心点计数，也即是类别

    dataNum = len(data)     # 样本数
    distance = np.zeros((dataNum,))
    minDistance = np.zeros((dataNum,))
    classes = np.zeros((dataNum,))
    centerIndex = [index]

    # 初始选择第一个为聚类中心点
    ptrCen = data[0]
    # 寻找第二个聚类中心，即与第一个聚类中心最大距离的样本点
    for i in range(dataNum):
        ptr1 = data[i]
        d = calcuDistance(ptr1, ptrCen)
        distance[i] = d
        classes[i] = k + 1
        if (maxDistance < d):
            maxDistance = d
            index = i  # 与第一个聚类中心距离最大的样本

    minDistance = distance.copy()
    maxVal = maxDistance
    while maxVal > (maxDistance * Theta):
        k = k + 1
        centerIndex += [index]  # 新的聚类中心
        for i in range(dataNum):
            ptr1 = data[i]
            ptrCen = data[centerIndex[k]]
            d = calcuDistance(ptr1, ptrCen)
            distance[i] = d
            # 按照当前最近临方式分类，哪个近就分哪个类别
            if minDistance[i] > distance[i]:
                minDistance[i] = distance[i]
                classes[i] = k + 1
        # 寻找minDistance中的最大距离，若maxVal > (maxDistance * Theta)，则说明存在下一个聚类中心
        index = np.argmax(minDistance)
        maxVal = minDistance[index]
    return classes, centerIndex


if __name__ == '__main__':
    data = [[0, 0, 0, 0], [30, 8, 15, 20], [92, 92, 94, 93], [91, 91, 87, 88], [95, 93, 89, 98], [64, 68, 60, 70], [46, 33, 37, 78], [95, 93, 89, 98], [94, 88, 96, 88], [66, 63, 64, 78]]
    Theta = 0.5
    classes, centerIndex = maxmin_distance_cluster(data, Theta)
    print(classes)

    print(centerIndex)

