# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 19:58
# @Author  : Shiqi Wang
# @FileName: plot_curve.py
from time import strftime, localtime, time

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_curve(self, result):
    # 多条折线图
    plt.figure()
    title = "epoch-reward"
    plt.title(title)
    plt.grid()  # 显示网格
    dict_len = len(result[1])
    x = list(range(dict_len))
    plt.xticks(np.arange(0, dict_len, 1))  # 设置横坐标轴的刻度
    for i in range(len(result)):
        y = result[i]
        x_smooth = np.linspace(0, len(x), 50)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        # plt.plot(x_smooth, y_smooth, marker='p', linestyle="-", label='epoch' + str(i))
        plt.plot(x, y, marker='*', linestyle="-", label='iters' + str(i))
    plt.xlabel('times')  # 设置X轴标签
    plt.ylabel('reward')  # 设置Y轴标签
    # plt.legend()  # 显示图例，即每条线对应 label 中的内容
    currentTime = strftime('%Y-%m-%d %H:%M:%S', localtime(time()))
    filename = "curve&" + "@" + currentTime
    filepath = self.output['-dir'] + 'IncreasedCnt/'
    plt.savefig(filepath + filename)
    plt.show()


def plot_curve2(self, result, title):
    # 折线图
    plt.figure()
    title = title
    plt.title(title)
    plt.grid()  # 显示网格
    plt.plot(result, marker='*', linestyle="-")
    plt.xlabel('times')  # 设置X轴标签
    plt.ylabel('reward')  # 设置Y轴标签
    currentTime = strftime("%Y-%m-%d-%H", localtime(time()))
    filename = "avgcurve&" + "@" + currentTime
    filepath = self.output['-dir'] + 'IncreasedCnt/'
    plt.savefig(filepath + filename)
    plt.show()


def plot_fig(self, result):
    # 创建窗口,散点图
    plt.figure()
    plt.grid()  # 显示网格
    title = 'random insert'
    # title = str(self.insert_item_num) + str(self.itemType) + "&500"  + str(self.userType)
    plt.title(title)
    plt.xlabel('iters')  # 设置X轴标签
    plt.ylabel('increased_cnt')  # 设置Y轴标签
    x = list(range(0, len(result)))
    plt.scatter(x, result, marker='.', c='k')
    # plt.scatter(result[-1, 0], result[-1, 1], marker='x', c='r')
    currentTime = strftime("'%Y-%m-%d %H:%M:%S'", localtime(time()))
    filename = "randominsert&" + "@" + currentTime
    filepath = self.output['-dir'] + 'IncreasedCnt/'
    plt.savefig(filepath + filename)
    plt.show()