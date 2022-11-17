# -*- coding: utf-8 -*-
# @Time    : 2021/9/13 10:25 上午
# @Author  : Chongming GAO
# @FileName: visual_RL.py

import argparse
import os
import re
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from utils import create_dir
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser()
    # --result_dir "./visualization/results/KuaishouEnv-v0"
    # movielens1M,Ciao
    parser.add_argument("--result_dir", type=str, default="./results/BPR-movielens100k/smile/logs")
    parser.add_argument("--use_filename", type=str, default="Yes")

    args = parser.parse_known_args()[0]
    return args


def walk_paths(result_dir):
    g = os.walk(result_dir)

    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name[0] == '.' or file_name[0] == '_':
                continue
            print(os.path.join(path, file_name))
            files.append(file_name)
    return files


def loaddata(dirpath, filenames, args, is_info=False):
    pattern_epoch = re.compile("Epoch: \[(\d+)]")
    pattern_info = re.compile("Info: \[(\{.+\})]")
    pattern_array = re.compile("array\((.+?)\)")

    dfs = {}
    df = pd.DataFrame()
    for filename in filenames:
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0

            for i, line in enumerate(lines):
                res = re.search(pattern_epoch, line)
                if res:
                    epoch = int(res.group(1))
                    if (start == False) and epoch == 0:
                        add = 1
                        start = True
                    epoch += add
                    info = re.search(pattern_info, line)
                    try:
                        info1 = info.group(1).replace("\'", "\"")
                    except Exception as e:
                        print("jump incomplete line: [{}]".format(line))
                        continue
                    info2 = re.sub(pattern_array, lambda x: x.group(1), info1)

                    data = json.loads(info2)
                    df_data = pd.DataFrame(data, index=[epoch],dtype=float)
                    df = df.append(df_data)

            if args.use_filename == "Yes":
                message = filename[:-4] # xxx.log

            df = df[["ave_rew","recnum"]]

        dfs[message] = df

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))

    indices = [list(dfs.keys()), df.columns.to_list()]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["method", "metrics"]))

    for message, df in dfs.items():
        df = df.sort_values(by=['ave_rew'])
        for col in df.columns:
            df1 = df.reset_index()
            df_all[message, col] = df1[col]

    df_all.columns = df_all.columns.swaplevel(0, 1)
    df_all.sort_index(axis=1, level=0, inplace=True)

    return df_all

def axis_shift(ax1 ,x_shift=0.01, y_shift=0):
    position = ax1.get_position().get_points()
    pos_new = position
    pos_new[:, 0] += x_shift
    pos_new[:, 1] += y_shift
    ax1.set_position(Bbox(pos_new))


def visual4(df1, df2, df3, save_fig_dir, savename="three"):
    visual_cols = ['ave_rew', 'recnum']

    df1 = df1.loc[:10].reset_index()
    df2 = df2.loc[:10].reset_index()
    df3 = df3.loc[:10].reset_index()

    dfs = [df1, df2, df3]
    series = "ABC"
    dataset = ["Movielens100k", "Movielens1M", "Ciao"]
    fontsize = 11.5

    all_method = sorted(set(df1['recnum'].columns.to_list()))

    colors = sns.color_palette(n_colors=len(all_method))
    markers = ["o", "s", "p", "P", "X", "*", "h", "D", "v", "^", ">", "<"]

    color_kv = dict(zip(all_method, colors))
    marker_kv = dict(zip(all_method, markers))

    fig = plt.figure(figsize=(12, 7))

    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    axs = []

    for index in range(len(dfs)):
        alpha=series[index]
        cnt = 1
        df = dfs[index]
        data_ave = df[visual_cols[0]]
        data_recnum = df[visual_cols[1]]

        color = [color_kv[name] for name in data_ave.columns]
        marker = [marker_kv[name] for name in data_ave.columns]

        ax1 = plt.subplot2grid((2,3), (0,index))
        data_ave.plot(kind="line", linewidth=1, ax=ax1, legend=None, color=color, markevery=int(len(data_ave)/10), fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax1.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        # ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        ax1.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, loc="left", x=0.4, y=.97)
        ax1.set_title("{}".format(dataset[index]), fontsize=fontsize, y=1.1, fontweight=700)
        ax1.set_xlabel("epoch", fontsize=11)
        cnt += 1

        ax2 = plt.subplot2grid((2,3), (1,index))
        data_recnum.plot(kind="line", linewidth=1, ax=ax2, legend=None, color=color, markevery=int(len(data_ave)/10), fillstyle='none', alpha=.8, markersize=3)
        for i, line in enumerate(ax2.get_lines()):
            line.set_marker(marker[i])
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
        plt.grid(linestyle='dashdot', linewidth=0.8)
        ax2.set_title("({}{})".format(alpha, cnt), fontsize=fontsize, y=.97)
        cnt += 1
        ax2.set_xlabel("epoch", fontsize=11)

        if index == 1:
            axis_shift(ax1, .015)
            axis_shift(ax2, .015)

        if index == 2:
            axis_shift(ax1, .005)
            axis_shift(ax2, .005)

        axs.append((ax1, ax2))

    ax1, ax2 = axs[0]
    ax1.set_ylabel("Single-item reward", fontsize=10, fontweight=700)
    ax2.set_ylabel("Final exposure", fontsize=10, fontweight=700)
    ax2.yaxis.set_label_coords(-0.17, 0.5)


    lines1, labels1 = ax1.get_legend_handles_labels()
    dict_label = dict(zip(labels1, lines1))
    dict_label = OrderedDict(sorted(dict_label.items(), key=lambda x: x[0]))
    dict_label = {k :v for k,v in dict_label.items()}

    ax1.legend(handles=dict_label.values(), labels=dict_label.keys(), ncol=8,
               loc='lower left', columnspacing=0.7,
               bbox_to_anchor=(-0.20, 2.24), fontsize=10.5)

    fig.savefig(os.path.join(save_fig_dir, savename + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)


def main(args):

    realpath = os.path.dirname(__file__)
    save_fig_dir = os.path.join(realpath, "figures")

    create_dirs = [save_fig_dir]
    create_dir(create_dirs)
    # result_dir1 = "./results/kuaishou zo 30"
    result_dir1 = "./results/movielens100k"
    filenames = walk_paths(result_dir1)
    df1 = loaddata(result_dir1, filenames, args)

    result_dir2 = "./results/movielens1m"
    filenames = walk_paths(result_dir2)
    df2 = loaddata(result_dir2, filenames, args)

    result_dir3 = "./results/Ciao"
    filenames = walk_paths(result_dir3)
    df3 = loaddata(result_dir3, filenames, args)

    visual4(df1, df2, df3, save_fig_dir, savename="main_result")


if __name__ == '__main__':
    args = get_args()
    main(args)
