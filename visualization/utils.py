# -*- coding: utf-8 -*-


import os
import re
import pandas as pd
import json
from collections import OrderedDict


def create_dir(create_dirs):
    """
    创建所需要的目录
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))


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
    pattern_message = re.compile('"message": "(.+)"')
    pattern_array = re.compile("array\((.+?)\)")

    pattern_tau = re.compile('"tau": (.+),')
    pattern_read = re.compile('"read_message": "(.+)"')

    dfs = {}
    infos = {}
    df = pd.DataFrame()
    for filename in filenames:
        # if filename == ".DS_Store":
        #     continue
        if filename[0] == '.' or filename[0] == '_':
            continue
        df = pd.DataFrame()
        message = "None"
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            start = False
            add = 0
            info_extra = {'tau':0, 'read':""}
            for i, line in enumerate(lines):
                res_tau = re.search(pattern_tau, line)
                if res_tau:
                    info_extra['tau'] = res_tau.group(1)
                res_read = re.search(pattern_read, line)
                if res_read:
                    info_extra['read'] = res_read.group(1)

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
                res_message = re.search(pattern_message, line)
                if res_message:
                    message = res_message.group(1)

            if args.use_filename == "Yes":
                message = filename[:-4]

            df.rename(
                columns={"RL_val_trajectory_reward": "R_tra",
                         "RL_val_trajectory_len": 'len_tra',
                         "RL_val_CTR": 'ctr'},
                inplace=True)
            # print("JJ", filename)
            df = df[["R_tra","len_tra","ctr"]]

        dfs[message] = df
        infos[message] = info_extra

    dfs = OrderedDict(sorted(dfs.items(), key=lambda item: len(item[1]), reverse=True))

    indices = [list(dfs.keys()), df.columns.to_list()]

    df_all = pd.DataFrame(columns=pd.MultiIndex.from_product(indices, names=["Exp", "metrics"]))

    for message, df in dfs.items():
        # print(message, df)
        for col in df.columns:
            df_all[message, col] = df[col]

    # # Rename MultiIndex columns in Pandas
    # # https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    # df_all.rename(
    #     columns={"RL_val_trajectory_reward": "R_tra", "RL_val_trajectory_len": 'len_tra', "RL_val_CTR": 'ctr'},
    #     level=1,inplace=True)

    # change order of levels
    # https://stackoverflow.com/questions/29859296/how-do-i-change-order-grouping-level-of-pandas-multiindex-columns
    df_all.columns = df_all.columns.swaplevel(0, 1)
    df_all.sort_index(axis=1, level=0, inplace=True)

    if is_info:
        return df_all, infos

    return df_all
