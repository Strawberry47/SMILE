# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 16:20
# @Author  : Shiqi Wang
# @FileName: dataConvert.py
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# DATAPATH = '../dataset/ml-100k'
# DATAPATH = '../dataset/ml-1m'
DATAPATH = '../dataset/Ciao'

def load_mat():
    # data_path = '../dataset/ml-100k/u.data'
    # data_path = '../dataset/ml-1m/rating.dat'
    data_path = '../dataset/Ciao/ratings.txt'
    df_data = pd.read_csv(data_path, header = None, sep=' ', names =(['user_id', 'item_id', 'rating']),usecols=[0,1,2])

    lbe_user = LabelEncoder()
    lbe_user.fit(df_data['user_id'].unique())
    converted_user = lbe_user.transform(df_data['user_id'])

    lbe_item = LabelEncoder()  # 弄成离散的
    lbe_item.fit(df_data['item_id'].unique())
    converted_item = lbe_item.transform(df_data['item_id'])

    converted_data = pd.DataFrame()
    converted_data['user_id'] = converted_user
    converted_data['item_id'] = converted_item
    converted_data['rating'] = df_data['rating']

    # 对应关系
    user2id = {}
    for user in lbe_user.classes_:
        user2id.update({user: lbe_user.transform([user])[0]})

    item2id = {}
    for item in lbe_item.classes_:
        item2id.update({item: lbe_item.transform([item])[0]})

    return converted_data,user2id,item2id

def save(converted_data,user2id,item2id):
    sort = converted_data.sort_values(by=['user_id'])
    sort.to_csv(os.path.join(DATAPATH,'data_converted.csv'), header=None, index=False)
    np.save(os.path.join(DATAPATH,'user2id.npy'), user2id)
    np.save(os.path.join(DATAPATH,'item2id.npy'), item2id)
    print('successfully saved')


if __name__ == '__main__':
    converted_data,user2id,item2id = load_mat()
    save(converted_data,user2id,item2id)