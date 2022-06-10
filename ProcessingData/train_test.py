# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 23:00
# @Author  : Shiqi Wang
# @FileName: train_test.py
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split


def split(df):
    x_train, x_test = train_test_split(df, train_size=0.8)
    x_train.sort_index(axis=0, ascending=True, inplace=True)
    x_test.sort_index(axis=0, ascending=True, inplace=True)
    # trainingData = x_train.values.tolist()
    # testData = x_test.values.tolist()
    return x_train, x_test

DATAPATH = '../dataset'
dataset = 'ml-100k'
train_ratings_df = pd.read_csv(os.path.join(DATAPATH,dataset,'data_converted'),header=None,names =(['user_id', 'item_id', 'rating']))
trainingData,testData = split(train_ratings_df)
# save
trainingData.to_csv(os.path.join(DATAPATH,dataset,'train_dataset.csv'),header=False,index=False)
testData.to_csv(os.path.join(DATAPATH,dataset,'test_dataset.csv'),header=False,index=False)
print("done")