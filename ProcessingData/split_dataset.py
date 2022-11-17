import configparser
import os
import sys

from sklearn.model_selection import train_test_split

from structure import new_sparseMatrix

sys.path.append("..")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def loadDataSet(filePath):
    col_names = ['username', 'itemname', 'rating']
    ratings_df = pd.read_csv(filePath, names=col_names, usecols=[0, 1, 2], engine='python')
    ratings_df['username'] = ratings_df['username'].apply(str)
    ratings_df['itemname'] = ratings_df['itemname'].apply(str)
    ratings_df_copy = ratings_df.copy()
    return ratings_df_copy


def del_item(tem_data):
    trainingData = tem_data.values.tolist()
    sparseMatrix = new_sparseMatrix.SparseMatrix(trainingData)
    dict_i = sparseMatrix.matrix_Item
    del_res = []
    del_item = []
    for k, v in list(dict_i.items()):
        if (len(v) <= 3):
            del_item.append(k)
            del dict_i[k]

    for k, v in list(dict_i.items()):
        itemname = k
        for u, r in v.items():
            username = u
            rating = r
            del_res.append([username, itemname, rating])
    return del_res

def del_user(tem_data):
    sparseMatrix = new_sparseMatrix.SparseMatrix(tem_data)
    dict_u = sparseMatrix.matrix_User
    del_res = []
    del_item = []
    for k, v in list(dict_u.items()):
        if (len(v) <= 3):
            del_item.append(k)
            del dict_u[k]

    for k, v in list(dict_u.items()):
        username = k
        for i, r in v.items():
            itemname = i
            rating = r
            del_res.append([username, itemname, rating])
    return del_res

def split(ratings_df):
    df = pd.DataFrame(ratings_df, columns=['username', 'itemname', 'rating'])
    x_train, x_test = train_test_split(df, train_size=0.8)
    x_train.sort_index(axis=0, ascending=True, inplace=True)
    x_test.sort_index(axis=0, ascending=True, inplace=True)
    # trainingData = x_train.values.tolist()
    # testData = x_test.values.tolist()
    return x_train, x_test


config = configparser.ConfigParser()
_x = open('../config/config_Ciao')
config.read_file(_x)
train_flilePath = config['ENV']['ORIGN_RATING_FILE']
ratings_df_copy = loadDataSet(train_flilePath)
after_del_item = del_item(ratings_df_copy)
after_del_user = del_user(after_del_item)
trainingData,testData = split(after_del_user)

## save
trainingData.to_csv(config['ENV']['TRAIN_RATING_FILE'],header=False,index=False)
testData.to_csv(config['ENV']['TEST_RATING_FILE'],header=False,index=False)
print("ok")