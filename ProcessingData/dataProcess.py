import sys
from collections import Counter

sys.path.append("..")
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import pandas as pd

class DataProcess():
    def __init__(self, config):
        self.config = config
        # promoted items
        self.targetItem = self.config['ENV']['TARGET_ITEM_LIST'].split(', ')
        self.output = self.config['ENV']['OUT_PUT']

        self.converted_filePath = self.config['ENV']['ORIGN_RATING_FILE']
        self.train_filePath = self.config['ENV']['TRAIN_RATING_FILE']
        self.test_filePath = self.config['ENV']['TEST_RATING_FILE']
        self.binarized = self.config['META']['BINARIZED']

        # 划分数据集
        self.trainingData, self.rating, self.testData= self.loadDataSet()
        # 构建稀疏矩阵
        self.sparseMatrix = self.constructMatrix(self.rating)

    def loadDataSet(self): # 读取文件
        col_names = ['userid', 'itemid','rating']
        complete_ratings_df = pd.read_csv(self.converted_filePath, names=col_names, usecols=[0,1,2])
        train_ratings_df = pd.read_csv(self.train_filePath, names=col_names, usecols=[0,1,2])
        test_ratings_df = pd.read_csv(self.test_filePath,  names=col_names, usecols=[0,1,2])

        self.user_num = complete_ratings_df['userid'].nunique()
        self.item_num = complete_ratings_df['itemid'].nunique()
        self.rating_num = len(complete_ratings_df)

        # rating是评分没有二值化的df
        # rating = complete_ratings_df.values.tolist()
        # if self.binarized == '1':
        #     complete_ratings_df.loc[:, 'rating'] = np.where(complete_ratings_df.loc[:, 'rating'] >= 1, 1, 0)
        #     complete_ratings_df.loc[:, 'rating'] = np.where(complete_ratings_df.loc[:, 'rating'] >= 1, 1, 0)

        # train_copy.sort_index(axis=0, ascending=True, inplace=True)
        # test_copy.sort_index(axis=0, ascending=True, inplace=True)
        trainingData = train_ratings_df.values.tolist()
        testData = test_ratings_df.values.tolist()
        return trainingData, complete_ratings_df, testData

        # 用户物品矩阵
    def constructMatrix(self, data):
        self.max_num_user = data['userid'].max()+1
        self.max_num_item = data['itemid'].max()+1
        sparseMatrix = csr_matrix((np.ones(len(data)),(data['userid'], data['itemid'])),
            shape=(self.max_num_user, self.max_num_item)).todok()
        self.rating_sparse = csr_matrix((data['rating'],(data['userid'], data['itemid'])),
            shape=(self.max_num_user, self.max_num_item)).todok()
        return sparseMatrix

    def item_activity_ranks(self):
        item_sum = self.sparseMatrix.toarray().sum(axis=0)
        count = dict(zip(list(range(self.item_num)), item_sum))
        sorted_count = sorted(count.items(),key=lambda x:x[1],reverse=True)
        return sorted_count

    def user_activity_ranks(self):
        user_sum = self.sparseMatrix.toarray().sum(axis=1)
        count = dict(zip(list(range(self.user_num)), user_sum))
        sorted_count = sorted(count.items(),key=lambda x:x[1],reverse=True)
        return sorted_count

    def user_rating_ranks(self):
        user_sum_rating = self.rating_sparse.toarray().sum(axis=1)
        user_rated_sum = self.sparseMatrix.toarray().sum(axis=1)
        average_rating = np.array(user_sum_rating) / np.array(user_rated_sum)
        count = dict(zip(list(range(self.user_num)), average_rating))
        sorted_count = sorted(count.items(),key=lambda x:x[1],reverse=True)
        return sorted_count