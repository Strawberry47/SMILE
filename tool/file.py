import os.path
from os import makedirs,remove
from re import compile,findall,split
from .config import LineConfig
from collections import defaultdict
import numpy as np
# from insert.insertion import Insert

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def continueWrite(dir,content,op = 'a'):
        with open(dir,op) as f:
            res = content + "\n"
            f.writelines(res)



    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)

    @staticmethod
    def constructDataset(conf,filePath,threshold = 1.0):
        user2id = {}
        item2id = {}
        id2user = {}
        id2item = {}
        data = []
        ratingConfig = LineConfig(conf['ratings.setup'])
        # 处理数据集
        print("开始处理数据")
        with open(filePath) as f:
            ratings = f.readlines()
            delim = ' |,|\t'
            for lineNo, line in enumerate(ratings):
                items = split(delim, line.strip())
                username = items[0]
                itemname = items[1]
                rating = items[2]
                data.append([username, itemname, rating])
                # 避免重复
                if username not in user2id:
                    # 用户名-用户位置
                    user2id[username] = len(user2id)
                    # 用户位置-用户名
                    id2user[user2id[username]] = username
                if itemname not in item2id:
                    item2id[itemname] = len(item2id)
                    id2item[item2id[itemname]] = itemname
        num_user = len(user2id)
        num_item = len(item2id)
        dataset = np.zeros((num_user, num_item))
        for line in data:
            id_user = user2id[line[0]]
            id_item = item2id[line[1]]
            rating = float(line[2])
            if rating > threshold:  # 1, 2, 3, 4
                dataset[id_user][id_item] = 1
        print("数据集矩阵完成")
        return dataset

    # @staticmethod
    # 生成trainingdata
    # def loadDataSet(dataSet,conf, file, bTest=False,binarized = False, threshold = 1.0):
    #     trainingData = []
    #     dataSet = Insert(conf)
    #     trainingData = dataSet.data
    #     return trainingData
    #

    @staticmethod
    def loadDataSet(conf, file, bTest=False, binarized=False, threshold=3.0):
        trainingData = []
        testData = []
        ratingConfig = LineConfig(conf['ratings.setup'])
        binarized = conf['binarized']
        if not bTest:
            print('loading training data...')
        else:
            print('loading test data...')
        with open(file) as f:
            ratings = f.readlines()

        order = ratingConfig['-columns'].strip().split()
        delim = ' |,|\t'  # 按空格、逗号、制表符分割？
        if ratingConfig.contains('-delim'):
            delim = ratingConfig['-delim']
        for lineNo, line in enumerate(ratings):
            items = split(delim, line.strip())
            if not bTest and len(order) < 2:
                print('The rating file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            try:
                userId = items[int(order[0])]
                itemId = items[int(order[1])]
                if len(order) < 3:
                    rating = 1  # default value
                else:
                    rating = items[int(order[2])]
                if binarized == '1':
                    if float(items[int(order[2])]) < threshold:
                        continue
                    else:
                        rating = 1
            except ValueError:
                print('Error! Have you added the option -header to the rating.setup?')
                exit(-1)
            if not bTest:
                trainingData.append([userId, itemId, float(rating)])
            else:
                # if binarized:
                #     if rating == 1:
                #         testData.append([userId, itemId, float(rating)])
                #     else:
                #         continue
                testData.append([userId, itemId, float(rating)])
        if not bTest:
            return trainingData
        else:
            return testData

    @staticmethod
    def loadRelationship(conf, filePath):
        socialConfig = LineConfig(conf['social.setup'])
        relation = []
        print('loading social data...')
        with open(filePath) as f:
            relations = f.readlines()
            # ignore the headline
        if socialConfig.contains('-header'):
            relations = relations[1:]
        # order of the columns
        order = socialConfig['-columns'].strip().split()
        if len(order) <= 2:
            print('The social file is not in a correct format.')
        for lineNo, line in enumerate(relations):
            items = split(' |,|\t', line.strip())
            if len(order) < 2:
                print('The social file is not in a correct format. Error: Line num %d' % lineNo)
                exit(-1)
            userId1 = items[int(order[0])]
            userId2 = items[int(order[1])]
            if len(order) < 3:
                weight = 1
            else:
                weight = float(items[int(order[2])])
            relation.append([userId1, userId2, weight])
        return relation




