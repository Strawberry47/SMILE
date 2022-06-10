# insert 类
import sys
sys.path.append("..")
from re import compile,findall,split
from collections import defaultdict
from os import makedirs,remove
from tool.file import FileIO
from tool.config import Config,LineConfig
import numpy as np
from random import choice,random,shuffle,randint
import matplotlib.pyplot as plt # 图形工具
from time import strftime,localtime,time
from scipy.interpolate import make_interp_spline

class Insert():
    def __init__(self,orign_data,config,itemType = False,userType = False,itemName = False,numk = 0 ):
        self.config = config
        self.iters = 20 # 随机用户循环次数
        self.num_user = 0
        self.num_item = 0
        self.user2id = {}
        self.id2user = {}
        self.item2id = {}
        self.id2item = {}
        # self.insert_item_num = 5
        self.insert_item_num = 5 # 注入热门物品的数量
        self.itemType = itemType
        self.userType = userType
        self.trainSet_u = defaultdict(dict)
        self. trainSet_i = defaultdict(dict)
        self.output = LineConfig(self.config['output.setup']) # 输出位置
        self.ranking = LineConfig(self.config['item.ranking']) # topn
        # self.k = self.config['num.k'] # 选择注入的条数/用户数
        self.k = numk
        self.binarized = self.config['binarized']
        filePath = config['ratings']
        if (orign_data != None):
            self.data = orign_data
        else:
            self.data = self.loadDataSet(filePath)
        self.dataMatrix = self.constructMatrix(self.data)

        if itemType == "customized" :
            self.insertItem = itemName
            # 判断物品种类
        elif  self.itemType == "hotItem":
            self.insertItem = self.findHotItem()
        elif self.itemType == "coldItem":
            self.insertItem = self.findColdItem()
        elif self.itemType == "randomItem":
            self.insertItem = self.findRandomItem()

        # 判断用户种类
        if  self.userType == "hotUser":
            self.userList = self.findActiveUserList()
        elif self.userType == "coldUser":
            self.userList = self.findQuietUserList()
        elif self.userType == "generalUser":
            self.userList = self.findGeneralUserList()
        elif self.userType == "randomUser":
            self.userList = self.findRandomUserList()




    def loadDataSet(self,filePath):
        data = []
        # 处理数据集
        with open(filePath) as f:
            ratings = f.readlines()
            delim = ' |,|\t'
            for lineNo, line in enumerate(ratings):
                items = split(delim, line.strip())
                username = items[0]
                itemname = items[1]
                rating = items[2]
                if self.binarized == '1':
                    if float(rating) > 0:
                        rating = 1
                data.append([username, itemname, float(rating)])
        trainingData = data
        print("Data processing completed.")
        return trainingData

   # 用户物品矩阵
    def constructMatrix(self,data):
        tem_data = data[:]
        # 统计人数
        for lines in tem_data:
            username = lines[0]
            itemname = lines[1]
            rating = lines[2]
            # 避免重复
            if username not in self.user2id:
                # 用户名-用户位置
                self.user2id[username] = len(self.user2id)
                # 用户位置-用户名
                self.id2user[self.user2id[username]] = username
            if itemname not in self.item2id:
                self.item2id[itemname] = len(self.item2id)
                self.id2item[self.item2id[itemname]] = itemname

            self.trainSet_u[username][itemname] = rating
            self.trainSet_i[itemname][username] = rating

        self.num_user = len(self.user2id)
        self.num_item = len(self.item2id)

        dataset = np.zeros((self.num_user, self.num_item))
        print("user count: ", self.num_user)
        print("item count: ", self.num_item)
        print("rating count: ",str(len(tem_data)))
        for line in tem_data:
            id_user = self.user2id[line[0]]
            id_item = self.item2id[line[1]]
            rating = float(line[2])
            if rating > 0:  # 1, 2, 3, 4
                dataset[id_user][id_item] = 1
        print("Data matrix generation completed.")
        return dataset


    #热门物品
    def findHotItem(self):
        num = int(self.insert_item_num)
        links_item = np.zeros((self.num_item,))
        hot_item_list = np.zeros((num,), dtype=np.int64)
        for i in range(self.num_item):
            links_item[i] = list(self.dataMatrix[:,i]).count(1)
        indices = np.argsort(-links_item)
        n = 0

        for id in indices:
            itemname = self.id2item[id]
            hot_item_list[n] = itemname
            n += 1
            if n >= num:
                break
        print("The "+str(num) +" most popular"  + " items:")
        print(hot_item_list)
        return hot_item_list


    # 寻找最冷门的5物品
    def findColdItem(self):
        num = int(self.insert_item_num)
        links_item = np.zeros((self.num_item,))
        cold_item_list = np.zeros((num,), dtype=np.int64)
        for i in range(self.num_item):
            links_item[i] = list(self.dataMatrix[:,i]).count(1)

# #查看用户活跃程度比例
#         import csv
#         # mT，mY均为N行1列的ndarray数据
#         with open("epinSet_i.csv", "w",encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile, dialect='excel')
#             writer.writerow(list(links_item))

        indices = np.argsort(links_item)
        n = 0
        for id in indices:
            itemname = self.id2item[id]
            cold_item_list[n] = itemname
            n += 1
            if n >= num:
                break
        print("The "+str(num) +" coldest "  + "items:")
        print(cold_item_list)
        # cold_item = str(cold_item_list[0])
        # print("The coldest item：" + str(cold_item))
        return cold_item_list

    # 寻找随机物品
    def findRandomItem(self):
        itemList = []
        num = int(self.insert_item_num)
        random_item_list = np.zeros((num,), dtype=np.int64)
        temp_data = self.data[:]
        for i, entry in enumerate(temp_data):
            userName, itemName, rating = entry
            itemList.append(itemName)
        for j in range(num):
            random_item_list[j] = choice(itemList)
        print("Random item list：" + str(random_item_list))
        return random_item_list

    # 按要求寻找用户列表,返回按照用户活跃度递减的列表
    def findActiveUserList(self):
        print("Looking for Active users")
        cnt = 0  # 次数
        # 寻找最活跃的用户集合
        N = int(self.num_user)
        links_user = np.zeros((self.num_user,))
        active_user_list = np.zeros((N,), dtype=np.int64)

        for i in range(self.num_user):
            links_user[i] = list(self.dataMatrix[i]).count(1) # 每一行出现1的次数

        # import csv
        # # mT，mY均为N行1列的ndarray数据
        # with open("u5Set_u.csv", "w",encoding='utf-8') as csvfile:
        #     writer = csv.writer(csvfile, dialect='excel')
        #     writer.writerow(list(links_user))

        indices = np.argsort(-links_user)
        n = 0
        # 找出所有热门用户
        # insert_itemID = self.item2id[self.insertItem]
        for id in indices:
            # if self.dataMatrix[id][insert_itemID] != 1:  # 并未包含
                username = self.id2user[id]
                active_user_list[n] = username
                n = n+1
        return active_user_list

    # 寻找冷门用户
    def findQuietUserList(self):
        print("Looking for Cold users")
        # 寻找最冷门的用户集合
        N = int(self.num_user)
        links_user = np.zeros((self.num_user,))
        quiet_user_list = np.zeros((N,), dtype=np.int64)
        for i in range(self.num_user):
            links_user[i] = list(self.dataMatrix[i]).count(1)
        indices = np.argsort(links_user)
        n = 0
        # 找出未喜欢冷门物品的用户
        # insert_itemID = self.item2id[self.insertItem]
        for id in indices:
            # if self.dataMatrix[id][insert_itemID] != 1:  # 并未包含
                username = self.id2user[id]
                quiet_user_list[n] = username
                n += 1
                if n >= N:
                    break
        # N是用户总数的1%
        # print("The " + str(self.k) + " least popular " + "users:")
        # print(quiet_user_list)
        return quiet_user_list

        # 寻找一般用户
    def findGeneralUserList(self):
        print("Looking for General users")
        # 寻找最一般的用户集合
        N = int(self.num_user)
        links_user = np.zeros((self.num_user,))
        general_user_list = np.zeros((N,), dtype=np.int64)
        for i in range(self.num_user):
            links_user[i] = list(self.dataMatrix[i]).count(1)
        start = int(self.num_user * 0.25)
        end = int(self.num_user * 0.75)
        indices = np.argsort(links_user)
        tem_array = indices[start:end]
        shuffle(tem_array)
        n = 0

        # insert_itemID = self.item2id[self.insertItem]
        for id in tem_array:
            # if self.dataMatrix[id][insert_itemID] != 1:  # 并未包含
                username = self.id2user[id]
                general_user_list[n] = username
                n += 1
                if n >= N:
                    break
        # N是用户总数的1%
        # print("The " + str(self.k) + " general users :" )
        # print(general_user_list)
        return general_user_list



    # 寻找随机用户
    def findRandomUserList(self):
        print("Looking for Random users")
        N = int(self.num_user)
        links_user = np.zeros((self.num_user,))
        random_user_list = np.zeros((N,), dtype=np.int64)
        for i in range(self.num_user):
            links_user[i] = list(self.dataMatrix[i]).count(1)
        indices = np.argsort(links_user)
        shuffle(indices) # 打乱排序
        n = 0

        # insert_itemID = self.item2id[self.insertItem]
        for id in indices:
            # if self.dataMatrix[id][insert_itemID] != 1:  # 并未包含
                username = self.id2user[id]
                random_user_list[n] = username
                n += 1
                if n >= N:
                    break

        # print(str(N) +" random"  + "users:")
        # print(random_user_list)
        return random_user_list

    # def insertData(self):
    #     afterInsert = self.data[:]
    #     for id in self.userList:
    #         afterInsert.append([id, self.insertItem , float(1)])
    #     return afterInsert

    def getItemCnt(self):
        cnt = 0
        i = 0
        top = self.ranking['-topN'].split(',')
        top = [int(num) for num in top]
        TopN = int(top[-1])
        # resultPath = '../results/result319.txt'
        currentTime = strftime("%Y-%m-%d", localtime(time()))
        file = self.config['recommender'] +'train@'  + currentTime + '-top-' + str(
            TopN) + 'items' + '.txt'
        dir = self.output['-dir']
        resultPath = dir + file
        # print("开始处理结果")
        with open(resultPath) as f:
            result = f.readlines()
            recommenderList = {}
            for line in result:
                username = line.strip().split(":")[0]
                itemname = line.strip().split(":")[1]
                recommenderList[i] = itemname
                i = i + 1

            for item in self.insertItem:
                insert_item = str(item)
                for k, v in recommenderList.items():
                    for i in v.split(','):
                        if insert_item == i:
                            cnt += 1
            average_cnt = int(cnt/len(self.insertItem))

            print("The item "+str(self.insertItem)+" appears "+ str(average_cnt)+ " times in the user TopN list." )

        return average_cnt

    #  画图
    def plot_fig(self,result):
        result = np.array(result)
        # 创建窗口
        plt.figure()
        plt.grid()  # 显示网格
        title = str(self.insert_item_num) +str(self.itemType) +"&"+str(self.k) + str(self.userType)
        # title = str(self.insert_item_num) + str(self.itemType) + "&500"  + str(self.userType)
        plt.title(title)
        plt.xlabel('iters')  # 设置X轴标签
        plt.ylabel('increased_cnt')  # 设置Y轴标签
        # y=[442]
        # plt.yticks(y)
        plt.scatter(result[:, 0], result[:, 1], marker='.', c='k')
        #plt.scatter(result[-1, 0], result[-1, 1], marker='x', c='r')
        currentTime = strftime("%Y-%m-%d-%H", localtime(time()))
        filename = self.itemType +"&"+ self.userType+"@"+currentTime
        filepath = self.output['-dir']+'IncreasedCnt/'
        plt.savefig(filepath+filename)
        plt.show()

        #  画曲线图


    def plot_curve(self, result):
        result = np.array(result)
        # 创建窗口
        plt.figure()
        title = str(self.insert_item_num) + str(self.itemType) + "&" + str(self.userType)
        plt.title(title)
        plt.grid()  # 显示网格
        plt.xlabel('insert_num_user')  # 设置X轴标签
        plt.ylabel('increased_cnt')  # 设置Y轴标签
        x = result[:, 0]
        y = result[:, 1]
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        plt.plot(x_smooth, y_smooth, ls="-", lw=2)
        currentTime = strftime("%Y-%m-%d-%H", localtime(time()))
        filename = "curve&"+self.itemType + "&" + self.userType + "@" + currentTime
        filepath = self.output['-dir'] + 'IncreasedCnt/'
        plt.savefig(filepath + filename)
        plt.show()

# 注入
    def insertData(self):
        N = int(self.k)
        insert_user_list = np.zeros((N,), dtype=np.int64)
        for item in self.insertItem :
            n = 0
            for user in self.userList :
                username = str(user)
                insert_itemID = self.item2id[str(item)]
                user_id = self.user2id[str(user)]
                if self.dataMatrix[user_id][insert_itemID] != 1:
                    insert_user_list[n] = username
                    ratings = randint(1, 5)
                    if self.binarized == '1':
                        ratings = 1
                    self.data.append([username, str(item), float(ratings)])
                    n += 1
                    if n >= N:
                        break
            print("The " + str(N) + " inserted " + "users:")
            print(insert_user_list)
            print("Item " + str(item) + " insertion completed.\n")



    def writeCnt(self,cnt):
        top = self.ranking['-topN'].split(',')
        top = [int(num) for num in top]
        TopN = int(top[-1])
        # resultPath = '../results/result319.txt'
        fileName = self.itemType + "-records" + '-top-' + str(TopN) + '.txt'
        outDir = self.output['-dir']+'IncreasedCnt/'
        res = self.itemType +":"+str(self.insertItem)+" "+str(self.k)  +self.userType +" "+"IncreasedCnt:"+str(cnt)
        FileIO.continueWrite(outDir, fileName, res)
