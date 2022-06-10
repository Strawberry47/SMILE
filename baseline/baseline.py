# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 19:52
# @Author  : Shiqi Wang
# @FileName: baseline.py
def slice_func(self, listTemp, n):
    ind = int(len(listTemp) / n)
    for i in range(0, len(listTemp), ind):
        yield listTemp[i:i + ind]


# 对5类用户按照活跃度进行分类
def user_classify_active(self):
    active_rank = self.user_activity_classify()
    user_class = list(self.slice_func(active_rank, self.user_class))  # 分成n类
    # print("Dividing users into "+str(self.user_class)+" categories completed.")
    # print("length:"+str(len(user_class[0])))
    return user_class


# 将用户按照活跃度进行排序，
def user_activity_classify(self):
    # print("Sorting users by activity...")
    num_rank = {}
    active_user_list = []
    target_item = self.targetItem
    dict_i = self.sparseMatrix.matrix_User
    for k, v in dict_i.items():
        num_rank[k] = len(v)
    user_rank = sorted(num_rank.items(), key=lambda x: x[1], reverse=True)
    for items in user_rank:
        userid = self.user2id[items[0]]
        active_user_list.append(userid)
    ##哦！之前的 想法是统计没买过目标商品的用户
    #     repeat = set(v.keys()).intersection(set(target_item))
    #     if (len(repeat) == 0):
    #         num_rank[k] = len(v)
    # user_count = Counter(num_rank.values())
    # # print("user_count:" + str(user_count))
    # user_rank = sorted(num_rank.items(), key=lambda x: x[1], reverse=True)
    # for items in user_rank:
    #     active_user_list.append(items[0])
    return active_user_list

    # 将用户按照评分高低进行排序


def user_rating_classify(self):
    print("Sorting users by rating...")
    rating_user_list = []
    dict_u = new_sparseMatrix.SparseMatrix(self.rating).matrix_User
    num_rank = {}

    for username, ratings in dict_u.items():
        avg = sum(ratings.values()) / len(ratings)
        num_rank[username] = avg
    user_rank = sorted(num_rank.items(), key=lambda x: x[1], reverse=True)

    for items in user_rank:
        userid = self.user2id[items[0]]
        rating_user_list.append(userid)

    print("Sorting users by rating completed.")
    return rating_user_list

    # 物品按照交易量排序


def item_ranks(self):
    item_rank = {}
    dict_i = self.sparseMatrix.matrix_Item
    for k, v in dict_i.items():
        item_rank[k] = len(v)
    item_count = Counter(item_rank.values())
    # print(item_rank['50'])
    # print("item_count:"+str(item_count))
    # item_rank2 = sorted(item_rank.items(), key=lambda x: x[1], reverse=False)
    return item_rank

    # 随机选取
def select_user_random(self,num):
    print("Selecting random users...")
    user_list = self.user_activity_classify()
    random.shuffle(user_list)
    res_user_list = random.sample(user_list,num)
    return res_user_list