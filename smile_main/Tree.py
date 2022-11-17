# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 9:54
# @Author  : Shiqi Wang
# @FileName: Tree.py
#coding: utf-8
import sys
sys.path.append("..")
from core import run_time_tools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from SmileEnv import SmileEnv
import numpy as np
import random
from tool import utils
import math
import os

# 构建平衡层次聚类树 主要是不停递归调用
class Tree():
    def __init__(self, config,dataset):
        self.config = config
        self.dataset = dataset
        self.log = utils.Log()
        self.bc_dim = int(self.config['USERSELECTOR']['TREE_DEPTH']) # 深度
        self.env = SmileEnv(self.config, dataset)
        # 通过总用户数，深度，算出非叶节点孩子数
        self.child_num = int(math.ceil(math.pow(self.dataset.max_num_user, 1 / self.bc_dim)))
        # self.child_num = int(self.config['USERSELECTOR']['CHILD_NUM']) # 非叶节点的孩子数，相当于论文里的c
        self.clustering_type = self.config['USERSELECTOR']['CLUSTERING_TYPE'] # pca聚类
        self.output = self.config['ENV']['OUT_PUT']
        # 存聚类需要的embedding '../results/BPR-movielens/smile/rating_vector'
        self.clustering_vector_file_path = self.output + '%s_vector' % (self.config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE'].lower())
        # 树结构 id2code之类的
        self.tree_file_path = self.output + 'tree_model_%s_%s_c%d' % (self.config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE'].lower(),
                                                                               self.config['USERSELECTOR']['CLUSTERING_TYPE'].lower(), self.child_num)


    def construct_tree(self):
        top_to_bottom, leaf_id = self.build_mapping()
        obj = {'top_to_bottom': top_to_bottom, 'leaf_id': leaf_id, 'dataset': self.config['ENV']['ORIGN_RATING_FILE'],
               'child_num': int(self.child_num), 'clustering_type': self.config['USERSELECTOR']['CLUSTERING_TYPE']}
        utils.pickle_save(obj, self.tree_file_path)

    # 建立映射
    def build_mapping(self):
        top_to_bottom = np.zeros(dtype=float, shape=[self.dataset.max_num_user, self.bc_dim])# user_num*2
        # 获取聚类需要的embedding
        if not os.path.exists(self.clustering_vector_file_path):
            run_time_tools.clustering_vector_constructor(self.config,self.dataset)
        id_to_vector = np.loadtxt(self.clustering_vector_file_path, delimiter='\t') # embedding
        # 传入 item_list, code_range, top_to_bottom, id_to_vector ；修改了top_to_bottom
        self.hierarchical_code(list(range(self.dataset.max_num_user)), (0, int(int(math.pow(self.child_num, self.bc_dim)))), top_to_bottom, id_to_vector)
        leaf_id = self.get_leaf_id(top_to_bottom) # 返回叶节点位置对应id
        return (top_to_bottom, leaf_id)

    # 这里是获取辈分的，有几层就有多长
    def get_code(self, id):
        code = np.zeros(dtype=int, shape=[self.bc_dim])
        for i in range(self.bc_dim):
            c = id % self.child_num # 从当前辈分开始算的哦
            code[self.bc_dim-i-1] = c
            id = int(id / self.child_num)
            if id == 0:
                break
        return code

    # user_list代表要聚类的列表，code_range表示要编码的数量  一开始是孩子数^深度
    # 这里传入的userlist都是userid哦，要对这些user建树
    def hierarchical_code(self, user_list, code_range, top_to_bottom, id_to_vector):
        if len(user_list) == 0:
            return
        # 递归终止条件：只剩下一个user了；此时计算辈分表
        if len(user_list) == 1:
            top_to_bottom[user_list[0]] = self.get_code(code_range[0])
            return
        # 最终返回分好组的item_list_assign，分成child num个组哦
        if self.clustering_type=='PCA':
            user_list_assign = self.pca_clustering(user_list, id_to_vector) # 返回n个一组的item list(n=总长度除以孩子数10)
        if self.clustering_type=='KMEANS':
            user_list_assign = self.kmeans_clustering(user_list, id_to_vector)
        if self.clustering_type=='RANDOM':
            user_list_assign = self.random_clustering(user_list, id_to_vector)
        range_len = int((code_range[1]-code_range[0])/self.child_num) # 分完组后,每组长度
        for i in range(self.child_num): # 10个组里，每个组都继续递归调用；直到len(item_list)=1
            self.hierarchical_code(user_list_assign[i], (code_range[0]+i*range_len, code_range[0]+(i+1)*range_len), top_to_bottom, id_to_vector)

    def kmeans_clustering(self, user_list, id_to_vector):
        if len(user_list) <= self.child_num:
            return [[item] for item in user_list] + [[] for i in range(self.child_num - len(user_list))]

        random.shuffle(user_list)
        vectors = [id_to_vector[item] for item in user_list]
        vi_to_id = {}
        id_to_vi = {}
        for i, item in zip(range(len(user_list)), user_list):
            vi_to_id[i] = item
            id_to_vi[item] = i
        kmeans = KMeans(n_clusters=self.child_num)
        kmeans.fit(vectors)
        cs = kmeans.cluster_centers_
        labels = kmeans.labels_
        ds = [[] for i in range(self.child_num)]
        for i, l in zip(range(len(labels)), labels):
            ds[l].append(vi_to_id[i])

        index2len = [(i, len(ds[i])) for i in range(self.child_num)]
        reordered_index = [item[0] for item in sorted(index2len, key = lambda x: x[1], reverse=True)]
        tmp_cs = [cs[index] for index in reordered_index]
        tmp_ds = [ds[index] for index in reordered_index]
        result_cs = list(tmp_cs)
        result_ds = []

        list_len = int(math.ceil(len(user_list) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(user_list))
        target_len = [list_len for i in range(non_decrease_num)] + [list_len - 1 for i in range(self.child_num - non_decrease_num)]

        spare_ps = []
        for i in range(self.child_num):
            tmp_d = tmp_ds[i]
            if len(tmp_d) > target_len[i]:
                result_ds.append(list(tmp_d[0: target_len[i]]))
                for j in range(target_len[i], len(tmp_d)):
                    spare_ps.append(tmp_d[j])
            else:
                result_ds.append(tmp_d)

        for i in range(self.child_num):
            num = target_len[i] - len(result_ds[i])
            if num > 0:
                p_dis_pairs = []
                for p in spare_ps:
                    p_dis_pairs.append((p, self.dis(p, result_cs[i])))
                top_n_p_dis_pairs = sorted(p_dis_pairs, key=lambda x: x[1])[:num]
                for pair in top_n_p_dis_pairs:
                    result_ds[i].append(pair[0])
                    spare_ps.remove(pair[0])

        return result_ds

    def random_clustering(self, user_list, id_to_vector):
        if len(user_list) <= self.child_num:
            return [[user] for user in user_list] + [[] for i in range(self.child_num - len(user_list))]

        random.shuffle(user_list)
        list_len = int(math.ceil(len(user_list) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(user_list))
        target_len = [list_len for i in range(non_decrease_num)] + [list_len - 1 for i in range(self.child_num - non_decrease_num)]

        result_ds = [[] for i in range(self.child_num)]
        count = 0
        for i in range(self.child_num):
            for j in range(target_len[i]):
                result_ds[i].append(user_list[count+j])
            count += target_len[i]

        return result_ds

    def pca_clustering(self, user_list, id_to_vector):
        if len(user_list) <= self.child_num: # 如果传入的需要聚类的list长度比规定的聚类数还小，就直接返回
            return [[item] for item in user_list] + [[] for i in range(self.child_num - len(user_list))]
        # user对应的向量
        data = id_to_vector[user_list]
        pca = PCA(n_components=1) # 把原始数据降到一维
        pca.fit(data) #用data对pca这个对象进行训练
        w = pca.components_[0] # 返回所保留的特征个数
        user_to_projection = [(user, np.dot(id_to_vector[user], w)) for user in user_list] # 相乘，得到item和对应的数
        result = sorted(user_to_projection, key=lambda x: x[1]) # 从小到大排列

        user_list_assign = []
        list_len = int(math.ceil(len(result) * 1.0 / self.child_num))  # 代表多少一组？10
        non_decrease_num = self.child_num - (self.child_num * list_len - len(result)) # 好像代表不能再减少了
        start = 0
        end = list_len
        for i in range(self.child_num): # 进行分组，一共分成child_num个组，每组里有list_len(-1)长度
            user_list_assign.append([result[j][0] for j in range(start, end)])
            start = end
            if i >= non_decrease_num - 1:
                end = end + list_len - 1
            else:
                end = end + list_len
        return user_list_assign #返回分好组的item



    # k-means算法会用到
    def dis(self, a, b):
        return np.power(np.sum(np.square(a-b)), 0.5)

    # 给出top_to_bottom，返回leaf_id
    def get_leaf_id(self, top_to_bottom):
        bottom_index_to_id = -np.ones(shape=[int(int(math.pow(self.child_num, float(self.bc_dim))))], dtype=int) # 长度为叶节点数
        for i in range(len(top_to_bottom)): # 有值的叶节点数量
            t2b = top_to_bottom[i] # 得到当前节点的3元组[gf,father,son]，这也是聚类的顺序，也是建树的顺序
            cur_index = self.get_index(t2b) # 叶节点在树中序号（从上到下从零开始编号）
            bottom_index_to_id[cur_index] = i # 树中位置对应的id（数据集中）从0开始哦
        print('leaf num count: %d'%(len(bottom_index_to_id)))
        return bottom_index_to_id


    # 根据几个祖宗的辈分，求出叶节点index(整棵树从0编号)
    def get_index(self, code):
        # print("code:",code)
        result = 0
        for c in code:
            result = self.child_num * result + int(c)
        # print("result:", result)
        return result
