import time
import sys
from random import random

from baseline.baseline import user_activity_classify
from tool.plot_curve import plot_curve2

sys.path.append("..")
from RecQ import RecQ
from tool.config import Config
from ProcessingData.dataProcess import DataProcess


# 随机选取
def select_user_random(num):
    print("Selecting random users...")
    user_list = user_activity_classify()
    random.shuffle(user_list)
    res_user_list = random.sample(user_list, num)
    return res_user_list

if __name__ == '__main__':
    s= time.time()
    # conf = Config('../config/movielens/BPR.conf')
    conf = Config('../config/filmtrust/BPR.conf')
    user_class = 5  # 用户分类
    dataProcess = DataProcess(conf, user_class)
    # 接下来运行一次算法
    flilePath = conf['ratings']
    reward = []
    max_reward = 0
    print('*' * 80)
 # 每次注入数据，就会有一个reward，重复多次，得到一个reward序列，用这个reward对参数进行更新
    for train in range(50): # 重复次数
        print('=' * 80)
        print("Random insert trainNum" + str(train) + "...")
        selected_user_list=select_user_random(200) # 随机选取300位用户id（不是indice）
        dataProcess.insertData(selected_user_list)
        recSys = RecQ(conf, dataProcess.data)  # 将数据集分成训练集和测试集
        res = recSys.execute()  # 执行算法，生成结果
        # 查看注入后物品出现次数
        after_insert_cnt = dataProcess.getItemCnt(res)
        print("After inserting, they appear " + str(after_insert_cnt) + " times before insertion.")
        recNum = after_insert_cnt
        print('\033[1;31m' + 'Added：' + str(recNum) + '\033[0m')
        max_reward = max(max_reward, recNum)
        print('*' * 80)
        # 写入文件
        # dataProcess.writeCnt(recNum)  # 20次的reward
        reward.append(int(recNum))
    print("Max reward:" + str(max_reward))
    print("avg reward:"+str(sum(reward)/len(reward)))
    print('*' * 80)
    plot_curve2(reward, title="randomavg_reward")  # 曲线图
    print("Run time: %f s" %(time.time()-s))

