import configparser
import sys
sys.path.append("..")
from time import time
import numpy as np
from algorithm.BPR import BPR
import matplotlib.pyplot as plt # 图形工具
from ProcessingData.dataProcess import DataProcess

# nohup python -u compare.py > 2021.log 2>&1 &
# nohup python compare.py >> top20.log 2>&1 &
# ps -ef|grep python

def four_methods(conf,training_data,test_data,selected_user_list=[]):
    recNum = 0
    ##BPR 使用完整数据 正常运行
    # bpr2 = BPR(complete_data)
    # bpr2.buildModel()
    # bpr2.evalRanking()
    # res_b = bpr2.resTrain
    # after_insert_cnt4, reward_count4 = dataProcess.getItemCnt(res_b)
    #
    # # BPR-tf 使用完整数据 正常运行
    s1 = time()
    bpr = BPR(conf,training_data,test_data)
    bpr.buildModel_tf()
    bpr.evalRankingCandidate(selected_user_list)
    recNum = bpr.total_recnum

    # 运行test
    # bpr.evalRankingTest()
    # print("在测试集上看效果啦")

    print("total run_time: %f s" % ((time() - s1)))
    # s2 = time()
    # bpr.evalRanking_Candidate()
    # recNum2 = bpr.total_recnum
    # e2 = time()
    # print(" 改过的的总时间: %f s" % ((e2 - s2)))
    # res_t = bpr.resTrain
    # after_insert_cnt, reward_count = dataProcess.getItemCnt(res_t)

    return recNum
    #
    # ISGD，直接用完整数据
    # isgd2 = ISGD(dataProcess=dataProcess, inputData=complete_data)
    # s3 = time.time()
    # isgd2.buildModel_tf()
    # s4 = time.time()
    # print("complete training time:", s4 - s3)
    # isgd2.evalRanking()
    # after_insert_cnt2, reward_count2 = dataProcess.getItemCnt(isgd2.resTrain)

    # 创建ISGD，带有增量
    # isgd1 = ISGD(dataProcess=dataProcess)
    # isgd1.buildModel_tf()
    # isgd1.evalRanking()
    # s1 = time.time()
    # isgd1.incremental_build(insert_data)
    # s2 = time.time()
    # print("incremental training time:",s2-s1)
    # isgd1.evalRanking()
    # after_insert_cnt1, reward_count1 = dataProcess.getItemCnt(isgd1.resTrain)

def plot_curve(result):
    plt.figure()
    plt.title("top10")
    plt.grid()  # 显示网格
    plt.plot(result[0],result[1], marker='*', linestyle="-")
    plt.xlabel('num_user')  # 设置X轴标签
    plt.ylabel('total_reward')  # 设置Y轴标签
    # plt.savefig("reward&iters")
    plt.show()


if __name__ == '__main__':
    s = time()
    config = configparser.ConfigParser()
    _x = open('../config/config_movielens1M')
    config.read_file(_x)
    # conf = Config('../config/movielens/BasicMF_del.conf')
    dataProcess = DataProcess(config)  # 处理数据，得到n类用户，target_item
    # mf = MF()
    reward = [[0],[0]]
    popular_reward = []
    i_recNum = {}
    # active_rank = dataProcess.item_ranks()
    # item_rank = sorted(active_rank.items(), key=lambda x: x[1], reverse=True)
    user_rating_rank = dataProcess.user_rating_classify()
    user_activity_rank = dataProcess.user_activity_classify()


    ## 运行两次 看看注入后的精确度变化
    # recNum = four_methods(config, dataProcess.trainingData, dataProcess.testData)
    # print("刚刚运行了一次 zero recnum:",recNum)


    for i in range(100):
        num_user = i+20
        # print("~"*20,"iters:",num_user,"~"*20)
        # selected_user_list = user_rating_rank[:50]
        selected_user_list = user_activity_rank[:num_user] # 这里返回的是userid
        # selected_user_list = dataProcess.select_user_random(i)
        # 每次固定用户
        # selected_user_list = [892, 368, 121, 104, 46, 895, 429, 5, 727, 737, 364, 422, 596, 449, 694, 601, 531, 763, 230, 526, 69, 912, 412, 62, 226, 660, 679, 194, 919, 471, 597, 387, 183, 311, 865, 389, 180, 80, 823, 925, 158, 935, 356, 454, 210, 353, 195, 914, 286, 140]

        # 进行注入，得到新数据
        insert_data, complete_data = dataProcess.insertData(selected_user_list)
        recNum = four_methods(config, complete_data, dataProcess.testData,selected_user_list)
        print(recNum)
        # print("RecNum",recNum)

        reward[0].append(num_user)
        reward[1].append(recNum)
        # i_recNum[i]=reward[1][-1]-reward[1][-2]
        # print('num_user: ',num_user,'recNum: ',recNum)
    print(reward[1])
    plot_curve(reward)
    print("max: ",np.max(reward[1]))
    print("avg:",np.mean(reward[1]))
    # print("recNum: ",i_recNum)
    e = time()
    print(" Run time: %f s" % ((e - s)))
    print("end")








