import time

# 按照活跃度注入用户（非随机）
from algorithm.RecQ import RecQ
from ProcessingData.dataProcess import DataProcess
from tool.config import Config

if __name__ == '__main__':
    print("按照活跃度进行注入")
    s = time.time()
    # conf = Config('../config/movielens/BPR.conf')
    conf = Config('../config/filmtrust/BPR.conf')
    user_class = 5  # 用户分类
    dataProcess = DataProcess(conf, user_class)  # 处理数据，得到n类用户，target_item
    recSys = RecQ(conf, dataProcess.data)  # 将数据集分成训练集和测试集
    res = recSys.execute()  # 执行算法，生成结果
    before_insert_cnt = dataProcess.getItemCnt(res)
    # 接下来运行一次算法
    flilePath = conf['ratings']
    reward = []
    max_reward = 0
    for i in range(30):
        user_active_rank = dataProcess.user_activity_classify()  # 活跃度排名
        selected_user_list = user_active_rank[:50]
        dataProcess.insertData(selected_user_list)
        recSys = RecQ(conf, dataProcess.data)  # 将数据集分成训练集和测试集
        res = recSys.execute()  # 执行算法，生成结果
        # 查看注入后物品出现次数
        after_insert_cnt = dataProcess.getItemCnt(res)
        print("After inserting, they appear " + str(after_insert_cnt) + " times before insertion.")
        recNum = after_insert_cnt - before_insert_cnt
        print("Added " + str(recNum) + " times")
        reward.append(int(recNum))
        max_reward = max(max_reward, recNum)

    dataProcess.plot_curve2(reward, title="hotavg_reward")
    print("avg reward:" + str(sum(reward)/len(reward)))
    print("max reward:" + str(max_reward))
    print("Run time: %f s" % (time.time() - s))





