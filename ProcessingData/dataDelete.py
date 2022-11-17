import configparser
import math
import sys
sys.path.append("..")
from ProcessingData.dataProcess import DataProcess
import pandas as pd
import time
s = time.time()

# nohup python -u rlmain.py > filmtrust200.log 2>&1 &
# ps -ef|grep python

class dataDelete():
    def __init__(self,config):

        self.dataProcess = DataProcess(config)  # 处理数据，得到n类用户，target_item
        # 要替换的热门target item数量(前1%的数据)  ciao数据集太特别了，不能选那么多 45
        self.choose_num = int(self.dataProcess.num_item * 0.002)
        active_rank = self.dataProcess.item_activity_ranks()
        self.item_rank = sorted(active_rank.items(), key=lambda x: x[1], reverse=True)
        self.popular_item_list = []
        self.find_target_item()

    def find_target_item(self):
        # 从第几开始算
        start = 0
        for items in self.item_rank[start:start+self.choose_num]:
            self.popular_item_list.append(items[0])
        print("Target item: " + str(self.popular_item_list))
        return self.popular_item_list


    def delete_rating(self):
        # remainNum = 5 # 保留多少交易量
        # 删除交易量
        dict_i = self.dataProcess.sparseMatrix.matrix_Item
        for item in self.popular_item_list:
            # 传入itemname
            pro = 0.1
            remain_num = math.ceil(len(dict_i[item])*pro)
            while len(dict_i[item]) > remain_num:
                last_item = dict_i[item].popitem()
        del_res = []
        for k, v in list(dict_i.items()):
            itemname = k
            for u, r in v.items():
                username = u
                rating = r
                del_res.append([username, itemname, rating])
        print("Delete completed")
        return del_res

    def writeCnt(self, content):
        outDir = config['ENV']['DELETED_TRAIN_RATING_FILE']
        df = pd.DataFrame(content, columns=['username', 'itemname', 'rating'])
        df.to_csv(outDir, header=False, index=False)
        # for i in content:
        #     username = i[0]
        #     itemname = i[1]
        #     rating = str(i[2])
        #     res = username + '\t' + itemname + '\t' + rating
        #     FileIO.continueWrite(outDir, res)
        print('The result has been output to ', outDir, '.')

if __name__ == '__main__':
    config = configparser.ConfigParser()
    _x = open('../config/config_Ciao')
    config.read_file(_x)

    delete = dataDelete(config) # 创建类.
    target_item = delete.popular_item_list # 找到目标物品
    del_res=delete.delete_rating() # 删除交易量
    delete.writeCnt(del_res) # 写入文件


