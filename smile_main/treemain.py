# coding: utf-8
# nohup python -u treemain.py > tpgr0808.log 2>&1 &
# ps -ef|grep python
from runRec import Recommender
import configparser
from time import  time

def main():
    # 解析配置文件
    config = configparser.ConfigParser()
    # _x = open('config')
    _x = open('../config/config_movielens100k')
    config.read_file(_x)
    show_info(config)
    rec = Recommender(config)
    rec.run()

def show_info(config):
    print('dataset path: %s' % config['ENV']['ORIGN_RATING_FILE'])

if __name__ == '__main__':
    s = time()
    main()
    e = time()
    print("total run time: %f min" %((e-s)/60))
