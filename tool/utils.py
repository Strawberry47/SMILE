#coding: utf-8

import pickle
import time

class Log():
    def log(self, text, log_time=False):
        print('log: %s' % text)
        if log_time:
            print('time: %s' % time.asctime(time.localtime(time.time())))

def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f) #object保存到f中

def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)  # 如果文件目录为空，就会报错
