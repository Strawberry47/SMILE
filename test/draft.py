# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 13:29
# @Author  : Shiqi Wang
# @FileName: draft.py
import time
from tqdm import tqdm

# 发呆0.5s
def action():
    time.sleep(0.5)
with tqdm(total=100000, desc='Example', leave=False, ncols=100, unit='B', unit_scale=True) as pbar:
    for i in range(10):
        # 发呆0.5秒
        action()
        # 更新发呆进度
        pbar.update(10000)
