import matplotlib.pyplot as plt
import numpy as np

reward = [0, 1, 0, 2, 7, 4,  1, 2, 1,  8, 0,  9, 1, 3, 0, 14, 4, 2, 5,  2,  2,  7, 7, 0, 2, 18, 2,  1, 5, 9,9,
          1, 4, 4, 6, 2,36, 4,  19, 15, 19, 1, 3, 23, 0, 4, 0, 2, 1, 0]
# reward = [0, 1, 1, 2, 7, 4, 1, 2, 1, 0, 8,  0, 9, 1, 0, 0, 0, 3, 0, 14, 0, 0, 4, 3, 0,  0, 0, 2,
#           5, 0, 2, 7, 7, 0, 2, 18, 2, 0, 1, 5, 9, 0, 0, 9, 0, 1, 4, 1, 0, 0, 4, 6, 2, 1, 0, 36, 0,4, 0,
#           4, 0, 19, 1, 0, 15, 19, 1, 3, 23, 0, 4, 0, 2, 1, 0]
fig = plt.figure(figsize=[13,9])  #画布大小默认为6.4*4.8，将宽变为2倍 12.8
# 23.5,9.0
fig_reward = fig.add_subplot(111)
# fig_reward.set_title('Results of selecting the same user each time',fontsize=35)
fig_reward.set_xlabel("times",fontsize=40)
fig_reward.set_ylabel("reward",fontsize=40)
lns1 = fig_reward.plot(reward, ls='-', marker='o',markersize=10)
fig_reward.tick_params(labelsize=30)#设置坐标值字体大小

plt.grid(b=True, ls=':')
# with plt.rc_context({'image.composite_image': False}):
#     fig.savefig('fig_reward.pdf',dpi=1000)
plt.savefig('fig_reward.pdf')#保存图片
plt.show()