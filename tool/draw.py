import matplotlib.pyplot as plt
import numpy as np

# tree-depth那个图
# 创建数据
x = ['1','2','3','4']
avg_reward = [132,138.3,127.0,112]
runtime = [3.85,0.32,0.22,0.26]

# plt.title('Influence of tree depth')
fig = plt.figure(figsize=[23.5,9.0])  #画布大小默认为6.4*4.8，将宽变为2倍 12.8

fig_reward = fig.add_subplot(122)  #通过fig添加子图，参数：行数，列数，第几个。这是第二个子图
fig_reward.set_title('Influence of tree depth',fontsize=35)

fig_reward.set_xlabel("tree depth",fontsize=35)
fig_reward.set_ylabel("average reward",fontsize=35)
lns1 = fig_reward.plot(x,avg_reward, color='green',label="avg_reward", ls='--', marker='o',markersize=10)
fig_reward.tick_params(labelsize=25)#设置坐标值字体大小
# fig_reward.set_yticks(labelsize=25)
plt.grid(b=True, ls=':')

fig_runtime = fig_reward.twinx()
fig_runtime.set_ylabel("seconds per sampling time",fontsize=35)
lns2 = fig_runtime.plot(x,runtime,  color='skyblue', label="sampling time", ls='-.',  marker='*',markersize=10)
lns = lns1+lns2
labels = [l.get_label() for l in lns]

fig_runtime.tick_params(labelsize=25)#设置坐标值字体大小
# fig_runtime.set_yticks(size=25)
fig_reward.legend(lns,labels,loc='upper right',fontsize=20)
# plt.yticks(size=20)

#下面是柱状图
fig_bar = fig.add_subplot(121) #这是第一个子图
fig_bar.set_title('influence of tree structure',fontsize=35) #第一个子图标题

#第一个子图数据
x = np.arange(3)
_x = ['Movielens100K', 'Movielens1M', 'Ciao']
tree_structure = [0.26, 0.58, 0.29]
without_tree_structure = [3.85, 26.86, 26.71]

#双列柱状图设置
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

#传入双列柱状图数据
bar1 = fig_bar.bar(x, tree_structure, width=0.3, label='tree-structure',color='skyblue')
bar2 = fig_bar.bar(x + width, without_tree_structure, width=0.3, label='without tree-structure',color='bisque')

#设置柱状图数据范围最大为原始数据最大值的1.2倍，避免柱状图上的数据出现在图表外
maxvalue = max(tree_structure + without_tree_structure)
fig_bar.set_ylim(0, int(1.2 * maxvalue))

#柱状图x轴y轴标签
fig_bar.set_xlabel('dataset',fontsize=35)
fig_bar.set_ylabel('seconds per sampling time',fontsize=35)

#柱状图顶部数字
for i, j in enumerate(tree_structure):
    fig_bar.text(i - 0.5 * width, j + 0.3, j, ha='center', fontsize=25)
for i, j in enumerate(without_tree_structure):
    fig_bar.text(i + 0.5 * width, j + 0.3, j, ha='center', fontsize=25)

plt.grid(b=True, ls=':')
plt.xticks(np.arange(3), _x,size=25)#替换柱状图x轴，由数字变为数据集名
plt.yticks(size=25)

plt.legend(loc='upper left',fontsize=20)

plt.savefig('time.pdf')#保存图片
plt.show()

print("")