import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties  
from matplotlib import rcParams
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



# rcParams['font.family']='serif'
# rcParams['font.serif']=['Times New Roman']

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.weight':'bold', #or 'blod'
        'font.size':'13'
        }
rcParams.update(params)

 

# 创建数据
x = ['1','2','3','4']
avg_reward = [112,118.3,107.0,92]
runtime = [3.85,0.32,0.22,0.26]

# plt.title('Influence of tree depth')
fig = plt.figure(figsize=[19.5,9.0])  #画布大小默认为6.4*4.8，将宽变为2倍

fig_reward = fig.add_subplot(212)  #通过fig添加子图，几乘几的网格、第几个子图。这是第二个子图
fig_reward.set_title('Influence of tree depth',fontsize='16')

fig_reward.set_xlabel("tree depth",fontsize='16')
fig_reward.set_ylabel("final exposure",fontsize='16',fontweight='bold')
lns1 = fig_reward.plot(x,avg_reward, color='green',label="final_exposure", ls='--', marker='o',markersize=10)

plt.grid(b=True, ls=':')

# 共同x轴，不同y轴
fig_runtime = fig_reward.twinx()
fig_runtime.set_ylabel("seconds per sampling time",fontsize='16',fontweight='bold')
lns2 = fig_runtime.plot(x,runtime,  color='steelblue', label="sampling time", ls='-.',  marker='*',markersize=10)
lns = lns1+lns2
labels = [l.get_label() for l in lns]

fig_reward.legend(lns,labels,loc='upper right')


#下面是柱状图

fig_bar = fig.add_subplot(211) #这是第一个子图
fig_bar.set_title('influence of tree structure',fontsize='16') #第一个子图标题

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
bar1 = fig_bar.bar(x, tree_structure, width=0.3,color='pink', label='tree-structure')
bar2 = fig_bar.bar(x + width, without_tree_structure, width=0.3,color='orange', label='w/o tree-structure')

#设置柱状图数据范围最大为原始数据最大值的1.2倍，避免柱状图上的数据出现在图表外
maxvalue = max(tree_structure + without_tree_structure)
fig_bar.set_ylim(0, int(1.2 * maxvalue))

#柱状图x轴y轴标签
fig_bar.set_xlabel('dataset',fontsize='16')
fig_bar.set_ylabel('seconds per sampling time',fontsize='16',fontweight='bold')

#柱状图顶部数字
for i, j in enumerate(tree_structure):
    fig_bar.text(i - 0.5 * width, j + 0.3, j, ha='center', fontsize=13)
for i, j in enumerate(without_tree_structure):
    fig_bar.text(i + 0.5 * width, j + 0.3, j, ha='center', fontsize=13)



plt.tight_layout() 
plt.xticks(np.arange(3), _x)#替换柱状图x轴，由数字变为数据集名
plt.grid(b=True, ls=':')
plt.legend(loc='upper left')

plt.subplots_adjust(top=0.955,
bottom=0.104,
left=0.509,
right=0.845,
hspace=0.351,
wspace=0.2)



plt.savefig('time.pdf',bbox_inches='tight')
plt.show()


