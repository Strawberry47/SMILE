import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties  
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.weight':'bold', #or 'blod'
        'font.size':'16'
        }
rcParams.update(params)



# 创建数据
x = np.arange(0,500,10)
exposure = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
1, 2, 11, 14, 25, 38, 44, 75, 97, 106, 146, 188, 267, 284, 360, 393, 432, 448, 418, 472, 501, 503, 499, 482, 480, 471, 461, 451, 442]

# 平滑
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = make_interp_spline(x, exposure)(x_smooth)
# plt.plot(x_smooth, y_smooth, ls="-", lw=2)

# plt.title('Influence of tree depth')
fig = plt.figure(figsize=[6.4,8])  #画布大小默认为6.4*4.8

fig1 = fig.add_subplot(211)  


fig1.set_xlabel("number of trial users",fontsize='16')
fig1.set_ylabel("increased exposure",fontsize='16',fontweight='bold')
lns1 = fig1.plot(x_smooth, y_smooth, color='tab:blue',label="exposure", ls='solid')

plt.grid(b=True, ls=':')


############### 创建数据 ###############
x2 = np.arange(0,50,1)
exposure2 = [419,419,455,432,446,435,436,428,370,420,
355,433,447,344,425,407,406,459,457,424,
437,487,414,467,414,412,443,442,395,421,
433,434,390,430,380,427,402,401,436,430,
458,419,386,445,446,449,465,403,437,416]


x_smooth2 = np.linspace(x2.min(), x2.max(), 300)
y_smooth2 = make_interp_spline(x2, exposure2)(x_smooth2)
# plt.plot(x_smooth, y_smooth, ls="-", lw=2)

# plt.title('Influence of tree depth')


fig2 = fig.add_subplot(212)  


fig2.set_xlabel("select 300 adopters each time",fontsize='16')
fig2.set_ylabel("increased exposure",fontsize='16',fontweight='bold')
lns2 = fig2.plot(x_smooth2, y_smooth2, color='tab:blue',label="exposure", ls='solid')

plt.grid(b=True, ls=':')


plt.subplots_adjust(top=0.955,
bottom=0.104,
left=0.414,
right=0.71,
hspace=0.351,
wspace=0.2)


plt.show()