# 读取数据
from matplotlib import pyplot as plt
import pandas as pd

Sec_Buildings = pd.read_excel(r'./aaaa.xlsx')
# 绘制箱线图
plt.boxplot(x = Sec_Buildings.deepxplore, # 指定绘图数据
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans=True, # 以点的形式显示均值
            boxprops = {'color':'black','facecolor':'steelblue'}, # 设置箱体属性，如边框色和填充色
            # 设置异常点属性，如点的形状、填充色和点的大小
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            # 设置均值点的属性，如点的形状、填充色和点的大小
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
            # 设置中位数线的属性，如线的类型和颜色
            medianprops = {'linestyle':'--','color':'orange'},
            labels = [''] # 删除x轴的刻度标签，否则图形显示刻度标签为1
           )
# 绘制箱线图
plt.boxplot(x = Sec_Buildings.deepMC, # 指定绘图数据
            patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
            showmeans=True, # 以点的形式显示均值
            boxprops = {'color':'black','facecolor':'steelblue'}, # 设置箱体属性，如边框色和填充色
            # 设置异常点属性，如点的形状、填充色和点的大小
            flierprops = {'marker':'o','markerfacecolor':'red', 'markersize':3},
            # 设置均值点的属性，如点的形状、填充色和点的大小
            meanprops = {'marker':'D','markerfacecolor':'indianred', 'markersize':4},
            # 设置中位数线的属性，如线的类型和颜色
            medianprops = {'linestyle':'--','color':'orange'},
            labels = [''] # 删除x轴的刻度标签，否则图形显示刻度标签为1
           )
# 添加图形标题
plt.title('二手房单价分布的箱线图')
# 显示图形
plt.show()