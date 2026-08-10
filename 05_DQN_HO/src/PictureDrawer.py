from KPIManagement import *
import matplotlib.pyplot as plt
from HO_DQN_main import *
from StateCalculator import *


# 以EPOCH为横坐标，乒乓切换次数为纵坐标画图
def draw_lPPHOAllEpoch(picture_epoch):
    # 假设 a 和 b 是你的数据
    x_EPOCH = list(range(1, picture_epoch+1))
    Y_lPPHOALLEpoch = lPPHOAllEpoch

    # 绘制以 a 为横坐标、b 为纵坐标的线图
    plt.plot(x_EPOCH, Y_lPPHOALLEpoch, marker='o', linestyle='-')

    # 设置图表标题和坐标轴标签
    plt.title('numbers of PPHO')
    plt.xlabel('epoch')
    plt.ylabel('PPHO')

    # 显示网格线
    plt.grid(True)

    # 设置纵坐标从0开始
    plt.ylim(bottom=0)

    # 设置横纵坐标为整数
    plt.xticks(range(1, picture_epoch + 1))
    plt.yticks(range(0, max(Y_lPPHOALLEpoch) + 1))

    # 显示图表
    plt.show()

# 以EPOCH为横坐标，乒乓切换次数为纵坐标画图
def draw_lFHOAllEpoch(picture_epoch):
    # 假设 a 和 b 是你的数据
    x_EPOCH = list(range(1, picture_epoch+1))
    Y_lFHOALLEpoch = lFHOALLEPOCH

    # 绘制以 a 为横坐标、b 为纵坐标的线图
    plt.plot(x_EPOCH, Y_lFHOALLEpoch, marker='o', linestyle='-')

    # 设置图表标题和坐标轴标签
    plt.title('Plot of b vs a')
    plt.xlabel('a')
    plt.ylabel('b')

    # 显示网格线
    plt.grid(True)

    # 显示图表
    plt.show()









