import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
import matplotlib.patches as mpatches

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 自定义模块
from DebugPrint import *

# 定义节点列表
enbList = []
gnbList = []
nodeList = []


class Node:
    def __init__(self, pos_x, pos_y, node_type, radius, pci):
        self.posX = pos_x
        self.posY = pos_y
        self.nodeType = node_type
        self.radius = radius
        self.pci = pci
        self.neighborList = []
        self.collisionList = []
        self.confusionList = []
        self.enbMod30CollisionList = []
        self.enbMod6CollisionList = []
        self.enbMod3CollisionList = []
        self.gnbMod30CollisionList = []
        self.gnbMod4CollisionList = []
        self.gnbMod3CollisionList = []
        self.enbConfusionList = []
        self.gnbConfusionList = []


    def __str__(self):
        return "Node: posX = %d, posY = %d, nodeType = %s, radius = %d, pci = %d" % (
            self.posX,
            self.posY,
            self.nodeType,
            self.radius,
            self.pci,
        )

    def __repr__(self):
        return "Node: posX = %d, posY = %d, nodeType = %s, radius = %d, pci = %d" % (
            self.posX,
            self.posY,
            self.nodeType,
            self.radius,
            self.pci,
        )


def network_topology_construct(
    x_max, y_max, n_nodes, r_enb_min, r_enb_max, r_gnb_min, r_gnb_max
):
    # 生成基站
    while len(nodeList) < n_nodes:
        posX = np.random.randint(1, x_max)
        posY = np.random.randint(1, y_max)

        for enb in enbList:
            if enb.posX == posX and enb.posY == posY:
                continue
        for gnb in gnbList:
            if gnb.posX == posX and gnb.posY == posY:
                continue

        # 生成基站类型
        # nodeType = "enb" if np.random.randint(0, 2) == 0 else "gnb"
        nodeType = "enb" if np.random.choice([0, 1], p=[0.3, 0.7]) == 0 else "gnb"

        # 生成基站覆盖范围并检查是否与其他基站重叠
        flagTooClose = False
        radius = 0

        if nodeType == "enb":
            radius = np.random.randint(r_enb_min, r_enb_max + 1)
            for enb in enbList:
                if (
                    np.sqrt((enb.posX - posX) ** 2 + (enb.posY - posY) ** 2)
                    < radius + enb.radius
                ):
                    flagTooClose = True
        elif nodeType == "gnb":
            radius = np.random.randint(r_gnb_min, r_gnb_max + 1)
            for gnb in gnbList:
                if (
                    np.sqrt((gnb.posX - posX) ** 2 + (gnb.posY - posY) ** 2)
                    < radius + gnb.radius
                ):
                    flagTooClose = True

        if flagTooClose:
            continue

        # 基站初始PCI
        pci = -1

        # 生成基站
        objNode = Node(posX, posY, nodeType, radius, pci)
        # debug(node)

        # 将基站加入列表
        nodeList.append(objNode)
        if nodeType == "enb":
            enbList.append(objNode)
        elif nodeType == "gnb":
            gnbList.append(objNode)


if __name__ == "__main__":
    set_debug_mode(True)

    # 网络大小
    X_MAX = 100
    Y_MAX = 100

    # 节点数量
    N_NODES = 200

    # 节点覆盖范围
    R_ENB_MIN = 1
    R_ENB_MAX = 3
    R_GNB_MIN = 1
    R_GNB_MAX = 1

    network_topology_construct(
        X_MAX, Y_MAX, N_NODES, R_ENB_MIN, R_ENB_MAX, R_GNB_MIN, R_GNB_MAX
    )
    debug(f"len(nodeList) is {len(nodeList)}")
    debug(f"len(enbList) is {len(enbList)}")
    debug(f"len(gnbList) is {len(gnbList)}")

    # 绘制网络拓扑
    fig, ax = plt.subplots()
    for node in nodeList:
        color = "purple" if node.nodeType == "enb" else "green"
        circle = plt.Circle(
            (node.posX, node.posY), node.radius, color=color, fill=False
        )
        ax.add_artist(circle)

    # 创建图例
    purple_patch = mpatches.Patch(color='purple', label='eNB')
    green_patch = mpatches.Patch(color='green', label='gNB')
    ax.legend(handles=[purple_patch, green_patch], loc='upper right')

    # 设置刻度位置
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))

    # 设置刻度方向
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    # 在所有的坐标轴上显示刻度
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 在坐标轴上都显示刻度数字
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.xaxis.set_tick_params(labeltop=False)
    ax.yaxis.set_tick_params(labelright=False)

    # 设置标题和轴标签
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')

    ax.set_xlim([0, X_MAX])
    ax.set_ylim([0, Y_MAX])
    ax.set_aspect("equal")  # 防止图像变形

    # 获取当前脚本的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 拼接得到图片的全路径
    image_path = os.path.join(current_path, 'Topology.pdf')

    # 在显示图像之前保存图像
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()