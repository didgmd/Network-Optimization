import matplotlib.pyplot as plt
import numpy as np

# 自定义模块
from DebugPrint import *

# 定义基站列表
bsNodeList = []


class BSNode:
    def __init__(self, index, pos_x, pos_y):
        self.id = index  # ID从0开始
        self.posX = pos_x
        self.posY = pos_y

    def __repr__(self):
        return "BSNode: id = %d, posX = %d, posY = %d" % (
            self.id,
            self.posX,
            self.posY,
        )


def network_topology_construct(x_max, y_max, num_bs, bs_min_distance):
    flag_too_close = False
    bs_index = 0

    # 生成基站
    while len(bsNodeList) < num_bs:
        pos_x = np.random.randint(1, x_max)
        pos_y = np.random.randint(1, y_max)
        # debug(f"{pos_x}, {pos_y}")

        # 检查是否与其他基站距离过近
        for objBsNode in bsNodeList:
            if (
                np.sqrt((objBsNode.posX - pos_x) ** 2 + (objBsNode.posY - pos_y) ** 2)
                < bs_min_distance
            ):
                flag_too_close = True
                break

        if flag_too_close:
            flag_too_close = False
            continue

        # 创建基站并添加到基站列表
        bsNodeList.append(BSNode(bs_index, pos_x, pos_y))
        bs_index += 1


if __name__ == "__main__":
    set_debug_mode(True)

    # 网络大小
    X_MAX = 10000
    Y_MAX = 10000

    # 基站数量及基站间最小间距
    NUM_BS = 16
    BS_MIN_DISTANCE = 2000

    # 构建网络拓扑
    network_topology_construct(X_MAX, Y_MAX, NUM_BS, BS_MIN_DISTANCE)

    # 打印基站列表
    debug(f"Number of BS is {len(bsNodeList)}")
    for bsNode in bsNodeList:
        debug(bsNode)

    # 绘制网络拓扑
    fig, ax = plt.subplots()
    for node in bsNodeList:
        ax.plot(node.posX, node.posY, marker="*", color="purple")

    ax.set_xlim([0, X_MAX])
    ax.set_ylim([0, Y_MAX])
    ax.set_aspect("equal")
    plt.show()
