import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

#创建文件夹存放图片
# 获取当前脚本的绝对路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 拼接得到全局文件夹的绝对路径
global_folder = os.path.join(current_path, 'all_images')

# 如果文件夹不存在，则创建
if not os.path.exists(global_folder):
    os.mkdir(global_folder)


def plot_enb(enb_list, x_max, y_max):
    # 全局设置字体
    mpl.rcParams["font.family"] = "Times New Roman"
    # 将enb基站的PCI在图中显示
    fig, ax = plt.subplots()
    for enb in enb_list:
        fill_color = "lightblue" if enb.nodeType == "enb" else "lemonchiffon"
        center_color = "mediumturquoise" if enb.nodeType == "enb" else "khaki"
        center_size = enb.radius * 0.05
        ax.scatter(enb.posX, enb.posY, s=center_size, color=center_color)
        circle = plt.Circle((enb.posX, enb.posY), enb.radius, color=fill_color, fill=True, alpha=0.4)
        ax.add_artist(circle)
        ax.plot(enb.posX, enb.posY, marker=".", color=center_color)
        ax.annotate(str(enb.pci), (enb.posX, enb.posY), fontname="Times New Roman")

    ax.set_xlim([0 - 5, x_max + 5])
    ax.set_ylim([0 - 5, y_max + 5])
    ax.set_aspect("equal")  # 防止图像变形
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    # 设置刻度位置
    plt.axis([0, 100, 0, 100])

    # 设置刻度方向
    plt.tick_params(direction='in')

    # 在所有的坐标轴上显示刻度
    ax = plt.gca()  # 获取当前的Axes对象
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 在坐标轴上都显示刻度数字
    plt.tick_params(labelbottom=True)
    plt.tick_params(labelleft=True)
    plt.tick_params(labeltop=False)
    plt.tick_params(labelright=False)

    #plt.savefig(os.path.join(global_folder, 'enb_topology'))
    image_path = os.path.join(global_folder, 'enb_topology.svg')
    plt.savefig(image_path, dpi=300, format="svg")
    plt.show()


def plot_gnb(gnb_list, x_max, y_max):
    # 全局设置字体
    mpl.rcParams["font.family"] = "Times New Roman"
    # 将gnb基站的PCI在图中显示
    fig, ax = plt.subplots()
    for gnb in gnb_list:
        fill_color = "lightpink" if gnb.nodeType == "gnb" else "lightblue"
        center_color = "hotpink" if gnb.nodeType == "gnb" else "turquoise"
        center_size = gnb.radius * 0.005  # 将中心点的大小减小
        ax.scatter(gnb.posX, gnb.posY, s=center_size, color=center_color)
        circle = plt.Circle((gnb.posX, gnb.posY), gnb.radius, color=fill_color, fill=True, alpha=0.4)
        ax.add_artist(circle)
        ax.plot(gnb.posX, gnb.posY, marker=".", color=center_color)
        ax.annotate(str(gnb.pci), (gnb.posX, gnb.posY), fontname="Times New Roman")

    ax.set_xlim([0 - 5, x_max + 5])
    ax.set_ylim([0 - 5, y_max + 5])
    ax.set_aspect("equal")  # 防止图像变形
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    # 设置刻度位置
    plt.axis([0, 100, 0, 100])

    # 设置刻度方向
    plt.tick_params(direction='in')

    # 在所有的坐标轴上显示刻度
    ax = plt.gca()  # 获取当前的Axes对象
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # 在坐标轴上都显示刻度数字
    plt.tick_params(labelbottom=True)
    plt.tick_params(labelleft=True)
    plt.tick_params(labeltop=False)
    plt.tick_params(labelright=False)

    image_path = os.path.join(global_folder, 'gnb_topology.svg')
    plt.savefig(image_path, dpi=300, format="svg")
    plt.show()


# 绘制冲突点和混淆点的图像
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def plot_collision_and_confusion(node_list, epoch, episode, x_max, y_max):
    fig, ax = plt.subplots()
    for node in node_list:
        color = "purple" if node.nodeType == "enb" else "green"
        circle = plt.Circle(
            (node.posX, node.posY), node.radius, color=color, fill=False
        )
        ax.add_artist(circle)
        if node.collisionList:
            for collision_node in node.collisionList:
                plt.plot(collision_node[0], collision_node[1], "r*", markersize=10)
        if node.confusionList:
            for confusion_node in node.confusionList:
                plt.plot(confusion_node[0], confusion_node[1], "k*", markersize=10)

    # if episode == 1:
    #     ax.set_title(
    #         f"Collision and Confusion status at Initial state of Epoch {epoch}"
    #     )
    # else:
    #     ax.set_title(f"Collision and Confusion status at Episode {episode} of Epoch {epoch}")

    ax.set_xlim([0 - 5, x_max + 5])
    ax.set_ylim([0 - 5, y_max + 5])
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_aspect("equal")

    # 设置刻度位置
    plt.axis([0, 100, 0, 100])

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

    # 创建图例
    purple_patch = mpatches.Patch(color='purple', label='eNB')
    green_patch = mpatches.Patch(color='green', label='gNB')

    # 创建星型的图例标记
    black_star = Line2D([0], [0], marker='*', color='w', label='Confusion', markerfacecolor='black', markersize=15)
    red_star = Line2D([0], [0], marker='*', color='w', label='Collision', markerfacecolor='red', markersize=15)

    # 添加图例
    ax.legend(handles=[purple_patch, green_patch, black_star, red_star], loc='upper right')

    image_path = os.path.join(global_folder, f'Collision_and_Confusion_at_episode_{episode}_of_epoch_{epoch}.pdf')
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()


# 绘制冲突点和混淆点的图像
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def plot_collision_and_confusion_final(node_list, x_max, y_max, episode,epoch):
    fig, ax = plt.subplots()
    for node in node_list:
        color = "purple" if node.nodeType == "enb" else "green"
        circle = plt.Circle(
            (node.posX, node.posY), node.radius, color=color, fill=False
        )
        ax.add_artist(circle)
        if node.collisionList:
            for collision_node in node.collisionList:
                plt.plot(collision_node[0], collision_node[1], "r*", markersize=10)
        if node.confusionList:
            for confusion_node in node.confusionList:
                plt.plot(confusion_node[0], confusion_node[1], "k*", markersize=10)

    ax.set_xlim([0 - 5, x_max + 5])
    ax.set_ylim([0 - 5, y_max + 5])
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_aspect("equal")

    # 设置刻度位置
    plt.axis([0, 100, 0, 100])

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

    # 创建图例
    purple_patch = mpatches.Patch(color='purple', label='eNB')
    green_patch = mpatches.Patch(color='green', label='gNB')

    # 创建星型的图例标记
    black_star = Line2D([0], [0], marker='*', color='w', label='Confusion', markerfacecolor='black', markersize=15)
    red_star = Line2D([0], [0], marker='*', color='w', label='Collision', markerfacecolor='red', markersize=15)

    # 添加图例
    ax.legend(handles=[purple_patch, green_patch, black_star, red_star], loc='upper right')

    image_path = os.path.join(global_folder, f'Collision_and_Confusion_end_at_episode_{episode}_of_epoch_{epoch}.pdf')
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()


# 绘制冲突和混淆点数目的曲线图
def plot_collision_and_confusion_curve(
    global_collision_num_list, global_confusion_num_list, epoch
):
    # 绘制冲突和混淆点数目的曲线图
    fig, ax = plt.subplots()
    ax.plot(global_collision_num_list, label="Collision")
    ax.plot(global_confusion_num_list, label="Confusion")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of Points")
    ax.set_title("Collision and Confusion Curve of Epoch " + str(epoch))
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.legend()

    image_path = os.path.join(global_folder, f'Collision_and_Confusion_curve_at_epoch_{epoch}.pdf')
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()


# 绘制loss损失函数值的曲线图
def plot_loss_curve(loss_list):
    # 绘制loss损失函数值的曲线图
    fig, ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    image_path = os.path.join(global_folder, 'Loss_curve.pdf')
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()


# 绘制每个epoch中的episode数目的曲线图
def plot_episodes_in_each_epoch(
    episodes_list_guided_target_enabled, episodes_list_guided_target_disabled
):
    fig, ax = plt.subplots()
    ax.plot(episodes_list_guided_target_enabled, label="With Lead Target")
    ax.plot(episodes_list_guided_target_disabled, label="Without Lead Target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of Episodes")
    ax.set_title("Episodes Curve")
    ax.legend()
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    # 设置x轴的刻度为整数
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    image_path = os.path.join(global_folder, 'Episodes_in_each_Epoch.pdf')
    plt.savefig(image_path, dpi=300, format="pdf")
    plt.show()
