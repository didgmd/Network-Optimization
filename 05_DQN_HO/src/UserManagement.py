# 自定义模块
from DebugPrint import *

# 创建用户列表
userList = []


class User:
    def __init__(self, index):
        self.id = index
        self.qNetList = []
        self.optimizerList = []
        self.qPredictList = []
        self.qTargetList = []
        self.lossList = []
        self.levyPositionList = []
        self.levyVelocityList = []
        self.currState = {
            "NN1": [],
            "NN2": [],
            "NN3": [],
        }
        self.action = {
            "NN1": 0,
            "NN2_A2": 0,
            "NN2_A4": 0,
            "NN3": 0,
        }
        self.nextState = {
            "NN1": [],
            "NN2": [],
            "NN3": [],
        }
        self.nn1PrevLoss = 0.0
        self.nn1AvgLoss = 0.0
        self.nn1LossList = []
        self.nn2PrevLoss = 0.0
        self.nn2AvgLoss = 0.0
        self.nn2LossList = []
        self.nn3PrevLoss = 0.0
        self.nn3AvgLoss = 0.0
        self.nn3LossList = []
        self.s6EpochList = []  # 每个Epoch的S6值，一维列表
        self.s6TotalList = []  # 所有Epoch的S6值，二维列表
        self.handoverResult = True  # True表示切换成功，False表示切换失败
        self.handoverStatus = 0.0  # 切换状态标志，初始为0.0，成功为1.0，失败为-1.0
        self.radioLinkStatus = 0.0  # 无线链路状态标志，初始为0.0，正常为0.1，中断为-1.0
        self.hoSuccessProbability = 1.0
        self.hoInitiatedInEpoch = 0
        self.hoSuccessInEpoch = 0
        self.hoFailInEpoch = 0
        self.rlfInEpoch = 0
        self.nn3SameNodeSelectedNum = 0
        self.nn3Triggered = False

    def __repr__(self):
        return f"{self.id}"


def user_list_initialization(num_user):
    index = 0
    while len(userList) < num_user:
        userList.append(User(index))
        index += 1

    for user in userList:
        debug(user)
