# 自定义模块
from NetworkTopology import *
from UserManagement import *
from QNet import *
from LevyWalk import *
import random

# from StateCalculator import *
# from RewardCalculator import *
# from KPIManagement import *
from Validation import *
from PictureDrawer import *
from Normalization import *
from Parameters import *
from Anchor import *

# 每Epoch是否更新Levy Walk路径
flagUpdateLevyWalk = False


# 定义用户数量
NUM_USER = 1

# 定义Levy Walk步数及相关参数
LEVY_STEPS = 10000
LEVY_ALPHA = 4  # 帕累托分布的参数
LEVY_SCALE = 1000  # 帕累托分布的尺度参数 250 -> 500 -> 1000

# 定义神经网络1
NN1_ACTIONS = [0, 1]  # 0：不执行，1：执行
NN1_N_INPUT_FEATURES = 9  # S1, S2, S3, S4, S5, S6, X, Y, RSRP
NN1_N_HIDDEN_NEURONS = 100
NN1_N_OUTPUT_FEATURES = len(NN1_ACTIONS)
NN1_ALPHA = 0.001
NN1_EPSILON = 0.5
NN1_GAMMA = 0.9

# 定义神经网络2
sourceRsrpThresholdA2Max = SOURCE_RSRP_THRESHOLD_A2
# sourceRsrpThresholdA2Min = int(
#     (RADIO_LINK_NORMAL_THRESHOLD - RADIO_LINK_FAILURE_THRESHOLD) / 2
#     + RADIO_LINK_FAILURE_THRESHOLD
# )
sourceRsrpThresholdA2Min = -80

targetRsrpThresholdA4Max = HANDOVER_SUCCESS_THRESHOLD
# targetRsrpThresholdA4Min = TARGET_RSRP_THRESHOLD_A4 + (
#     TARGET_RSRP_THRESHOLD_A4 - targetRsrpThresholdA4Max
# )
targetRsrpThresholdA4Min = -80
NN2_ACTIONS_A2 = list(range(sourceRsrpThresholdA2Min, sourceRsrpThresholdA2Max + 1))
NN2_ACTIONS_A4 = list(range(targetRsrpThresholdA4Min, targetRsrpThresholdA4Max + 1))
NN2_N_INPUT_FEATURES = 5  # X, Y, SINR, RSRP, Distance
NN2_N_HIDDEN_NEURONS = 100
NN2_N_OUTPUT_FEATURES_A2 = len(NN2_ACTIONS_A2)
NN2_N_OUTPUT_FEATURES_A4 = len(NN2_ACTIONS_A4)
NN2_ALPHA = 0.05
NN2_INIT_EPSILON = 0.1
NN2_GAMMA = 0.9

# 定义神经网络3
NN3_STATES = list(range(0, NUM_BS))
NN3_ACTIONS = list(range(0, NUM_BS))
NN3_N_INPUT_FEATURES = len(NN3_STATES) * 3  # 到所有基站的X, Y, RSRP
NN3_N_HIDDEN_NEURONS = 100
NN3_N_OUTPUT_FEATURES = len(NN3_ACTIONS)
NN3_ALPHA = 0.1
NN3_INIT_EPSILON = 0.1
NN3_GAMMA = 0.9

# 增加EMA损失函数机制
EMA_LOSS = 0.6

# 定义画图用的主循环计数器
PICTURE_EPOCH = 1

# 初始化基站之间的值,初始值为1
initial_node_values = 1
node_values_graph = np.full((NUM_BS, NUM_BS), initial_node_values)

# 计算切换失败率EMA滚动平均 20240316
EMA_HO_FAIL_RATE = 0.6


def rl():
    for user in userList:
        # 为第i个用户创建神经网络1及优化器1
        q_net1 = QNet(
            NN1_N_INPUT_FEATURES,
            NN1_N_HIDDEN_NEURONS,
            NN1_N_OUTPUT_FEATURES,
        ).to(device)
        optimizer1 = torch.optim.Adam(q_net1.parameters(), lr=NN1_ALPHA)
        # 将网络和优化器存入用户对象
        user.qNetList.append(q_net1)
        user.optimizerList.append(optimizer1)

        # 创建神经网络2及优化器2
        q_net2 = MultipleHeadsQNet(
            NN2_N_INPUT_FEATURES,
            NN2_N_HIDDEN_NEURONS,
            NN2_N_OUTPUT_FEATURES_A2,
            NN2_N_OUTPUT_FEATURES_A4,
        ).to(device)
        optimizer2 = torch.optim.Adam(q_net2.parameters(), lr=NN2_ALPHA)
        # 将网络和优化器存入用户对象
        user.qNetList.append(q_net2)
        user.optimizerList.append(optimizer2)

        # 创建神经网络3及优化器3
        q_net3 = QNet(
            NN3_N_INPUT_FEATURES,
            NN3_N_HIDDEN_NEURONS,
            NN3_N_OUTPUT_FEATURES,
        ).to(device)
        optimizer3 = torch.optim.Adam(q_net3.parameters(), lr=NN3_ALPHA)
        # 将网络和优化器存入用户对象
        user.qNetList.append(q_net3)
        user.optimizerList.append(optimizer3)

        if not flagUpdateLevyWalk:
            # 创建Levy Walk位置及速度并存入用户对象
            levy_position_list = [np.array([X_MAX / 2, Y_MAX / 2])]  # 初始位置为原点
            levy_velocity_list = [np.array([0, 0])]  # 初始速度为0，与位置相对应
            levy_step_x_list = [X_MAX / 2]
            levy_step_y_list = [Y_MAX / 2]

            levy_walk(
                LEVY_STEPS,
                LEVY_ALPHA,
                LEVY_SCALE,
                levy_position_list,
                levy_velocity_list,
                levy_step_x_list,
                levy_step_y_list,
                X_MAX,
                Y_MAX,
            )
            debug(f"levy_position_list: {levy_position_list}")
            debug(f"levy_velocity_list: {levy_velocity_list}")
            debug(f"levy_step_x_list: {levy_step_x_list}")
            debug(f"levy_step_y_list: {levy_step_y_list}")
            user.levyPositionList = levy_position_list.copy()
            user.levyVelocityList = levy_velocity_list.copy()
            # levy_position_list.clear()
            # levy_velocity_list.clear()
            # levy_step_x_list.clear()
            # levy_step_y_list.clear()

    # 检查用户对象
    for user in userList:
        debug(
            f"user.qNetList {id(user.qNetList[0])} {id(user.qNetList[1])} {id(user.qNetList[2])}"
        )
        debug(
            f"user.optimizerList {id(user.optimizerList[0])} {id(user.optimizerList[1])} {id(user.optimizerList[2])}"
        )
        debug(
            f"user.levy_position_list {user.levyPositionList}, length is {len(user.levyPositionList)}"
        )
        debug(
            f"user.levy_velocity_list {user.levyVelocityList}, length is {len(user.levyVelocityList)}"
        )

    # 初始化主循环计数器
    epoch = 1

    # 初始化切换失败率列表
    list_handover_fail_rate = []

    # 初始化切换次数列表
    list_handover_count = []

    # 初始化切换失败次数列表
    list_handover_fail_count = []

    # 初始化RLF列表
    list_rlf = []

    # 初始化贪婪度
    nn2_epsilon = NN2_INIT_EPSILON
    nn3_epsilon = NN3_INIT_EPSILON

    # 计算切换失败率EMA滚动平均 20240316
    list_ema_handover_fail_rate = []
    last_ema_handover_fail_rate = 0.0

    print(
        f"#######################################################\n"
        "################ Simulation Parameters ################\n"
        "#######################################################\n"
        f"REWARD_SCALE_FACTOR = {REWARD_SCALE_FACTOR}\n"
        f"SOURCE_RSRP_THRESHOLD_A2 = {SOURCE_RSRP_THRESHOLD_A2}\n"
        f"TARGET_RSRP_THRESHOLD_A4 = {TARGET_RSRP_THRESHOLD_A4}\n"
        f"HANDOVER_SUCCESS_THRESHOLD = {HANDOVER_SUCCESS_THRESHOLD}\n"
        f"HANDOVER_FAILURE_THRESHOLD = {HANDOVER_FAILURE_THRESHOLD}\n"
        f"RADIO_LINK_FAILURE_THRESHOLD = {RADIO_LINK_FAILURE_THRESHOLD}\n"
        f"RADIO_LINK_NORMAL_THRESHOLD = {RADIO_LINK_NORMAL_THRESHOLD}\n"
        f"X_MAX = {X_MAX}\n"
        f"Y_MAX = {Y_MAX}\n"
        f"NUM_BS = {NUM_BS}\n"
        f"BS_MIN_DISTANCE = {BS_MIN_DISTANCE}\n"
        f"HO_SUCCESS_RATE_MATRIX_PLUS_MULTIPLIER = {HO_SUCCESS_RATE_MATRIX_PLUS_MULTIPLIER}\n"
        f"HO_SUCCESS_RATE_MATRIX_MINUS_MULTIPLIER = {HO_SUCCESS_RATE_MATRIX_MINUS_MULTIPLIER}\n"
        f"NUM_USER = {NUM_USER}\n"
        f"LEVY_STEPS = {LEVY_STEPS}\n"
        f"LEVY_ALPHA = {LEVY_ALPHA}\n"
        f"LEVY_SCALE = {LEVY_SCALE}\n"
        f"NN1_ACTIONS = {NN1_ACTIONS}\n"
        f"NN1_N_INPUT_FEATURES = {NN1_N_INPUT_FEATURES}\n"
        f"NN1_N_HIDDEN_NEURONS = {NN1_N_HIDDEN_NEURONS}\n"
        f"NN1_N_OUTPUT_FEATURES = {NN1_N_OUTPUT_FEATURES}\n"
        f"NN1_ALPHA = {NN1_ALPHA}\n"
        f"NN1_EPSILON = {NN1_EPSILON}\n"
        f"NN1_GAMMA = {NN1_GAMMA}\n"
        f"sourceRsrpThresholdA2Max = {sourceRsrpThresholdA2Max}\n"
        f"sourceRsrpThresholdA2Min = {sourceRsrpThresholdA2Min}\n"
        f"targetRsrpThresholdA4Max = {targetRsrpThresholdA4Max}\n"
        f"targetRsrpThresholdA4Min = {targetRsrpThresholdA4Min}\n"
        f"NN2_ACTIONS_A2 = {NN2_ACTIONS_A2}\n"
        f"NN2_ACTIONS_A4 = {NN2_ACTIONS_A4}\n"
        f"NN2_N_INPUT_FEATURES = {NN2_N_INPUT_FEATURES}\n"
        f"NN2_N_HIDDEN_NEURONS = {NN2_N_HIDDEN_NEURONS}\n"
        f"NN2_N_OUTPUT_FEATURES_A2 = {NN2_N_OUTPUT_FEATURES_A2}\n"
        f"NN2_N_OUTPUT_FEATURES_A4 = {NN2_N_OUTPUT_FEATURES_A4}\n"
        f"NN2_ALPHA = {NN2_ALPHA}\n"
        f"NN2_INIT_EPSILON = {NN2_INIT_EPSILON}\n"
        f"NN2_EPSILON_INCREASE_STEP = {NN2_EPSILON_INCREASE_STEP}\n"
        f"NN2_GAMMA = {NN2_GAMMA}\n"
        f"NN3_STATES = {NN3_STATES}\n"
        f"NN3_ACTIONS = {NN3_ACTIONS}\n"
        f"NN3_N_INPUT_FEATURES = {NN3_N_INPUT_FEATURES}\n"
        f"NN3_N_HIDDEN_NEURONS = {NN3_N_HIDDEN_NEURONS}\n"
        f"NN3_N_OUTPUT_FEATURES = {NN3_N_OUTPUT_FEATURES}\n"
        f"NN3_ALPHA = {NN3_ALPHA}\n"
        f"NN3_INIT_EPSILON = {NN3_INIT_EPSILON}\n"
        f"NN3_EPSILON_INCREASE_STEP = {NN3_EPSILON_INCREASE_STEP}\n"
        f"NN3_GAMMA = {NN3_GAMMA}\n"
        f"EMA_LOSS = {EMA_LOSS}\n"
        f"EMA_HO_FAIL_RATE = {EMA_HO_FAIL_RATE}\n"
        f"#######################################################\n"
        f"\n"
    )

    # 主循环
    while True:
        debug_print(f"################ Epoch {epoch} Start ################")
        # 初始化奖励和
        nn1_reward_sum = 0.0
        nn2_reward_sum = 0.0
        nn3_reward_sum = 0.0
        # 更新Epoch切换计数器
        for user in userList:
            user.hoInitiatedInEpoch = 0
            user.hoSuccessInEpoch = 0
            user.hoFailInEpoch = 0
            user.rlfInEpoch = 0
            user.nn3SameNodeSelectedNum = 0

        # 更新Levy Walk路径
        if flagUpdateLevyWalk:
            debug_print("################ Update Levy Walk ################")
            for user in userList:
                # 创建Levy Walk位置及速度并存入用户对象
                levy_position_list = [np.array([X_MAX / 2, Y_MAX / 2])]  # 初始位置为原点
                levy_velocity_list = [np.array([0, 0])]  # 初始速度为0，与位置相对应
                levy_step_x_list = [X_MAX / 2]
                levy_step_y_list = [Y_MAX / 2]

                levy_walk(
                    LEVY_STEPS,
                    LEVY_ALPHA,
                    LEVY_SCALE,
                    levy_position_list,
                    levy_velocity_list,
                    levy_step_x_list,
                    levy_step_y_list,
                    X_MAX,
                    Y_MAX,
                )
                debug(f"levy_position_list: {levy_position_list}")
                debug(f"levy_velocity_list: {levy_velocity_list}")
                debug(f"levy_step_x_list: {levy_step_x_list}")
                debug(f"levy_step_y_list: {levy_step_y_list}")
                user.levyPositionList = levy_position_list.copy()
                user.levyVelocityList = levy_velocity_list.copy()
                # levy_position_list.clear()
                # levy_velocity_list.clear()
                # levy_step_x_list.clear()
                # levy_step_y_list.clear()

        levy_step = 0

        # 从0走到LEVY_STEPS-1
        while levy_step < LEVY_STEPS:
            debug(f"################ LevyStep {levy_step} Start ################")

            for user in userList:
                # 初始化NN3触发标签与NN3选择同一基站的计数器
                user.nn3Triggered = False
                user.nn3SameNodeSelectedNum = 0

                debug(
                    f"levy_position_list[{levy_step}] is {user.levyPositionList[levy_step]}"
                )
                debug(
                    f"levy_velocity_list[{levy_step}] is {user.levyVelocityList[levy_step]}"
                )

                # 更新NN1当前状态
                nn1_curr_state = nn1_current_state_calculation(
                    levy_step, user, bsNodeList
                )

                # 保存S6值
                user.s6EpochList.append(nn1_curr_state[5])

                # 根据NN1当前状态计算NN1估计值，qNetList[0]对应NN1
                normalized_nn1_curr_state = nn1_state_normalization(nn1_curr_state)
                nn1_q_predict = user.qNetList[0](
                    torch.tensor(normalized_nn1_curr_state).float().to(device)
                ).max()
                debug(f"nn1_q_predict is {nn1_q_predict}")

                # 计算NN1的动作，NN1对应qNetList[0]
                nn1_action = nn1_choose_action(
                    nn1_curr_state,
                    user.qNetList[0],
                    NN1_EPSILON,
                    NN1_ACTIONS,
                    device,
                )

                debug(f"nn1_action is {nn1_action}")
                user.action["NN1"] = nn1_action  # 将NN1动作保存到用户对象中

                # 更新NN1下个状态
                nn1_next_state = nn1_next_state_calculation(
                    nn1_curr_state,
                    nn1_action,
                    levy_step,
                    user,
                    bsNodeList,
                    nn2_epsilon,
                    NN2_ACTIONS_A2,
                    NN2_ACTIONS_A4,
                    device,
                    nn3_epsilon,
                    NN3_ACTIONS,
                    anchor_list,
                )
                debug(f"nn1_next_state is {nn1_next_state}")

                # 根据NN1下个状态S5与S6的值确定奖励，当S5为0，即未发生切换时，奖励为0
                nn1_reward = nn1_reward_calculation(
                    user, levy_step, nn1_curr_state, bsNodeList, nn1_next_state
                )
                debug(f"nn1_reward is {nn1_reward}")
                nn1_reward_sum += nn1_reward

                # 根据NN1下个状态计算NN1目标值
                normalized_nn1_next_state = nn1_state_normalization(nn1_next_state)
                nn1_q_target = nn1_reward + NN1_GAMMA * (
                    user.qNetList[0](
                        torch.tensor(normalized_nn1_next_state).float().to(device)
                    ).max()
                )
                debug(f"nn1_q_target is {nn1_q_target}")

                # 根据NN1估计值与目标值计算损失并通过反向传播优化网络
                nn1_loss = nn.functional.mse_loss(nn1_q_predict, nn1_q_target)
                debug(f"nn1_loss is {nn1_loss}")

                # 增加EMA损失函数机制
                if epoch == 1 and levy_step == 0:
                    user.nn1AvgLoss = nn1_loss.cpu().detach().numpy().item()
                else:
                    user.nn1AvgLoss = (
                        EMA_LOSS * nn1_loss.cpu().detach().numpy().item()
                        + (1 - EMA_LOSS) * user.nn1PrevLoss
                    )

                user.nn1PrevLoss = user.nn1AvgLoss
                user.nn1LossList.append(user.nn1AvgLoss)

                user.optimizerList[0].zero_grad()
                nn1_loss.backward()
                user.optimizerList[0].step()

                # 计算估计值与目标值时的A2与A4的权重
                weight_a2_vs_a4 = 0.5  # 0 ~ 1

                # 根据NN2当前状态计算NN2估计值，qNetList[1]对应NN2
                normalized_nn2_curr_state = nn2_state_normalization(
                    user.currState["NN2"]
                )

                nn2_q_predict_a2 = user.qNetList[1](
                    torch.tensor(normalized_nn2_curr_state).float().to(device)
                )[0].max()
                debug(f"nn2_q_predict_a2 is {nn2_q_predict_a2}")

                nn2_q_predict_a4 = user.qNetList[1](
                    torch.tensor(normalized_nn2_curr_state).float().to(device)
                )[1].max()
                debug(f"nn2_q_predict_a4 is {nn2_q_predict_a4}")

                nn2_q_predict = (
                    weight_a2_vs_a4 * nn2_q_predict_a2
                    + (1 - weight_a2_vs_a4) * nn2_q_predict_a4
                )
                debug(f"nn2_q_predict is {nn2_q_predict}")

                # 更新NN2的下个状态
                nn2_next_state = nn2_next_state_calculation(levy_step, user, bsNodeList)

                # 根据是否应该切换以及NN2选择的动作，确定NN2的奖励
                nn2_reward_a2, nn2_reward_a4 = nn2_reward_calculation(user)
                debug(
                    f"nn2_reward_a2 is {nn2_reward_a2}, nn2_reward_a4 is {nn2_reward_a4}"
                )

                # 根据NN2下个状态计算NN2目标值
                normalized_nn2_next_state = nn2_state_normalization(nn2_next_state)

                nn2_q_target_a2 = nn2_reward_a2 + NN2_GAMMA * (
                    user.qNetList[1](
                        torch.tensor(normalized_nn2_next_state).float().to(device)
                    )[0].max()
                )
                debug(f"nn2_q_target_a2 is {nn2_q_target_a2}")

                nn2_q_target_a4 = nn2_reward_a4 + NN2_GAMMA * (
                    user.qNetList[1](
                        torch.tensor(normalized_nn2_next_state).float().to(device)
                    )[1].max()
                )
                debug(f"nn2_q_target_a4 is {nn2_q_target_a4}")

                nn2_q_target = (
                    weight_a2_vs_a4 * nn2_q_target_a2
                    + (1 - weight_a2_vs_a4) * nn2_q_target_a4
                )
                debug(f"nn2_q_target is {nn2_q_target}")

                nn2_reward_sum += (
                    weight_a2_vs_a4 * nn2_reward_a2
                    + (1 - weight_a2_vs_a4) * nn2_reward_a4
                )

                # 根据NN2估计值与目标值计算损失并通过反向传播优化网络
                nn2_loss = nn.functional.mse_loss(nn2_q_predict, nn2_q_target)

                # 增加EMA损失函数机制
                if epoch == 1 and levy_step == 0:
                    user.nn2AvgLoss = nn2_loss.cpu().detach().numpy().item()
                else:
                    user.nn2AvgLoss = (
                        EMA_LOSS * nn2_loss.cpu().detach().numpy().item()
                        + (1 - EMA_LOSS) * user.nn2PrevLoss
                    )

                user.nn2PrevLoss = user.nn2AvgLoss
                user.nn2LossList.append(user.nn2AvgLoss)

                debug(f"nn2_loss is {nn2_loss}")
                user.optimizerList[1].zero_grad()
                nn2_loss.backward()
                user.optimizerList[1].step()

                # 根据NN1的动作与当前位置的RSRP，即nn1_curr_state[8]，
                # 当NN1决定切换，或者RSRP小于等于A2阈值时，触发NN3。注意此时触发NN3并不一定会发生切换，还要看A4
                # if (nn1_action == 1) or (nn1_curr_state[8] <= user.action["NN2_A2"]):
                if user.nn3Triggered:
                    # debug_print("NN3 Triggered!")
                    # 根据NN3当前状态计算NN3估计值，qNetList[2]对应NN3
                    nn3_q_predict = user.qNetList[2](
                        torch.tensor(user.currState["NN3"]).float().to(device)
                    ).max()
                    debug(f"nn3_q_predict is {nn3_q_predict}")

                    # 更新NN3的下个状态
                    nn3_next_state = nn3_next_state_calculation(
                        user, levy_step, bsNodeList
                    )

                    # 根据NN3选取的基站是否为最近基站，确定NN3的奖励
                    nn3_reward = nn3_reward_calculation(user, levy_step)
                    debug(f"nn3_reward is {nn3_reward}")
                    nn3_reward_sum += nn3_reward

                    # 根据NN3下个状态计算NN3目标值
                    nn3_q_target = nn3_reward + NN3_GAMMA * (
                        user.qNetList[2](
                            torch.tensor(nn3_next_state).float().to(device)
                        ).max()
                    )
                    debug(f"nn3_q_target is {nn3_q_target}")

                    # 根据NN3估计值与目标值计算损失并通过反向传播优化网络
                    nn3_loss = nn.functional.mse_loss(nn3_q_predict, nn3_q_target)

                    # 增加EMA损失函数机制
                    if epoch == 1 and levy_step == 0:
                        user.nn3AvgLoss = nn3_loss.cpu().detach().numpy().item()
                    else:
                        user.nn3AvgLoss = (
                            EMA_LOSS * nn3_loss.cpu().detach().numpy().item()
                            + (1 - EMA_LOSS) * user.nn3PrevLoss
                        )

                    user.nn3PrevLoss = user.nn3AvgLoss
                    user.nn3LossList.append(user.nn3AvgLoss)

                    debug(f"nn3_loss is {nn3_loss}")
                    user.optimizerList[2].zero_grad()
                    nn3_loss.backward()
                    user.optimizerList[2].step()

                # 更新用户当前状态
                user.currState["NN1"] = user.nextState["NN1"].copy()
                user.currState["NN2"] = user.nextState["NN2"].copy()
                user.currState["NN3"] = user.nextState["NN3"].copy()

            levy_step += 1

        debug_print(
            f"Final avgLoss of epoch {epoch} is "
            f"{np.mean([user.nn1AvgLoss for user in userList])} "
            f"{np.mean([user.nn2AvgLoss for user in userList])} "
            f"{np.mean([user.nn3AvgLoss for user in userList])}"
        )
        debug_print(
            f"nn1_reward_sum: {nn1_reward_sum}, "
            f"nn2_reward_sum: {nn2_reward_sum}, "
            f"nn3_reward_sum: {nn3_reward_sum}"
        )

        # 保存每个Epoch中所有的S6值
        for user in userList:
            user.s6TotalList.append(user.s6EpochList.copy())
            user.s6EpochList.clear()

        # 将每个Epoch中的切换失败率保存到列表中
        if sum([user.hoInitiatedInEpoch for user in userList]) > 0:
            list_handover_fail_rate.append(
                sum([user.hoFailInEpoch for user in userList])
                / sum([user.hoInitiatedInEpoch for user in userList])
            )

        # 计算切换失败率EMA滚动平均 20240316
        if sum([user.hoInitiatedInEpoch for user in userList]) > 0:
            curr_handover_fail_rate = sum(
                [user.hoFailInEpoch for user in userList]
            ) / sum([user.hoInitiatedInEpoch for user in userList])

            curr_ema_handover_fail_rate = (
                curr_handover_fail_rate * EMA_HO_FAIL_RATE
                + last_ema_handover_fail_rate * (1 - EMA_HO_FAIL_RATE)
            )

            list_ema_handover_fail_rate.append(curr_ema_handover_fail_rate)
            last_ema_handover_fail_rate = curr_ema_handover_fail_rate

        # 将每个Epoch中的切换次数保存到列表中
        list_handover_count.append(sum([user.hoInitiatedInEpoch for user in userList]))

        # 将每个Epoch中的切换失败次数保存到列表中
        list_handover_fail_count.append(sum([user.hoFailInEpoch for user in userList]))

        # 将每个Epoch中的RLF次数保存到列表中
        list_rlf.append(sum([user.rlfInEpoch for user in userList]))

        # 输出切换失败率列表
        print("HO Failure Rate: ", end="")
        print(" ".join(f"{round(i * 100, 0)}%" for i in list_handover_fail_rate))

        # 计算切换失败率EMA滚动平均 20240316
        print("EMA HO Failure Rate: ", end="")
        print(" ".join(f"{round(i * 100, 0)}%" for i in list_ema_handover_fail_rate))

        """
        # 动态调整贪婪度。暂时取消RLF为零时对NN2贪婪度的调整。
        # if list_rlf[-1] == 0:
        #     nn2_epsilon *= 1.2
        # elif len(list_rlf) > 1:
        if len(list_rlf) > 1:
            if list_rlf[-1] < list_rlf[-2]:
                nn2_epsilon *= 1.1
            elif list_rlf[-1] > list_rlf[-2]:
                nn2_epsilon *= 0.9

        # 考虑将NN2的贪婪度调整与切换失败率相关联
        if list_handover_fail_rate[-1] < min(list_handover_fail_rate):
            nn2_epsilon *= 1.5
            nn3_epsilon *= 1.5
        elif len(list_handover_fail_rate) > 1:
            if list_handover_fail_rate[-1] < list_handover_fail_rate[-2]:
                nn2_epsilon *= 1.1
                nn3_epsilon *= 1.1
            elif list_handover_fail_rate[-1] > list_handover_fail_rate[-2]:
                nn2_epsilon *= 0.9
                nn3_epsilon *= 0.9
        
        if nn2_epsilon > 1.0:
            nn2_epsilon = 1.0

        if nn3_epsilon > 1.0:
            nn3_epsilon = 1.0
        """
        # 20240318 更改动态调整贪婪度策略：将上述根据RLF与HOF动态调整贪婪度更改为下方以固定步长按EPOCH数增加贪婪度，且最大值为0.9
        nn2_epsilon = NN2_EPSILON_INCREASE_STEP * epoch
        nn2_epsilon = 0.9 if nn2_epsilon > 0.9 else nn2_epsilon
        nn3_epsilon = NN3_EPSILON_INCREASE_STEP * epoch
        nn3_epsilon = 0.9 if nn3_epsilon > 0.9 else nn3_epsilon

        debug_print(f"nn2_epsilon: {nn2_epsilon}, nn3_epsilon: {nn3_epsilon}")

        # 输出切换次数列表
        print("HO Count: ", end="")
        print(" ".join(f"{i}" for i in list_handover_count))

        # 输出切换失败次数列表
        print("HO Fail Count: ", end="")
        print(" ".join(f"{i}" for i in list_handover_fail_count))

        # 输出RLF列表
        debug_print(f"list_rlf is {list_rlf}")

        # 判断收敛条件
        flag_converged = True
        # converge_threshold = 0.00000001
        # for user in userList:
        #     if (
        #         user.nn1AvgLoss > converge_threshold
        #         or user.nn2AvgLoss > converge_threshold
        #         or user.nn3AvgLoss > converge_threshold
        #     ):
        #         flag_converged = False
        #         break

        # 收敛条件改为切换失败率小于1.0%
        if list_handover_fail_rate[-1] < 0.0:
            flag_converged = True
            debug_print(
                f"Converged at epoch {epoch} with handover failure rate {list_handover_fail_rate[-1]} < 1.0%"
            )
        else:
            flag_converged = False
            debug_print(
                f"Not converged yet, since handover failure rate {list_handover_fail_rate[-1]} >= 1.0%. Continue..."
            )

        # 收敛条件改为轮次数到达1500
        if epoch >= 1500:
            flag_converged = True
            debug_print(
                f"Converged at epoch {epoch} with epoch {epoch} >= 2000"
            )

        if flag_converged:
            debug_print(
                f"Converged at epoch {epoch} with "
                f"nn1AvgLoss {[user.nn1AvgLoss for user in userList]}, "
                f"nn2AvgLoss {[user.nn2AvgLoss for user in userList]}, "
                f"nn3AvgLoss {[user.nn3AvgLoss for user in userList]}"
            )

            for user in userList:
                debug_print(f"nn1LossList is {user.nn1LossList}")
                debug_print(f"nn2LossList is {user.nn2LossList}")
                debug_print(f"nn3LossList is {user.nn3LossList}")

            # 计算乒乓切换PPHO发生次数
            ppho_calculation(userList, epoch)

            # 输出切换失败率列表
            print("HO Failure Rate: ", end="")
            print(" ".join(f"{round(i * 100, 0)}%" for i in list_handover_fail_rate))
            # 输出切换次数列表
            print("HO Count: ", end="")
            print(" ".join(f"{i}" for i in list_handover_count))
            # 输出切换失败次数列表
            print("HO Fail Count: ", end="")
            print(" ".join(f"{i}" for i in list_handover_fail_count))
            # 输出RLF列表
            print(f"list_rlf is {list_rlf}")

            break

        # 调整主循环计数器
        epoch += 1

        # 调整画图用的主循环计数器
        global PICTURE_EPOCH
        PICTURE_EPOCH += 1

    # 计算切换失败次数
    NEWLIST_FHO = [
        sum(lFHOALLEPOCH[i : i + (NUM_USER * LEVY_STEPS)])
        for i in range(0, len(lFHOALLEPOCH), NUM_USER * LEVY_STEPS)
    ]
    debug_print(f"lFHOAllEpoch {NEWLIST_FHO}")

    # 画图
    # draw_lPPHOAllEpoch(PICTURE_EPOCH)

    # draw_lFHOAllEpoch(PICTURE_EPOCH)

    # 测试部分
    # 由于HO优化过程中可以允许得到反馈，因此考虑两种测试方式：有反馈和无反馈
    while True:
        set_debug_mode(False)
        flag_test_with_feedback = input(
            "The training process is over. "
            "Before we start the testing process, "
            "please choose whether to test with feedback (training) or not. y/n: "
        )

        if flag_test_with_feedback == "y":
            debug_print("################ Test with Feedback Start ################")
            test_with_feedback(userList, bsNodeList, device)
            break
        elif flag_test_with_feedback == "n":
            debug_print("################ Test without Feedback Start ################")
            test_without_feedback(userList, bsNodeList, device)
            break
        else:
            debug_print("Invalid input, please try again.")
            continue


if __name__ == "__main__":
    set_debug_mode(False)

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"device is {device}")

    # 构建网络拓扑
    network_topology_construct(X_MAX, Y_MAX, NUM_BS, BS_MIN_DISTANCE)

    # 打印基站列表
    debug(f"Number of BS is {len(bsNodeList)}")
    for bsNode in bsNodeList:
        debug(bsNode)

    # 初始化锚点列表
    ANCHOR_RADIUS = 10000
    anchor_list = anchor_list_initialization(ANCHOR_RADIUS, X_MAX, Y_MAX, NUM_BS)

    # 初始化用户列表
    user_list_initialization(NUM_USER)

    debug_print(f"NN2_ACTIONS_A2: {NN2_ACTIONS_A2}")
    debug_print(f"NN2_ACTIONS_A4: {NN2_ACTIONS_A4}")

    debug_print(
        f"NN1_N_INPUT_FEATURES:{NN1_N_INPUT_FEATURES}, "
        f"NN1_N_HIDDEN_NEURONS:{NN1_N_HIDDEN_NEURONS}, "
        f"NN1_N_OUTPUT_FEATURES:{NN1_N_OUTPUT_FEATURES}"
    )
    debug_print(
        f"NN2_N_INPUT_FEATURES:{NN2_N_INPUT_FEATURES}, "
        f"NN2_N_HIDDEN_NEURONS:{NN2_N_HIDDEN_NEURONS}, "
        f"NN2_N_OUTPUT_FEATURES_A2:{NN2_N_OUTPUT_FEATURES_A2}, "
        f"NN2_N_OUTPUT_FEATURES_A4:{NN2_N_OUTPUT_FEATURES_A4}"
    )
    debug_print(
        f"NN3_N_INPUT_FEATURES:{NN3_N_INPUT_FEATURES}, "
        f"NN3_N_HIDDEN_NEURONS:{NN3_N_HIDDEN_NEURONS}, "
        f"NN3_N_OUTPUT_FEATURES:{NN3_N_OUTPUT_FEATURES}"
    )

    # 强化学习
    rl()
