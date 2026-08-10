from HO_DQN_main import (
    LEVY_STEPS,
    LEVY_ALPHA,
    LEVY_SCALE,
    NN1_ACTIONS,
    NN2_ACTIONS_A2,
    NN2_ACTIONS_A4,
    NN3_ACTIONS,
    NN1_GAMMA,
    NN2_GAMMA,
    NN3_GAMMA,
    EMA_LOSS,
    levy_walk,
)
from StateCalculator import *
from RewardCalculator import *
from Normalization import *
from Parameters import *


def test_with_feedback_with_path(
    user_list,
    bs_node_list,
    device,
):
    # 初始化计数器
    levy_step = 0

    # 从0走到LEVY_STEPS-1
    while levy_step < LEVY_STEPS:
        debug(f"################ LevyStep {levy_step} Start ################")

        for user in user_list:
            debug(
                f"levyPositionList[{levy_step}] is {user.levyPositionList[levy_step]}"
            )
            debug(
                f"levyVelocityList[{levy_step}] is {user.levyVelocityList[levy_step]}"
            )

            # 更新NN1当前状态
            nn1_curr_state = nn1_current_state_calculation(
                levy_step, user, bs_node_list
            )

            # 保存S6值
            user.s6EpochList.append(nn1_curr_state[5])

            # 根据NN1当前状态计算NN1估计值，qNetList[0]对应NN1
            nn1_q_predict = user.qNetList[0](
                torch.tensor(nn1_curr_state).float().to(device)
            ).max()
            debug(f"nn1_q_predict is {nn1_q_predict}")

            # 计算NN1的动作，NN1对应qNetList[0]
            nn1_action = nn1_choose_action(
                nn1_curr_state,
                user.qNetList[0],
                1.0,
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
                bs_node_list,
                1.0,
                NN2_ACTIONS_A2,
                NN2_ACTIONS_A4,
                device,
                1.0,
                NN3_ACTIONS,
            )
            debug(f"nn1_next_state is {nn1_next_state}")

            # 根据NN1下个状态S5与S6的值确定奖励，当S5为0，即未发生切换时，奖励为0
            nn1_reward = nn1_reward_calculation(
                user, levy_step, nn1_curr_state, bs_node_list, nn1_next_state
            )
            debug(f"nn1_reward is {nn1_reward}")

            # 根据NN1下个状态计算NN1目标值
            nn1_q_target = nn1_reward + NN1_GAMMA * (
                user.qNetList[0](torch.tensor(nn1_next_state).float().to(device)).max()
            )
            debug(f"nn1_q_target is {nn1_q_target}")

            # 根据NN1估计值与目标值计算损失并通过反向传播优化网络
            nn1_loss = torch.nn.functional.mse_loss(nn1_q_predict, nn1_q_target)
            debug(f"nn1_loss is {nn1_loss}")

            # 增加EMA损失函数机制
            if levy_step == 0:
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
            weight_a2_vs_a4 = 0.5

            # 根据NN2当前状态计算NN2估计值，qNetList[1]对应NN2
            normalized_nn2_curr_state = nn2_state_normalization(user.currState["NN2"])

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
            nn2_next_state = nn2_next_state_calculation(levy_step, user, bs_node_list)

            # 根据是否应该切换以及NN2选择的动作，确定NN2的奖励
            nn2_reward_a2, nn2_reward_a4 = nn2_reward_calculation(user)
            debug(f"nn2_reward_a2 is {nn2_reward_a2}, nn2_reward_a4 is {nn2_reward_a4}")

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

            # 根据NN2估计值与目标值计算损失并通过反向传播优化网络
            nn2_loss = torch.nn.functional.mse_loss(nn2_q_predict, nn2_q_target)

            # 增加EMA损失函数机制
            if levy_step == 0:
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
            if (nn1_action == 1) or (nn1_curr_state[8] <= user.action["NN2_A2"]):
                # 根据NN3当前状态计算NN3估计值，qNetList[2]对应NN3
                nn3_q_predict = user.qNetList[2](
                    torch.tensor(user.currState["NN3"]).float().to(device)
                ).max()
                debug(f"nn3_q_predict is {nn3_q_predict}")

                # 更新NN3的下个状态
                nn3_next_state = nn3_next_state_calculation(
                    user, levy_step, bs_node_list
                )

                # 根据NN3选取的基站是否为最近基站，确定NN3的奖励
                nn3_reward = nn3_reward_calculation(user, levy_step)

                # 根据NN3下个状态计算NN3目标值
                nn3_q_target = nn3_reward + NN3_GAMMA * (
                    user.qNetList[2](
                        torch.tensor(nn3_next_state).float().to(device)
                    ).max()
                )
                debug(f"nn3_q_target is {nn3_q_target}")

                # 根据NN3估计值与目标值计算损失并通过反向传播优化网络
                nn3_loss = torch.nn.functional.mse_loss(nn3_q_predict, nn3_q_target)

                # 增加EMA损失函数机制
                if levy_step == 0:
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
        f"Final avgLoss is "
        f"{np.mean([user.nn1AvgLoss for user in user_list])} "
        f"{np.mean([user.nn2AvgLoss for user in user_list])} "
        f"{np.mean([user.nn3AvgLoss for user in user_list])}"
    )


def test_with_feedback(user_list, bs_node_list, device):
    # 用相同路径测试
    debug_print("################ Test with same path ################")
    test_with_feedback_with_path(user_list, bs_node_list, device)

    # 重新生成路径
    for user in user_list:
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

    # 用不同路径测试
    debug_print("################ Test with different path ################")
    test_with_feedback_with_path(user_list, bs_node_list, device)


def test_without_feedback_with_path(user_list, bs_node_list, device):
    # 初始化计数器
    levy_step = 0

    # 从0走到LEVY_STEPS-1
    while levy_step < LEVY_STEPS:
        debug(f"################ LevyStep {levy_step} Start ################")

        for user in user_list:
            debug(
                f"levyPositionList[{levy_step}] is {user.levyPositionList[levy_step]}"
            )
            debug(
                f"levyVelocityList[{levy_step}] is {user.levyVelocityList[levy_step]}"
            )

            # 更新NN1当前状态
            nn1_curr_state = nn1_current_state_calculation(
                levy_step, user, bs_node_list
            )

            # 保存S6值
            user.s6EpochList.append(nn1_curr_state[5])

            # 根据NN1当前状态计算NN1估计值，qNetList[0]对应NN1
            nn1_q_predict = user.qNetList[0](
                torch.tensor(nn1_curr_state).float().to(device)
            ).max()
            debug(f"nn1_q_predict is {nn1_q_predict}")

            # 计算NN1的动作，NN1对应qNetList[0]
            nn1_action = nn1_choose_action(
                nn1_curr_state,
                user.qNetList[0],
                1.0,
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
                bs_node_list,
                1.0,
                NN2_ACTIONS,
                device,
                1.0,
                NN3_ACTIONS,
            )
            debug(f"nn1_next_state is {nn1_next_state}")

            # 根据NN1下个状态S5与S6的值确定奖励，当S5为0，即未发生切换时，奖励为0
            nn1_reward = nn1_reward_calculation(
                user, levy_step, nn1_curr_state, bs_node_list, nn1_next_state
            )
            debug(f"nn1_reward is {nn1_reward}")

            # 根据NN1下个状态计算NN1目标值
            nn1_q_target = nn1_reward + NN1_GAMMA * (
                user.qNetList[0](torch.tensor(nn1_next_state).float().to(device)).max()
            )
            debug(f"nn1_q_target is {nn1_q_target}")

            # 根据NN1估计值与目标值计算损失并通过反向传播优化网络
            nn1_loss = torch.nn.functional.mse_loss(nn1_q_predict, nn1_q_target)
            debug(f"nn1_loss is {nn1_loss}")

            # 增加EMA损失函数机制
            if levy_step == 0:
                user.nn1AvgLoss = nn1_loss.cpu().detach().numpy().item()
            else:
                user.nn1AvgLoss = (
                    EMA_LOSS * nn1_loss.cpu().detach().numpy().item()
                    + (1 - EMA_LOSS) * user.nn1PrevLoss
                )

            user.nn1PrevLoss = user.nn1AvgLoss
            user.nn1LossList.append(user.nn1AvgLoss)

            # 无反馈测试过程中不做优化
            # user.optimizerList[0].zero_grad()
            # nn1_loss.backward()
            # user.optimizerList[0].step()

            # 计算估计值与目标值时的A2与A4的权重
            weight_a2_vs_a4 = 0.5

            # 根据NN2当前状态计算NN2估计值，qNetList[1]对应NN2
            nn2_q_predict_a2 = user.qNetList[1](
                torch.tensor(user.currState["NN2"]).float().to(device)
            )[0].max()
            debug(f"nn2_q_predict_a2 is {nn2_q_predict_a2}")

            nn2_q_predict_a4 = user.qNetList[1](
                torch.tensor(user.currState["NN2"]).float().to(device)
            )[1].max()
            debug(f"nn2_q_predict_a4 is {nn2_q_predict_a4}")

            nn2_q_predict = (
                weight_a2_vs_a4 * nn2_q_predict_a2
                + (1 - weight_a2_vs_a4) * nn2_q_predict_a4
            )
            debug(f"nn2_q_predict is {nn2_q_predict}")

            # 更新NN2的下个状态
            nn2_next_state = nn2_next_state_calculation(levy_step, user, bs_node_list)

            # 根据是否应该切换以及NN2选择的动作，确定NN2的奖励
            nn2_reward_a2, nn2_reward_a4 = nn2_reward_calculation(user)
            debug(f"nn2_reward_a2 is {nn2_reward_a2}, nn2_reward_a4 is {nn2_reward_a4}")

            # 根据NN2下个状态计算NN2目标值
            nn2_q_target_a2 = nn2_reward_a2 + NN2_GAMMA * (
                user.qNetList[1](torch.tensor(nn2_next_state).float().to(device))[
                    0
                ].max()
            )
            debug(f"nn2_q_target_a2 is {nn2_q_target_a2}")

            nn2_q_target_a4 = nn2_reward_a4 + NN2_GAMMA * (
                user.qNetList[1](torch.tensor(nn2_next_state).float().to(device))[
                    1
                ].max()
            )
            debug(f"nn2_q_target_a4 is {nn2_q_target_a4}")

            nn2_q_target = (
                weight_a2_vs_a4 * nn2_q_target_a2
                + (1 - weight_a2_vs_a4) * nn2_q_target_a4
            )
            debug(f"nn2_q_target is {nn2_q_target}")

            # 根据NN2估计值与目标值计算损失并通过反向传播优化网络
            nn2_loss = torch.nn.functional.mse_loss(nn2_q_predict, nn2_q_target)

            # 增加EMA损失函数机制
            if levy_step == 0:
                user.nn2AvgLoss = nn2_loss.cpu().detach().numpy().item()
            else:
                user.nn2AvgLoss = (
                    EMA_LOSS * nn2_loss.cpu().detach().numpy().item()
                    + (1 - EMA_LOSS) * user.nn2PrevLoss
                )

            user.nn2PrevLoss = user.nn2AvgLoss
            user.nn2LossList.append(user.nn2AvgLoss)

            debug(f"nn2_loss is {nn2_loss}")
            # 无反馈测试过程中不做优化
            # user.optimizerList[1].zero_grad()
            # nn2_loss.backward()
            # user.optimizerList[1].step()

            # 当根据NN1和NN2的动作决定尝试切换时，对NN3进行相关计算
            if user.action["NN1"] == 1 and user.action["NN2"] == 1:
                # 根据NN3当前状态计算NN3估计值，qNetList[2]对应NN3
                nn3_q_predict = user.qNetList[2](
                    torch.tensor(user.currState["NN3"]).float().to(device)
                ).max()
                debug(f"nn3_q_predict is {nn3_q_predict}")

                # 更新NN3的下个状态
                nn3_next_state = nn3_next_state_calculation(
                    user, levy_step, bs_node_list
                )

                # 根据NN3选取的基站是否为最近基站，确定NN3的奖励
                nn3_reward = nn3_reward_calculation(user, levy_step)

                # 根据NN3下个状态计算NN3目标值
                nn3_q_target = nn3_reward + NN3_GAMMA * (
                    user.qNetList[2](
                        torch.tensor(nn3_next_state).float().to(device)
                    ).max()
                )
                debug(f"nn3_q_target is {nn3_q_target}")

                # 根据NN3估计值与目标值计算损失并通过反向传播优化网络
                nn3_loss = torch.nn.functional.mse_loss(nn3_q_predict, nn3_q_target)

                # 增加EMA损失函数机制
                if levy_step == 0:
                    user.nn3AvgLoss = nn3_loss.cpu().detach().numpy().item()
                else:
                    user.nn3AvgLoss = (
                        EMA_LOSS * nn3_loss.cpu().detach().numpy().item()
                        + (1 - EMA_LOSS) * user.nn3PrevLoss
                    )

                user.nn3PrevLoss = user.nn3AvgLoss
                user.nn3LossList.append(user.nn3AvgLoss)

                debug(f"nn3_loss is {nn3_loss}")
                # 无反馈测试过程中不做优化
                # user.optimizerList[2].zero_grad()
                # nn3_loss.backward()
                # user.optimizerList[2].step()

            # 更新用户当前状态
            user.currState["NN1"] = user.nextState["NN1"].copy()
            user.currState["NN2"] = user.nextState["NN2"].copy()
            user.currState["NN3"] = user.nextState["NN3"].copy()

        levy_step += 1

    debug_print(
        f"Final avgLoss is "
        f"{np.mean([user.nn1AvgLoss for user in user_list])} "
        f"{np.mean([user.nn2AvgLoss for user in user_list])} "
        f"{np.mean([user.nn3AvgLoss for user in user_list])}"
    )


def test_without_feedback(user_list, bs_node_list, device):
    # 用相同路径测试
    debug_print("################ Test with same path ################")
    test_without_feedback_with_path(user_list, bs_node_list, device)

    # 重新生成路径
    for user in user_list:
        # 创建Levy Walk位置及速度并存入用户对象
        levy_position_list = [np.array([0, 0])]  # 初始位置为原点
        levy_velocity_list = [np.array([0, 0])]  # 初始速度为0，与位置相对应
        levy_step_x_list = [0]
        levy_step_y_list = [0]

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

    # 用不同路径测试
    debug_print("################ Test with different path ################")
    test_without_feedback_with_path(user_list, bs_node_list, device)
