import numpy as np
import torch
from DebugPrint import *
from Classes import DrxState
from StateChanger import drx_state_change
from StateChanger import ql_state_change, ql_reward_calculation
from ActionChooser import choose_action
from Parameters import *

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is {device}")


# 初始化Q表
def build_q_table(n_states, n_actions):
    table = torch.zeros((n_states, n_actions)).to(device)
    debug_print(f"Q table initialized as \n{table}")
    return table


# 主函数
def rl():
    dsQTable = build_q_table(N_STATES, N_ACTIONS)  # 初始化短睡眠Q表
    epoch = 1  # 初始化主循环计数器
    currDrxState = DrxState.S1  # 初始化当前DRX状态为S1
    dsCounter = 0  # 初始化从S1到S8的过程中，停留在S2的短睡眠计时器
    dsCounterOneEpoch = 0  # 初始化每次进入S2时的短睡眠计时器

    # 最终数据列表
    list_overall_delay = []
    list_power_consumption_coefficient = []

    # 主循环: 根据状态转移概率在八种状态S1~S8之间相互转换
    while True:
        # 状态转换
        debug(f"Current DRX state is {currDrxState}")
        match currDrxState:
            case DrxState.S1:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S2:
                debug_print(f"################ Epoch {epoch} Start ################")

                # 初始化停留于S2期间的所有状态和动作列表
                state_action_in_s2.clear()
                # q_predict_in_s2.clear()
                # q_target_in_s2.clear()
                loss_in_s2.clear()

                epoch_loss = 0.0  # 初始化Epoch损失值
                dsCurrState = np.random.choice(STATE_SPACE)  # 初始化短睡眠Q学习当前状态
                episode = 1  # 初始化子循环计数器

                while dsCounterOneEpoch <= t_N:
                    debug(f"################ Episode {episode} Start ################")

                    debug(f"DS current state is {dsCurrState}")

                    # 选择短睡眠Q学习的动作
                    dsAction = choose_action(
                        dsCurrState, dsQTable, EPSILON, ACTION_SPACE
                    )
                    debug(f"DS action is {dsAction}")

                    # 确定短睡眠Q学习的下个状态与奖励
                    dsNextState = ql_state_change(STATE_SPACE, dsAction)
                    debug(f"DS next state is {dsNextState}")

                    # 将当前状态、选择的动作、下个状态保存入列表
                    state_action_in_s2.append((dsCurrState, dsAction, dsNextState))
                    debug(f"state_action_in_s2 is {state_action_in_s2}")

                    # 将当前状态、选择的动作、下个状态保存入长期列表
                    long_term_state_action_in_s2.append(
                        (dsCurrState, dsAction, dsNextState)
                    )

                    # 更新短睡眠Q学习的状态并开始下一轮迭代
                    dsCurrState = dsNextState

                    dsCounter += (dsAction + 1) * t_DS  # 2为短周期t_DS的值
                    dsCounterOneEpoch += (dsAction + 1) * t_DS  # 2为短周期t_DS的值
                    debug(
                        f"dsCounterOneEpoch is {dsCounterOneEpoch}, dsCounter is {dsCounter}"
                    )
                    # debug(dsQTable)

                    # 子循环计数器自增
                    episode += 1

                # Epoch结束后，将每轮的dsCounter存入长期列表
                long_term_dsCounter.append(dsCounterOneEpoch)
                debug(f"Long term dsCounterOneEpoch is {long_term_dsCounter}")

                # 如果dsCounterOneEpoch超过t_N, 则转换到S3或S4
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                debug(
                    f"S2 >> Average Delay {avgDelayS2}, Power Consumption {pwrConsumCoeff}"
                )

                dsCounterOneEpoch = 0  # 重置dsCounterOneEpoch

                # 计算奖励
                dsReward = ql_reward_calculation(avgDelayS2, pwrConsumCoeff, "S2")
                debug(f"dsReward is {dsReward}")

                # 遍历所有的状态和动作，目的为同时更新多个Q值
                for curr_state, action, next_state in state_action_in_s2:
                    # 计算短睡眠Q学习的估计值
                    dsQPredict = dsQTable[curr_state, action]
                    debug(f"DS predict value is {dsQPredict}")
                    # q_predict_in_s2.append(dsQPredict)

                    # 计算短睡眠Q学习的目标值
                    dsQTarget = dsReward + GAMMA * dsQTable[next_state, :].max()
                    debug(f"DS target value is {dsQTarget}")
                    # q_target_in_s2.append(dsQTarget)

                    # 根据短睡眠Q学习的估计值与目标值, 结合学习率更新Q表
                    dsQTable[curr_state, action] += ALPHA * (dsQTarget - dsQPredict)
                    debug(f"DS updated Q table is {dsQTable}")

                    # 暂时用估计值与目标值的差异作为损失值, 用作收敛判断条件
                    dsLoss = np.abs(dsQTarget.cpu() - dsQPredict.cpu())
                    loss_in_s2.append(dsLoss)

                # 计算本Epoch的损失值
                if len(loss_in_s2) > 0:
                    epoch_loss = sum(loss_in_s2) / len(loss_in_s2)
                    list_epoch_loss.append(epoch_loss)

                currDrxState = nextDrxState

                debug_print(f"################ Epoch {epoch} End ################")
                # 主循环计数器自增
                epoch += 1

            case DrxState.S3:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S4:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S5:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S6:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S7:
                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")
                currDrxState = nextDrxState

            case DrxState.S8:
                short_term_delay.clear()  # 状态为S8时，清空短期时延列表

                (
                    nextDrxState,
                    avgDelayS2,
                    pwrConsumCoeff,
                    overall_delay,
                ) = drx_state_change(currDrxState, dsCounter)
                debug(f"Next DRX state is {nextDrxState}")

                debug_print(
                    f"S8 >> Overall Delay {overall_delay}, Power Consumption Coefficient {pwrConsumCoeff}"
                )

                list_overall_delay.append(overall_delay)
                list_power_consumption_coefficient.append(pwrConsumCoeff)

                # 到达S8，即用完dsCounter后，将其清零
                dsCounter = 0

                # 计算奖励
                dsReward = ql_reward_calculation(overall_delay, pwrConsumCoeff, "S8")
                debug(f"dsReward is {dsReward}")

                # 遍历所有的状态和动作，目的为同时更新多个Q值
                for curr_state, action, next_state in long_term_state_action_in_s2:
                    # 计算短睡眠Q学习的估计值
                    dsQPredict = dsQTable[curr_state, action]
                    debug(f"DS predict value is {dsQPredict}")
                    # q_predict_in_s2.append(dsQPredict)

                    # 计算短睡眠Q学习的目标值
                    dsQTarget = dsReward + GAMMA * dsQTable[next_state, :].max()
                    debug(f"DS target value is {dsQTarget}")
                    # q_target_in_s2.append(dsQTarget)

                    # 根据短睡眠Q学习的估计值与目标值, 结合学习率更新Q表
                    dsQTable[curr_state, action] += ALPHA * (dsQTarget - dsQPredict)
                    debug(f"DS updated Q table is {dsQTable}")

                    # 暂时用估计值与目标值的差异作为损失值, 用作收敛判断条件
                    dsLoss = np.abs(dsQTarget.cpu() - dsQPredict.cpu())
                    long_term_loss_in_s2.append(dsLoss)

                # 计算本Epoch的损失值
                if len(long_term_loss_in_s2) > 0:
                    epoch_loss = sum(long_term_loss_in_s2) / len(long_term_loss_in_s2)
                    long_term_list_epoch_loss.append(epoch_loss)

                currDrxState = nextDrxState

            case _:
                debug("Unknown current DRX state!")

        # ToDo @LianXiaoHui: 考虑一下收敛条件
        if len(list_epoch_loss) > 0 and len(long_term_list_epoch_loss) > 0:
            # debug_print(
            #     f"Loss >> list_epoch_loss[-1] is {list_epoch_loss[-1]} and long_term_list_epoch_loss[-1] is {long_term_list_epoch_loss[-1]}"
            # )

            if epoch % 100 == 0:
                debug_print(f"Loss @ epoch {epoch}: {long_term_list_epoch_loss[-1]}")

            if (
                # list_epoch_loss[-1] < 0.01
                # and list_epoch_loss[-1] != 0
                long_term_list_epoch_loss[-1] < 0.1
                and long_term_list_epoch_loss[-1] != 0
            ):
                debug_print(
                    f"Converged !!! @ list_epoch_loss[-1] is {list_epoch_loss[-1]} and long_term_list_epoch_loss[-1] is {long_term_list_epoch_loss[-1]}"
                )

                debug_print(f"dsQTable: \n{dsQTable}")
                debug_print(f"list_overall_delay: \n{list_overall_delay}")
                debug_print(
                    f"list_power_consumption_coefficient: \n{list_power_consumption_coefficient}"
                )
                debug_print(
                    f"long_term_list_epoch_loss: \n{[t.item() for t in long_term_list_epoch_loss]}"
                )
                break


# 主程序
if __name__ == "__main__":
    set_debug_mode(False)
    print_parameters()

    N_t = N_DS
    N_e2 = N_DS

    STATE_SPACE = [x for x in range(N_STATES)]
    debug_print(f"STATE_SPACE is {STATE_SPACE}")

    N_ACTIONS = N_DS
    ACTION_SPACE = [i for i in range(N_ACTIONS)]
    debug_print(f"ACTION_SPACE is {ACTION_SPACE}")

    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 初始化Q学习参数
    ALPHA = 0.01
    EPSILON = 0.5  # 为1时，每次都是随机选择动作，即不用Q学习
    GAMMA = 0.9
    debug_print(f"ALPHA is {ALPHA}, EPSILON is {EPSILON}, GAMMA is {GAMMA}")

    # 主函数
    rl()
