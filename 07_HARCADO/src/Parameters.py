import numpy as np
import csv
import os
from experiment_contract import (
    env_bool,
    env_float,
    env_int,
    trajectory_file,
    trajectory_dir,
)
from metric_contract import RLF_SINR_THRESHOLD_DB
from rl_training_config import (
    DDQN_TARGET_UPDATE_STEPS,
    RL_ALPHA,
    RL_EPSILON,
    RL_GAMMA,
    RL_MAX_EPOCHS,
)

# CSV数据所在文件夹的路径（请根据实际情况修改）
# CSV_FOLDER_PATH = r'd:\Study\sut\课题\20250304_NS3_v7_dataset\Vel6\20250311_v7_TTT1_HOM0.2\11'
# CSV_FOLDER_PATH = r"..\NS3\20250304_NS3_v7_dataset\Vel6\20250311_v7_TTT1_HOM0.2\11"
# CSV_FOLDER_PATH = r"../NS3/20250304_NS3_v7_dataset/Vel6/20250311_v7_TTT1_HOM0.2/11"
# 获取该文件夹下所有CSV文件的完整路径（如果有子目录，也可使用递归方式）
# CSV_FILE_PATHS = glob.glob(CSV_FOLDER_PATH + r"\*.csv")
# CSV_FILE_PATHS = glob.glob(CSV_FOLDER_PATH + r"/*.csv")

# 例如后续的数据加载代码可以遍历 CSV_FILE_PATHS 列表


# 训练参数
# NUM_EPOCHS = 100
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001

# 数据、模型参数
# SEQ_LEN = 15  # 时间步数
# FEATURE_DIM = 17  # 每个时间步的特征数
# D_MODEL = 128  # Transformer隐藏层维度
# NHEAD = 8  # 多头注意力的头数
# NUM_LAYERS = 2  # Transformer Encoder层数
# DIM_FEEDFORWARD = 512  # Transformer中前馈网络维度
# DROPOUT_RATE = 0.5  # dropout率
# NUM_CLASSES = 3  # 分类类别数（1：过早，2：过晚，3：理想）


"""
基站信息
"""
AREA_SCALE_X, AREA_SCALE_Y = 5000, 5000  # 5000m*5000m
NUM_BS = 25  # 25个基站
BS_GRID_SIZE = int(np.sqrt(NUM_BS))  # 每边基站数量，5
BS_SPACING = 833  # 每个基站间距（m）

bs_location_list = []

bs_index = 0
for i in range(BS_GRID_SIZE):  # i 控制 x 方向
    for j in range(BS_GRID_SIZE):  # j 控制 y 方向
        bs_location_x = (i + 1) * BS_SPACING
        bs_location_y = (j + 1) * BS_SPACING
        bs_location_list.extend([bs_index, bs_location_x, bs_location_y])
        bs_index += 1

if env_bool("CGDQN_VERBOSE_CONFIG", False):
    print(f"bs_location_list: {bs_location_list}")
BS_LOCATION_LIST = bs_location_list
BS_TX_POWER = env_float("CGDQN_BS_TX_POWER", 46.0)  # 基站发射功率, dBm
BS_FREQUENCY = 3.5  # 基站工作频率, GHz
SHADOW_SIGMA_DB = env_float("CGDQN_SHADOW_SIGMA_DB", float(np.sqrt(2.0)))
STRESS_CONFIG = os.environ.get("CGDQN_STRESS_CONFIG", "nominal").strip() or "nominal"
INIT_TTT, INIT_HOM = 1000, 0.2  # 初始TTT和HOM参数
FORCED_HANDOVER_RSRP_THRESHOLD = env_float(
    "CGDQN_FORCED_HANDOVER_RSRP_THRESHOLD", -96
)  # 强制切换RSRP阈值
WCFH_DWELL_TICKS = env_int("CGDQN_WCFH_DWELL_TICKS", 3)
WCFH_SAFE_MARGIN_DB = env_float("CGDQN_WCFH_SAFE_MARGIN_DB", 0.5)
WCFH_TARGET_RSRP_MIN = env_float("CGDQN_WCFH_TARGET_RSRP_MIN", -96.0)
ENABLE_RADIO_LINK_TRACE = env_bool("CGDQN_ENABLE_RADIO_LINK_TRACE", False)
RADIO_LINK_TRACE_FINAL_EPOCH_ONLY = env_bool(
    "CGDQN_RADIO_LINK_TRACE_FINAL_EPOCH_ONLY", True
)

"""
用户轨迹信息
"""
PREFER_INTERPOLATED_TRAJECTORY = env_bool("CGDQN_USE_INTERPOLATED_TRAJECTORY", False)
USER_TRAJECTORY_FOLDER_PATH = str(
    trajectory_dir(prefer_interpolated=PREFER_INTERPOLATED_TRAJECTORY)
)
# r"../NS3/20250603_trajectory_all_interpolation_Vel1_Vel3_Vel6"    r"../NS3/20250511_trajectory_all_interpolation"
# USER_TRAJECTORY_FILE_PATHS = glob.glob(USER_TRAJECTORY_FOLDER_PATH + r"/*.csv")

# ==== 用户轨迹加载范围控制 ====
LOAD_USER_START_IDX = env_int(
    "CGDQN_LOAD_USER_START_IDX",
    60,
)  # 起始编号（包含），如60表示从ue_60开始，0-59是Vel1、60-109是Vel3、110-159是Vel6
LOAD_USER_END_IDX = env_int(
    "CGDQN_LOAD_USER_END_IDX", 90
)  # 结束编号（不包含），如110表示截止到ue_109
MAX_STEP_PER_USER = env_int(
    "CGDQN_MAX_STEP_PER_USER", 50000
)  # 每个用户的最大步数 20250502晚增


def parse_user_index_list() -> list[int]:
    inline_value = os.environ.get("CGDQN_LOAD_USER_INDEX_LIST", "").strip()
    file_value = os.environ.get("CGDQN_LOAD_USER_INDEX_FILE", "").strip()
    if inline_value and file_value:
        raise ValueError(
            "Use either CGDQN_LOAD_USER_INDEX_LIST or CGDQN_LOAD_USER_INDEX_FILE, not both."
        )
    if inline_value:
        indices = [int(item.strip()) for item in inline_value.split(",") if item.strip()]
    elif file_value:
        indices = []
        with open(file_value, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "ue_index" in reader.fieldnames:
                for row in reader:
                    value = row.get("ue_index", "").strip()
                    if value:
                        indices.append(int(float(value)))
            else:
                handle.seek(0)
                for line in handle:
                    value = line.strip().strip(",")
                    if value and not value.lower().startswith("ue_index"):
                        indices.append(int(float(value.split(",")[0])))
    else:
        indices = list(range(LOAD_USER_START_IDX, LOAD_USER_END_IDX))
    if not indices:
        raise ValueError("No UE indices selected for CG-DQN run.")
    return indices


USER_INDEX_LIST = parse_user_index_list()
NUM_USER = len(USER_INDEX_LIST)  # 自动计算加载用户数


# def load_user_trajectories_by_range(start_idx, end_idx):
#     user_trajectory_list = []
#     for file_idx in range(start_idx, end_idx):
#         filename = f"{USER_TRAJECTORY_FOLDER_PATH}/yzc_v8_ue_{file_idx}_interpolation.csv"
#         with open(filename, "r", encoding="utf-8") as file:
#             reader = csv.reader(file)
#             next(reader)  # 跳过表头
#             traj = [list(map(float, row[:7])) for row in reader if row]
#             user_trajectory_list.append(traj)
#     return user_trajectory_list  # 只保留前7个字段:Time, UEId, PosX, PosY, VelX, VelY, Direction


def load_user_trajectories_by_indices(indices):
    user_trajectory_list = []
    for file_idx in indices:
        filename = trajectory_file(
            file_idx, prefer_interpolated=PREFER_INTERPOLATED_TRAJECTORY
        )
        if not filename.exists():
            raise FileNotFoundError(
                "Missing trajectory file for UE "
                f"{file_idx}: expected {filename}. "
                "Set CGDQN_TRAJECTORY_DIR or CGDQN_USE_INTERPOLATED_TRAJECTORY."
            )
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            traj = []
            for step, row in enumerate(reader):
                # 当达到最大步长时，提前终止加载
                if step >= MAX_STEP_PER_USER:
                    break
                traj.append(list(map(float, row[:7])))
            user_trajectory_list.append(traj)
    return user_trajectory_list  # 只保留前7个字段:Time, UEId, PosX, PosY, VelX, VelY, Direction


# 加载指定范围内的插值轨迹数据
USER_TRAJECTORY_LIST = load_user_trajectories_by_indices(USER_INDEX_LIST)

# print(f"USER_TRAJECTORY_LIST: {USER_TRAJECTORY_LIST}")

"""
强化学习模型参数
"""
RL_INPUT_DIM = len(BS_LOCATION_LIST) + 17 + 1  # 最后的1为label
RL_HIDDEN_DIM = (
    len(BS_LOCATION_LIST) * 17
)  # 为了让基站环境信息与UE当前的状态信息形成一一对应关系

action_space = []
for ttt in range(100, 3200, 500):  # TTT以100毫秒为间隔
    for hom in np.arange(0.1, 4.2, 0.5):  # HOM以0.1dB为间隔
        action_space.append((ttt, round(hom, 1)))


RL_ACTION_SPACE = action_space
RL_OUTPUT_DIM = len(RL_ACTION_SPACE)
# print(RL_ACTION_SPACE)

# === ε-greedy 固定探索率参数 ===
# Phase 7W fixes value-based DRL exploration at epsilon=0.1 for DQN/DDQN/D3QN.
