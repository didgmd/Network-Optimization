# -*- coding: utf-8 -*-
import math

# Training
MAX_EPOCHS = 10000
TAU_SOFT_UPDATE = 0.001
MOVING_AVG_WINDOW = 500

# Region parameters
AREA_SIZE_X = 10000  # m
AREA_SIZE_Y = 10000  # m

# Deployment locations
MACRO_BS_1_X, MACRO_BS_1_Y = (
    AREA_SIZE_X * (3 / 5) / 2,
    AREA_SIZE_Y * (2 / 5 + (3 / 5) / 2),
)
MACRO_BS_2_X, MACRO_BS_2_Y = (
    AREA_SIZE_X * (2 / 5 + (3 / 5) / 2),
    AREA_SIZE_Y * (3 / 5) / 2,
)
SMALL_BS_1_X, SMALL_BS_1_Y = (AREA_SIZE_X * (1 / 5) / 2, AREA_SIZE_Y * (1 / 5) / 2)
SMALL_BS_2_X, SMALL_BS_2_Y = (
    AREA_SIZE_X * (1 / 5 + (1 / 5) / 2),
    AREA_SIZE_Y * (1 / 5) / 2,
)
SMALL_BS_3_X, SMALL_BS_3_Y = (
    AREA_SIZE_X * (1 / 5) / 2,
    AREA_SIZE_Y * (1 / 5 + (1 / 5) / 2),
)
SMALL_BS_4_X, SMALL_BS_4_Y = (
    AREA_SIZE_X * (1 / 5 + (1 / 5) / 2),
    AREA_SIZE_Y * (1 / 5 + (1 / 5) / 2),
)

# Height settings
USER_HEIGHT = 1.65
MACRO_BS_HEIGHT = 30.0
SMALL_BS_HEIGHT = 10.0
UAV_MIN_HEIGHT = 50.0
UAV_MAX_HEIGHT = 500.0
UAV_DEFAULT_ALTITUDE = 120.0

# User parameters
NUM_OF_USER = 100
USER_STEP_SPEED = 1.2
USER_STEP_SPEED_X = USER_STEP_SPEED * 0.5
USER_STEP_SPEED_Y = USER_STEP_SPEED * 0.5

# Node counts
NUM_OF_SAT = 1
NUM_OF_MACRO = 2
NUM_OF_SMALL = 4
NUM_OF_UAV = 2

# Bandwidths (Hz)
MACRO_TOTAL_BW_HZ = 40_000_000
SMALL_TOTAL_BW_HZ = 20_000_000
UAV_TOTAL_BW_HZ = 20_000_000
SAT_TOTAL_BW_HZ = 36_000_000

# Noise (Section 3.3.4)
NOISE_PSD_dBm_perHz = -174.0
NOISE_FIGURE_dB = 0.0

# Frequencies (Hz)
MACRO_FREQUENCY = 2.4e9
SMALL_FREQUENCY = 3.5e9
UAV_FREQUENCY = 2.4e9
SAT_FREQUENCY = 12e9

# Transmit power (total, dBm)
MACRO_TX_POWER = 50.0
SMALL_TX_POWER = 36.0
UAV_TX_POWER = 36.0
SAT_TX_POWER = 65.0

# Subcarriers
MACRO_NUM_OF_SC = 1500
SMALL_NUM_OF_SC = 1200
UAV_NUM_OF_SC = 250
SAT_NUM_OF_SC = 400

# Capacities
MACRO_BS_CAPACITY = 64
SMALL_BS_CAPACITY = 16
UAV_BS_CAPACITY = 8
SAT_BS_CAPACITY = 100

# Satellite
SAT_ORBIT_HEIGHT_M = 1.2e6

# Path-loss extra offsets (dB)
DELTA_MACRO_dB = 5.0
DELTA_SMALL_dB = 15.0
DELTA_UAV_dB = 0.0
DELTA_SAT_dB = 20.0

# UAV motion
UAV_ACTION_SPACE = [
    (dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
]
UAV_STEP_SPEED = 10.0
UAV_INIT_HEIGHT = 100.0
UAV_INIT_X_1 = AREA_SIZE_X / 3
UAV_INIT_Y_1 = AREA_SIZE_Y / 2
UAV_INIT_X_2 = AREA_SIZE_X * 2 / 3
UAV_INIT_Y_2 = AREA_SIZE_Y / 2

# Cloud agent settings
CLOUD_INPUT_DIM = (
    (NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV)
    + (NUM_OF_SAT + NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV)
    + 3
)
CLOUD_OUTPUT_DIM = NUM_OF_SAT + NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV
CLOUD_HIDDEN_DIM = CLOUD_INPUT_DIM * CLOUD_OUTPUT_DIM
CLOUD_LEARNING_RATE = 5e-4
CLOUD_EPSILON_START = 1.0
CLOUD_EPSILON_END = 0.05
CLOUD_EPSILON_DECAY_EPOCHS = MAX_EPOCHS
CLOUD_GAMMA = 0.95
CLOUD_ASSOCIATION_PERIOD = 3  # K in the manuscript

# Association stability regularization
CLOUD_SWITCH_PENALTY = 0.012
CLOUD_STICKINESS_BONUS = 0.003

# Macro BS agent settings
MACRO_INPUT_DIM = (MACRO_BS_CAPACITY, 1 + 3)
MACRO_OUTPUT_DIM = MACRO_BS_CAPACITY * 2
MACRO_HIDDEN_DIM = MACRO_INPUT_DIM[0] * MACRO_INPUT_DIM[1] * 4
MACRO_LEARNING_RATE = 5e-4
MACRO_EPSILON_START = 1.0
MACRO_EPSILON_END = 0.05
MACRO_EPSILON_DECAY_EPOCHS = MAX_EPOCHS
MACRO_GAMMA = 0.9

# Small BS agent settings
SMALL_INPUT_DIM = (SMALL_BS_CAPACITY, 1 + 3)
SMALL_OUTPUT_DIM = SMALL_BS_CAPACITY * 2
SMALL_HIDDEN_DIM = SMALL_INPUT_DIM[0] * SMALL_INPUT_DIM[1] * 4
SMALL_LEARNING_RATE = 5e-4
SMALL_EPSILON_START = 1.0
SMALL_EPSILON_END = 0.05
SMALL_EPSILON_DECAY_EPOCHS = MAX_EPOCHS
SMALL_GAMMA = 0.9

# UAV agent settings
UAV_INPUT_DIM = UAV_BS_CAPACITY * 2 + 3
UAV_OUTPUT_DIM = 27
UAV_HIDDEN_DIM = UAV_INPUT_DIM * UAV_OUTPUT_DIM
UAV_LEARNING_RATE = 5e-4
UAV_EPSILON_START = 1.0
UAV_EPSILON_END = 0.05
UAV_EPSILON_DECAY_EPOCHS = MAX_EPOCHS
UAV_GAMMA = 0.8

# Reward weights (Section 4)
CLOUD_REWARD_WEIGHTS = {
    "qos": 0.52,
    "load": 0.35,
    "thr": 0.08,
    "cap": 0.05,
}
EDGE_REWARD_WEIGHTS = {
    "qos": 0.55,
    "fair": 0.38,
    "thr": 0.07,
}


# Reward normalization constants (saturation rate at reference distance)
def _tx_power_per_sc_dbm(total_tx_power_dbm, total_sc):
    total_sc = max(total_sc, 1)
    return total_tx_power_dbm - 10 * math.log10(total_sc)


def _rsrp_dbm(distance_m, frequency_hz, total_tx_power_dbm, total_sc, delta_db):
    fspl = 20 * math.log10(distance_m) + 20 * math.log10(frequency_hz) - 147.55
    pl = fspl + delta_db
    return _tx_power_per_sc_dbm(total_tx_power_dbm, total_sc) - pl


def _saturation_rate_mbps(
    distance_m,
    frequency_hz,
    total_tx_power_dbm,
    total_sc,
    total_bw_hz,
    delta_db,
):
    rsrp = _rsrp_dbm(distance_m, frequency_hz, total_tx_power_dbm, total_sc, delta_db)
    b_sc = total_bw_hz / max(total_sc, 1)
    noise_sc = NOISE_PSD_dBm_perHz + 10 * math.log10(b_sc) + NOISE_FIGURE_dB
    snr_db = rsrp - noise_sc
    snr_linear = 10 ** (snr_db / 10)
    rate_mbps = (total_sc * b_sc * math.log2(1 + snr_linear)) / 1_000_000
    return rate_mbps


_D_MACRO = MACRO_BS_HEIGHT - USER_HEIGHT
_D_SMALL = SMALL_BS_HEIGHT - USER_HEIGHT
_D_UAV = UAV_MIN_HEIGHT - USER_HEIGHT
_D_SAT = SAT_ORBIT_HEIGHT_M

EDGE_THR_NORM_MACRO_Mbps = _saturation_rate_mbps(
    _D_MACRO,
    MACRO_FREQUENCY,
    MACRO_TX_POWER,
    MACRO_NUM_OF_SC,
    MACRO_TOTAL_BW_HZ,
    DELTA_MACRO_dB,
)
EDGE_THR_NORM_SMALL_Mbps = _saturation_rate_mbps(
    _D_SMALL,
    SMALL_FREQUENCY,
    SMALL_TX_POWER,
    SMALL_NUM_OF_SC,
    SMALL_TOTAL_BW_HZ,
    DELTA_SMALL_dB,
)
EDGE_THR_NORM_UAV_Mbps = _saturation_rate_mbps(
    _D_UAV,
    UAV_FREQUENCY,
    UAV_TX_POWER,
    UAV_NUM_OF_SC,
    UAV_TOTAL_BW_HZ,
    DELTA_UAV_dB,
)
SAT_THR_NORM_Mbps = _saturation_rate_mbps(
    _D_SAT,
    SAT_FREQUENCY,
    SAT_TX_POWER,
    SAT_NUM_OF_SC,
    SAT_TOTAL_BW_HZ,
    DELTA_SAT_dB,
)
CLOUD_THR_NORM_Mbps = (
    EDGE_THR_NORM_MACRO_Mbps
    + EDGE_THR_NORM_SMALL_Mbps
    + EDGE_THR_NORM_UAV_Mbps
    + SAT_THR_NORM_Mbps
)

# Required throughput per user (Mbps)
EDGE_THR_REQ_MACRO_Mbps = EDGE_THR_NORM_MACRO_Mbps / MACRO_BS_CAPACITY
EDGE_THR_REQ_SMALL_Mbps = EDGE_THR_NORM_SMALL_Mbps / SMALL_BS_CAPACITY
EDGE_THR_REQ_UAV_Mbps = EDGE_THR_NORM_UAV_Mbps / UAV_BS_CAPACITY
SAT_THR_REQ_Mbps = SAT_THR_NORM_Mbps / SAT_BS_CAPACITY
CLOUD_THR_REQ_Mbps = CLOUD_THR_NORM_Mbps / (
    NUM_OF_SAT + NUM_OF_MACRO + NUM_OF_SMALL + NUM_OF_UAV
)


def print_parameters():
    print("[Training]")
    print(f"MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"TAU_SOFT_UPDATE: {TAU_SOFT_UPDATE}")
    print(f"MOVING_AVG_WINDOW: {MOVING_AVG_WINDOW}")
    print()

    print("[Region]")
    print(f"AREA_SIZE_X: {AREA_SIZE_X}")
    print(f"AREA_SIZE_Y: {AREA_SIZE_Y}")
    print()

    print("[Deployment Coordinates]")
    print(f"MACRO_BS_1_X: {MACRO_BS_1_X}")
    print(f"MACRO_BS_1_Y: {MACRO_BS_1_Y}")
    print(f"MACRO_BS_2_X: {MACRO_BS_2_X}")
    print(f"MACRO_BS_2_Y: {MACRO_BS_2_Y}")
    print(f"SMALL_BS_1_X: {SMALL_BS_1_X}")
    print(f"SMALL_BS_1_Y: {SMALL_BS_1_Y}")
    print(f"SMALL_BS_2_X: {SMALL_BS_2_X}")
    print(f"SMALL_BS_2_Y: {SMALL_BS_2_Y}")
    print(f"SMALL_BS_3_X: {SMALL_BS_3_X}")
    print(f"SMALL_BS_3_Y: {SMALL_BS_3_Y}")
    print(f"SMALL_BS_4_X: {SMALL_BS_4_X}")
    print(f"SMALL_BS_4_Y: {SMALL_BS_4_Y}")
    print()

    print("[Heights]")
    print(f"USER_HEIGHT: {USER_HEIGHT}")
    print(f"MACRO_BS_HEIGHT: {MACRO_BS_HEIGHT}")
    print(f"SMALL_BS_HEIGHT: {SMALL_BS_HEIGHT}")
    print(f"UAV_MIN_HEIGHT: {UAV_MIN_HEIGHT}")
    print(f"UAV_MAX_HEIGHT: {UAV_MAX_HEIGHT}")
    print(f"UAV_DEFAULT_ALTITUDE: {UAV_DEFAULT_ALTITUDE}")
    print(f"SAT_ORBIT_HEIGHT_M: {SAT_ORBIT_HEIGHT_M}")
    print()

    print("[Users]")
    print(f"NUM_OF_USER: {NUM_OF_USER}")
    print(f"USER_STEP_SPEED: {USER_STEP_SPEED}")
    print(f"USER_STEP_SPEED_X: {USER_STEP_SPEED_X}")
    print(f"USER_STEP_SPEED_Y: {USER_STEP_SPEED_Y}")
    print()

    print("[Node Counts]")
    print(f"NUM_OF_SAT: {NUM_OF_SAT}")
    print(f"NUM_OF_MACRO: {NUM_OF_MACRO}")
    print(f"NUM_OF_SMALL: {NUM_OF_SMALL}")
    print(f"NUM_OF_UAV: {NUM_OF_UAV}")
    print()

    print("[Bandwidths Hz]")
    print(f"MACRO_TOTAL_BW_HZ: {MACRO_TOTAL_BW_HZ}")
    print(f"SMALL_TOTAL_BW_HZ: {SMALL_TOTAL_BW_HZ}")
    print(f"UAV_TOTAL_BW_HZ: {UAV_TOTAL_BW_HZ}")
    print(f"SAT_TOTAL_BW_HZ: {SAT_TOTAL_BW_HZ}")
    print()

    print("[Noise]")
    print(f"NOISE_PSD_dBm_perHz: {NOISE_PSD_dBm_perHz}")
    print(f"NOISE_FIGURE_dB: {NOISE_FIGURE_dB}")
    print()

    print("[Frequencies Hz]")
    print(f"MACRO_FREQUENCY: {MACRO_FREQUENCY}")
    print(f"SMALL_FREQUENCY: {SMALL_FREQUENCY}")
    print(f"UAV_FREQUENCY: {UAV_FREQUENCY}")
    print(f"SAT_FREQUENCY: {SAT_FREQUENCY}")
    print()

    print("[Transmit Power dBm (Total)]")
    print(f"MACRO_TX_POWER: {MACRO_TX_POWER}")
    print(f"SMALL_TX_POWER: {SMALL_TX_POWER}")
    print(f"UAV_TX_POWER: {UAV_TX_POWER}")
    print(f"SAT_TX_POWER: {SAT_TX_POWER}")
    print()

    print("[Subcarriers]")
    print(f"MACRO_NUM_OF_SC: {MACRO_NUM_OF_SC}")
    print(f"SMALL_NUM_OF_SC: {SMALL_NUM_OF_SC}")
    print(f"UAV_NUM_OF_SC: {UAV_NUM_OF_SC}")
    print(f"SAT_NUM_OF_SC: {SAT_NUM_OF_SC}")
    print()

    print("[Capacities]")
    print(f"MACRO_BS_CAPACITY: {MACRO_BS_CAPACITY}")
    print(f"SMALL_BS_CAPACITY: {SMALL_BS_CAPACITY}")
    print(f"UAV_BS_CAPACITY: {UAV_BS_CAPACITY}")
    print(f"SAT_BS_CAPACITY: {SAT_BS_CAPACITY}")
    print()

    print("[Path Loss Offsets dB]")
    print(f"DELTA_MACRO_dB: {DELTA_MACRO_dB}")
    print(f"DELTA_SMALL_dB: {DELTA_SMALL_dB}")
    print(f"DELTA_UAV_dB: {DELTA_UAV_dB}")
    print(f"DELTA_SAT_dB: {DELTA_SAT_dB}")
    print()

    print("[UAV Motion]")
    print(f"UAV_STEP_SPEED: {UAV_STEP_SPEED}")
    print(f"UAV_INIT_HEIGHT: {UAV_INIT_HEIGHT}")
    print(f"UAV_INIT_X_1: {UAV_INIT_X_1}")
    print(f"UAV_INIT_Y_1: {UAV_INIT_Y_1}")
    print(f"UAV_INIT_X_2: {UAV_INIT_X_2}")
    print(f"UAV_INIT_Y_2: {UAV_INIT_Y_2}")
    print(f"UAV_ACTION_SPACE_SIZE: {len(UAV_ACTION_SPACE)}")
    print()

    print("[Cloud Agent]")
    print(f"CLOUD_INPUT_DIM: {CLOUD_INPUT_DIM}")
    print(f"CLOUD_OUTPUT_DIM: {CLOUD_OUTPUT_DIM}")
    print(f"CLOUD_HIDDEN_DIM: {CLOUD_HIDDEN_DIM}")
    print(f"CLOUD_LEARNING_RATE: {CLOUD_LEARNING_RATE}")
    print(f"CLOUD_GAMMA: {CLOUD_GAMMA}")
    print(f"CLOUD_EPSILON_START: {CLOUD_EPSILON_START}")
    print(f"CLOUD_EPSILON_END: {CLOUD_EPSILON_END}")
    print(f"CLOUD_EPSILON_DECAY_EPOCHS: {CLOUD_EPSILON_DECAY_EPOCHS}")
    print(f"CLOUD_ASSOCIATION_PERIOD: {CLOUD_ASSOCIATION_PERIOD}")
    print()

    print("[Macro Agent]")
    print(f"MACRO_INPUT_DIM: {MACRO_INPUT_DIM}")
    print(f"MACRO_OUTPUT_DIM: {MACRO_OUTPUT_DIM}")
    print(f"MACRO_HIDDEN_DIM: {MACRO_HIDDEN_DIM}")
    print(f"MACRO_LEARNING_RATE: {MACRO_LEARNING_RATE}")
    print(f"MACRO_GAMMA: {MACRO_GAMMA}")
    print(f"MACRO_EPSILON_START: {MACRO_EPSILON_START}")
    print(f"MACRO_EPSILON_END: {MACRO_EPSILON_END}")
    print(f"MACRO_EPSILON_DECAY_EPOCHS: {MACRO_EPSILON_DECAY_EPOCHS}")
    print()

    print("[Small Agent]")
    print(f"SMALL_INPUT_DIM: {SMALL_INPUT_DIM}")
    print(f"SMALL_OUTPUT_DIM: {SMALL_OUTPUT_DIM}")
    print(f"SMALL_HIDDEN_DIM: {SMALL_HIDDEN_DIM}")
    print(f"SMALL_LEARNING_RATE: {SMALL_LEARNING_RATE}")
    print(f"SMALL_GAMMA: {SMALL_GAMMA}")
    print(f"SMALL_EPSILON_START: {SMALL_EPSILON_START}")
    print(f"SMALL_EPSILON_END: {SMALL_EPSILON_END}")
    print(f"SMALL_EPSILON_DECAY_EPOCHS: {SMALL_EPSILON_DECAY_EPOCHS}")
    print()

    print("[UAV Agent]")
    print(f"UAV_INPUT_DIM: {UAV_INPUT_DIM}")
    print(f"UAV_OUTPUT_DIM: {UAV_OUTPUT_DIM}")
    print(f"UAV_HIDDEN_DIM: {UAV_HIDDEN_DIM}")
    print(f"UAV_LEARNING_RATE: {UAV_LEARNING_RATE}")
    print(f"UAV_GAMMA: {UAV_GAMMA}")
    print(f"UAV_EPSILON_START: {UAV_EPSILON_START}")
    print(f"UAV_EPSILON_END: {UAV_EPSILON_END}")
    print(f"UAV_EPSILON_DECAY_EPOCHS: {UAV_EPSILON_DECAY_EPOCHS}")
    print()

    print("[Reward Weights]")
    print(f"CLOUD_REWARD_WEIGHTS: {CLOUD_REWARD_WEIGHTS}")
    print(f"EDGE_REWARD_WEIGHTS: {EDGE_REWARD_WEIGHTS}")
    print(f"SWITCH_PENALTY: {CLOUD_SWITCH_PENALTY}")
    print(f"STICKINESS_BONUS: {CLOUD_STICKINESS_BONUS}")
    print()

    print("[Derived Rates Mbps]")
    print(f"EDGE_THR_NORM_MACRO_Mbps: {EDGE_THR_NORM_MACRO_Mbps:.2f}")
    print(f"EDGE_THR_NORM_SMALL_Mbps: {EDGE_THR_NORM_SMALL_Mbps:.2f}")
    print(f"EDGE_THR_NORM_UAV_Mbps: {EDGE_THR_NORM_UAV_Mbps:.2f}")
    print(f"SAT_THR_NORM_Mbps: {SAT_THR_NORM_Mbps:.2f}")
    print(f"CLOUD_THR_NORM_Mbps: {CLOUD_THR_NORM_Mbps:.2f}")
    print(f"EDGE_THR_REQ_MACRO_Mbps: {EDGE_THR_REQ_MACRO_Mbps:.2f}")
    print(f"EDGE_THR_REQ_SMALL_Mbps: {EDGE_THR_REQ_SMALL_Mbps:.2f}")
    print(f"EDGE_THR_REQ_UAV_Mbps: {EDGE_THR_REQ_UAV_Mbps:.2f}")
    print(f"SAT_THR_REQ_Mbps: {SAT_THR_REQ_Mbps:.2f}")
    print(f"CLOUD_THR_REQ_Mbps: {CLOUD_THR_REQ_Mbps:.2f}")
