# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime
import csv
from pathlib import Path
import torch
import numpy as np
from DebugPrint import debug, debug_print, set_debug_mode
from Parameters import (
    BS_LOCATION_LIST,
    BS_TX_POWER,
    SHADOW_SIGMA_DB,
    STRESS_CONFIG,
    NUM_USER,
    USER_TRAJECTORY_LIST,
    RL_INPUT_DIM,
    RL_HIDDEN_DIM,
    RL_OUTPUT_DIM,
    RL_ALPHA,
    RL_EPSILON,
    RL_GAMMA,
    AREA_SCALE_X,
    AREA_SCALE_Y,
    INIT_TTT,
    INIT_HOM,
    RL_ACTION_SPACE,
    MAX_STEP_PER_USER,
    FORCED_HANDOVER_RSRP_THRESHOLD,
    WCFH_DWELL_TICKS,
    WCFH_SAFE_MARGIN_DB,
    WCFH_TARGET_RSRP_MIN,
    RLF_SINR_THRESHOLD_DB,
    ENABLE_RADIO_LINK_TRACE,
    RADIO_LINK_TRACE_FINAL_EPOCH_ONLY,
    LOAD_USER_START_IDX,
    LOAD_USER_END_IDX,
    USER_INDEX_LIST,
    RL_MAX_EPOCHS,
    USER_TRAJECTORY_FOLDER_PATH,
    DDQN_TARGET_UPDATE_STEPS,
)
from Classes import ActorCritic, DQN, DuelingDQN, User, BS
from ActionChooser import choose_action, choose_ppo_action
from ExperienceReplayBuffer import ExperienceReplayBuffer, ReplayTransition
from catboost import CatBoostClassifier, Pool
from Formular import average_rsrp_calculation, estimate_throughput
from copy import deepcopy
from RewardCalculator import calculate_reward, outcome_based_reward
from catboost_feature_schema import CATBOOST_CATEGORICAL_FEATURES, CATBOOST_FEATURE_COLUMNS
from experiment_contract import LOG_ROOT, MODEL_PATH, contract_snapshot, env_int, write_json
from experiment_contract import env_bool
from rl_training_config import (
    ACTION_PRIOR_BETA,
    EXPLORATION_CATEGORICAL_POLICY_SAMPLING,
    EXPLORATION_EPSILON_GREEDY_FIXED,
    OUTCOME_REWARD_PROFILE,
    REPLAY_BATCH_SIZE,
    REPLAY_CAPACITY,
    REPLAY_ENABLED,
    REPLAY_SAMPLE_POLICY,
    REPLAY_UPDATES_PER_STEP,
    REPLAY_WARMUP_EPOCHS,
    REPLAY_WARMUP_TRANSITIONS,
    SMOOTH_L1_BETA,
    TARGET_UPDATE_MODE,
    VALUE_TD_LOSS,
    VALUE_TD_LOSS_MSE,
    VALUE_TD_LOSS_SMOOTH_L1,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_VALUE_COEF,
    REWARD_FAMILY_LEGACY_LABEL_SHAPED,
    REWARD_FAMILY_NONE,
    REWARD_FAMILY_OUTCOME,
    REWARD_FAMILY_RSRP,
    reward_weights_snapshot,
)

set_debug_mode(False)

# 控制是否启用RSRP奖励逻辑（用于调试或实验比较）True是启用
FLAG_REWARD_RSRP = True
DISABLE_LOSS_EARLY_STOP = env_bool("CGDQN_DISABLE_LOSS_EARLY_STOP", False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dl_model = torch.load("model.pth", weights_only=False)
# dl_model.load_state_dict(torch.load("model_params.pth"))

# 加载模型
dl_model = CatBoostClassifier()
dl_model.load_model(str(MODEL_PATH))
CAT_FEATURE_INDEX = [
    CATBOOST_FEATURE_COLUMNS.index(column) for column in CATBOOST_CATEGORICAL_FEATURES
]

VALID_VARIANTS = {
    "fth": {
        "uses_dqn": False,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_NONE,
        "policy": "fixed_threshold",
    },
    "fth_wcfh": {
        "uses_dqn": False,
        "uses_catboost_label": False,
        "uses_wcfh": True,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_NONE,
        "policy": "fixed_threshold_with_wcfh",
    },
    "adaptive_heuristic": {
        "uses_dqn": False,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_RSRP,
        "policy": "rsrp_adaptive_heuristic",
    },
    "vanilla_dqn": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_RSRP,
        "policy": "dqn_label_masked",
    },
    "double_dqn": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": True,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_RSRP,
        "policy": "double_dqn_label_masked",
    },
    "cg_dqn_no_wcfh": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_LEGACY_LABEL_SHAPED,
        "policy": "catboost_guided_dqn_without_wcfh",
    },
    "full_cg_dqn": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": True,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_LEGACY_LABEL_SHAPED,
        "policy": "catboost_guided_dqn_with_wcfh",
    },
    "dqn_outcome_reward": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "dqn_outcome_reward",
    },
    "ddqn_outcome_reward": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": True,
        "uses_dueling_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "ddqn_outcome_reward",
    },
    "d3qn_a3": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": True,
        "uses_dueling_dqn": True,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "dueling_ddqn_a3",
    },
    "cg_state_only_dqn": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "catboost_state_only_dqn",
    },
    "cg_prior_dqn": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": True,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "catboost_prior_dqn",
    },
    "cg_prior_ddqn": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": True,
        "uses_dueling_dqn": False,
        "uses_ppo": False,
        "uses_action_prior": True,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "catboost_prior_ddqn",
    },
    "cg_prior_d3qn": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": True,
        "uses_dueling_dqn": True,
        "uses_ppo": False,
        "uses_action_prior": True,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "catboost_prior_dueling_ddqn_a3",
    },
    "ppo_a3": {
        "uses_dqn": True,
        "uses_catboost_label": False,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": True,
        "uses_action_prior": False,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "categorical_ppo_a3",
    },
    "cg_prior_ppo": {
        "uses_dqn": True,
        "uses_catboost_label": True,
        "uses_wcfh": False,
        "uses_double_dqn": False,
        "uses_ppo": True,
        "uses_action_prior": True,
        "reward_family": REWARD_FAMILY_OUTCOME,
        "policy": "catboost_prior_categorical_ppo_a3",
    },
}

DQN_LOG_REWARD_INDEX = 25
DQN_LOG_REWARD_COMPONENT_START_INDEX = 26
DQN_LOG_REPLAY_START_INDEX = 36
ACTION_INDEX_BY_VALUE = {tuple(action): index for index, action in enumerate(RL_ACTION_SPACE)}


def value_td_loss_fn():
    if VALUE_TD_LOSS == VALUE_TD_LOSS_SMOOTH_L1:
        return torch.nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    if VALUE_TD_LOSS == VALUE_TD_LOSS_MSE:
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported value TD loss: {VALUE_TD_LOSS}")


def optimize_value_batch(
    *,
    dqn,
    target_dqn,
    optimizer,
    transitions: list[ReplayTransition],
    uses_double_dqn: bool,
    loss_fn,
    device,
) -> torch.Tensor:
    states = torch.tensor(
        [transition.state for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.tensor(
        [transition.next_state for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        [transition.action_index for transition in transitions],
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    q_estimates = dqn(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        if uses_double_dqn:
            next_actions = dqn(next_states).argmax(dim=1, keepdim=True)
            next_q = target_dqn(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q = target_dqn(next_states).max(dim=1).values
        q_targets = rewards + RL_GAMMA * next_q

    loss = loss_fn(q_estimates, q_targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
    optimizer.step()
    return loss

OBJECTIVE_RSRP_THRESHOLD = -90.0
NEUTRAL_LABEL = 2
def experiment_variant() -> str:
    variant = os.environ.get("CGDQN_VARIANT", "full_cg_dqn").strip()
    if variant not in VALID_VARIANTS:
        allowed = ", ".join(sorted(VALID_VARIANTS))
        raise ValueError(f"Unsupported CGDQN_VARIANT={variant!r}. Allowed: {allowed}")
    return variant


def run_root() -> Path:
    return Path(os.environ.get("CGDQN_RUN_ROOT", LOG_ROOT)).resolve()


def ppho_window_ms() -> int:
    return env_int("CGDQN_PPHO_WINDOW_MS", 2000)


def apply_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_catboost_input(user, user_idx: int, user_step: int) -> list:
    pre_mean_source_rsrp = (
        float(user.avg_rsrp_in_dbm)
        if user.avg_rsrp_in_dbm is not None
        else float(user.source_bs_rsrp)
    )
    source_rsrp = float(user.source_bs_rsrp)
    target_rsrp = float(user.target_bs_rsrp)
    values = {
        "SourceBsId": user.source_bs_id,
        "TargetBsId": user.target_bs_id,
        "SourceRsrp": source_rsrp,
        "TargetRsrp": target_rsrp,
        "SourceSinr": user.source_bs_sinr,
        "TargetSinr": user.target_bs_sinr,
        "TargetDis": user.target_bs_distance,
        "VelX": USER_TRAJECTORY_LIST[user_idx][user_step][4],
        "VelY": USER_TRAJECTORY_LIST[user_idx][user_step][5],
        "Direction": USER_TRAJECTORY_LIST[user_idx][user_step][6],
        "TargetSourceRsrpGap": target_rsrp - source_rsrp,
        "PreMeanSourceRsrp": pre_mean_source_rsrp,
        "SourceRsrpDrop": pre_mean_source_rsrp - source_rsrp,
    }
    return [values[column] for column in CATBOOST_FEATURE_COLUMNS]


def predict_catboost_label(user, user_idx: int, user_step: int) -> int:
    dl_input_np = np.array(
        [build_catboost_input(user, user_idx, user_step)],
        dtype=object,
    )
    for idx in CAT_FEATURE_INDEX:
        dl_input_np[:, idx] = dl_input_np[:, idx].astype(int)
    dl_input_pool = Pool(dl_input_np, cat_features=CAT_FEATURE_INDEX)
    return int(dl_model.predict(dl_input_pool)[0][0])


def objective_handover_label(rsrp_history: list[float], current_rsrp: float) -> int:
    previous = rsrp_history[:-1] if len(rsrp_history) > 1 else rsrp_history
    avg_rsrp = float(np.mean(previous)) if previous else float(current_rsrp)
    if avg_rsrp < OBJECTIVE_RSRP_THRESHOLD:
        return 1
    if current_rsrp < avg_rsrp and avg_rsrp >= OBJECTIVE_RSRP_THRESHOLD:
        return 0
    return 2


def nearest_action(ttt: int, hom: float) -> tuple[int, float]:
    return min(RL_ACTION_SPACE, key=lambda item: abs(item[0] - ttt) + abs(item[1] - hom))


def adaptive_heuristic_action(user) -> tuple[int, float]:
    delta = user.target_bs_rsrp - user.source_bs_rsrp
    ttt, hom = user.ttt_determined, user.hom_determined
    if user.source_bs_rsrp < FORCED_HANDOVER_RSRP_THRESHOLD or delta >= hom + 2.0:
        ttt -= 500
        hom -= 0.5
    elif delta <= hom:
        ttt += 500
        hom += 0.5
    return nearest_action(ttt, hom)


def rsrp_control_reward(prev_rsrp: float, post_rsrp: float, handover_executed: bool) -> float:
    if not handover_executed:
        return -0.05
    return max(min((post_rsrp - prev_rsrp) / 10.0, 1.0), -1.0)


def classify_wcfh_guard(user, weak_coverage_dwell_ticks: int) -> tuple[bool, str]:
    if weak_coverage_dwell_ticks < WCFH_DWELL_TICKS:
        return False, "dwell"
    if user.source_bs_rsrp >= FORCED_HANDOVER_RSRP_THRESHOLD:
        return False, "not_weak"
    if user.target_bs_rsrp <= user.source_bs_rsrp + WCFH_SAFE_MARGIN_DB:
        return False, "margin"
    if user.target_bs_rsrp <= WCFH_TARGET_RSRP_MIN:
        return False, "target_min"
    return True, "accepted"


RADIO_LINK_TRACE_HEADER = [
    "Epoch",
    "UserId",
    "Variant",
    "Time",
    "UserStep",
    "PosX",
    "PosY",
    "SourceBsId",
    "SourceSinr",
    "SourceRsrp",
    "TargetBsId",
    "TargetSinr",
    "TargetRsrp",
    "TTT",
    "HOM",
    "TTTCountdown",
    "StandardA3Triggered",
    "WCFHRawCandidate",
    "WCFHAccepted",
    "WCFHRejectReason",
    "HandoverExecuted",
    "RlfSinrThresholdDb",
    "SinrOutageRlf",
]


def radio_link_trace_enabled(epoch: int) -> bool:
    if not ENABLE_RADIO_LINK_TRACE:
        return False
    return (not RADIO_LINK_TRACE_FINAL_EPOCH_ONLY) or epoch == RL_MAX_EPOCHS


def append_radio_link_trace(
    rows: list[list],
    *,
    epoch: int,
    ue_id: int,
    variant: str,
    sample_time: float,
    user_step: int,
    pos_x: float,
    pos_y: float,
    user,
    standard_a3_triggered: bool,
    raw_wcfh_candidate: bool,
    wcfh_accepted: bool,
    wcfh_reject_reason: str,
    handover_executed: bool,
) -> None:
    rows.append(
        [
            epoch,
            ue_id,
            variant,
            sample_time,
            user_step,
            pos_x,
            pos_y,
            user.source_bs_id,
            user.source_bs_sinr,
            user.source_bs_rsrp,
            user.target_bs_id,
            user.target_bs_sinr,
            user.target_bs_rsrp,
            user.ttt_determined,
            user.hom_determined,
            user.ttt_countdown,
            int(bool(standard_a3_triggered)),
            int(bool(raw_wcfh_candidate)),
            int(bool(wcfh_accepted)),
            wcfh_reject_reason,
            int(bool(handover_executed)),
            RLF_SINR_THRESHOLD_DB,
            int(float(user.source_bs_sinr) < RLF_SINR_THRESHOLD_DB),
        ]
    )


def rl(run_name=None):
    variant = experiment_variant()
    variant_config = VALID_VARIANTS[variant]
    random_seed = env_int("CGDQN_RANDOM_SEED", 42)
    apply_random_seed(random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = run_name or os.environ.get("CGDQN_RUN_NAME", variant)
    log_base_dir = run_root() / f"{timestamp}_{run_suffix}"
    os.makedirs(log_base_dir, exist_ok=True)
    write_json(
        log_base_dir / "run_contract.json",
        {
            **contract_snapshot(),
            "rl": {
                "load_user_start_idx": LOAD_USER_START_IDX,
                "load_user_end_idx": LOAD_USER_END_IDX,
                "load_user_index_list": USER_INDEX_LIST,
                "max_step_per_user": MAX_STEP_PER_USER,
                "max_epochs": RL_MAX_EPOCHS,
                "variant": variant,
                "variant_config": variant_config,
                "random_seed": random_seed,
                "ppho_window_ms": ppho_window_ms(),
                "trajectory_folder": USER_TRAJECTORY_FOLDER_PATH,
                "catboost_model": str(MODEL_PATH),
                "catboost_feature_schema": "causal_v2",
                "wcfh_dwell_ticks": WCFH_DWELL_TICKS,
                "wcfh_safe_margin_db": WCFH_SAFE_MARGIN_DB,
                "wcfh_target_rsrp_min": WCFH_TARGET_RSRP_MIN,
                "rlf_sinr_threshold_db": RLF_SINR_THRESHOLD_DB,
                "enable_radio_link_trace": ENABLE_RADIO_LINK_TRACE,
                "radio_link_trace_final_epoch_only": RADIO_LINK_TRACE_FINAL_EPOCH_ONLY,
                "ddqn_target_update_steps": DDQN_TARGET_UPDATE_STEPS,
                "optimizer": "Adam" if variant_config["uses_dqn"] else "",
                "value_td_loss": VALUE_TD_LOSS if not variant_config.get("uses_ppo", False) else "",
                "smooth_l1_beta": (
                    SMOOTH_L1_BETA if VALUE_TD_LOSS == VALUE_TD_LOSS_SMOOTH_L1 else ""
                ),
                "target_update_mode": (
                    TARGET_UPDATE_MODE if not variant_config.get("uses_ppo", False) else ""
                ),
                "target_update_steps": (
                    DDQN_TARGET_UPDATE_STEPS if not variant_config.get("uses_ppo", False) else ""
                ),
                "replay_enabled": (
                    bool(REPLAY_ENABLED and not variant_config.get("uses_ppo", False))
                    if variant_config["uses_dqn"]
                    else False
                ),
                "replay_capacity": REPLAY_CAPACITY,
                "replay_batch_size": REPLAY_BATCH_SIZE,
                "replay_warmup_transitions": REPLAY_WARMUP_TRANSITIONS,
                "replay_warmup_epochs": REPLAY_WARMUP_EPOCHS,
                "replay_updates_per_step": REPLAY_UPDATES_PER_STEP,
                "replay_sample_policy": REPLAY_SAMPLE_POLICY,
                "exploration_policy": (
                    EXPLORATION_CATEGORICAL_POLICY_SAMPLING
                    if variant_config.get("uses_ppo", False)
                    else EXPLORATION_EPSILON_GREEDY_FIXED
                ),
                "epsilon": "" if variant_config.get("uses_ppo", False) else RL_EPSILON,
                "rl_alpha": RL_ALPHA,
                "rl_gamma": RL_GAMMA,
                "action_prior_beta": ACTION_PRIOR_BETA,
                "ppo_clip_eps": PPO_CLIP_EPS,
                "ppo_value_coef": PPO_VALUE_COEF,
                "ppo_entropy_coef": PPO_ENTROPY_COEF,
                "outcome_reward_profile": OUTCOME_REWARD_PROFILE,
                "disable_loss_early_stop": DISABLE_LOSS_EARLY_STOP,
                "outcome_reward_weights": reward_weights_snapshot(),
                "stress_config": STRESS_CONFIG,
                "bs_tx_power_dbm": BS_TX_POWER,
                "shadow_sigma_db": SHADOW_SIGMA_DB,
            },
        },
    )
    if variant_config.get("uses_ppo", False):
        dqn = ActorCritic(RL_INPUT_DIM, RL_HIDDEN_DIM, RL_OUTPUT_DIM).to(device)
        target_dqn = None
    else:
        q_network = DuelingDQN if variant_config.get("uses_dueling_dqn", False) else DQN
        dqn = q_network(RL_INPUT_DIM, RL_HIDDEN_DIM, RL_OUTPUT_DIM).to(device)
        target_dqn = q_network(RL_INPUT_DIM, RL_HIDDEN_DIM, RL_OUTPUT_DIM).to(device)
        target_dqn.load_state_dict(dqn.state_dict())
        target_dqn.eval()
    rl_optimizer = (
        torch.optim.Adam(dqn.parameters(), lr=RL_ALPHA)
        if variant_config["uses_dqn"]
        else None
    )
    if TARGET_UPDATE_MODE != "hard":
        raise ValueError(f"Unsupported target update mode: {TARGET_UPDATE_MODE}")
    if REPLAY_SAMPLE_POLICY != "uniform":
        raise ValueError(f"Unsupported replay sample policy: {REPLAY_SAMPLE_POLICY}")
    value_loss_fn = value_td_loss_fn()
    value_based_variant = variant_config["uses_dqn"] and not variant_config.get("uses_ppo", False)
    replay_buffer = (
        ExperienceReplayBuffer(REPLAY_CAPACITY)
        if REPLAY_ENABLED and value_based_variant
        else None
    )
    dqn_update_count = 0

    # 创建基站对象并存入列表 eg: BS(0, 1000, 1000,……)
    bs_list = []
    for bs_idx in range(int(len(BS_LOCATION_LIST) / 3)):
        bs_list.append(
            BS(
                BS_LOCATION_LIST[bs_idx * 3],
                BS_LOCATION_LIST[bs_idx * 3 + 1],
                BS_LOCATION_LIST[bs_idx * 3 + 2],
            )
        )
    epsilon = RL_EPSILON  # Phase 7W: fixed value-based exploration rate.

    epoch = 0
    # 定义损失记录列表
    list_dqn_loss = []
    # list_dqn_loss_comparison = []  # 用于比较损失的列表
    dqn_loss = 0.0

    while True:
        debug_print(f"################ Epoch {epoch} ################")

        epoch_dqn_loss_list = []  # 每轮初始化
        # ==== 初始化性能统计指标 ====
        early_ho_count = 0
        late_ho_count = 0
        ideal_ho_count = 0
        hof_count = 0
        ppho_count = 0
        rlf_count = 0
        total_ho_attempts = 0
        completed_ho_count = 0
        rsrp_gain_sum = 0.0
        wcfh_raw_candidate_count = 0
        wcfh_accepted_count = 0
        wcfh_rejected_margin_count = 0
        wcfh_rejected_target_min_count = 0

        # 吞吐量段统计（按用户每1万步划分）
        segment_step = 10000
        user_segment_throughput = {}  # 每个用户的段吞吐量 list

        for user_idx in range(NUM_USER):  # 遍历每个用户
            debug(f"################ Episode: User Index {user_idx} ################")
            ue_id = USER_INDEX_LIST[user_idx]

            user = User(USER_TRAJECTORY_LIST[user_idx])  # 创建当前用户对象
            curr_state, next_state = [], []  # 初始化当前状态与下个状态
            dummy_dl_source_bs_rsrp_list = []
            q_estimate, q_estimate, dqn_loss = 0.0, 0.0, 0.0  # 初始化Q值与损失函数
            dqn_log_list = []
            ho_log_list = []
            radio_link_trace_list = []
            last_departed_bs_id = None
            last_handover_time = None
            weak_coverage_dwell_ticks = 0
            # cumulative_reward = 0.0  # 在主循环开始前初始化累计reward变量，用于看奖励函数大致分布情况
            segment_sinr_sum = 0.0  # 吞吐量相关
            segment_step_count = 0  # 吞吐量相关
            segment_list = []  # 吞吐量相关

            # 遍历每个用户的轨迹
            # for user_step in range(len(USER_TRAJECTORY_LIST[user_idx]) - 1):
            # 遍历每个用户的轨迹(限制最大步数),20250502晚增
            for user_step in range(
                min(len(USER_TRAJECTORY_LIST[user_idx]) - 1, MAX_STEP_PER_USER)
            ):
                # if user_step % 10000 == 0:
                #     debug_print(
                #         f"#################### Episode: User {user_idx} Step {user_step} ####################"
                #         # f"Cumulative Reward: {cumulative_reward:.2f} ####################"
                #     )
                # # 清空累计reward
                # cumulative_reward = 0.0

                just_reached_zero_flag = False  # 20250520增加 控制切换逻辑
                flag_dqn_triggered = False  # DQN触发标记
                previous_action = None
                a3_satisfied = False
                dl_predicted_label_before_handover = None  # 用于记录切换前的预测标签
                dl_predicted = None
                raw_wcfh_candidate = False
                wcfh_accepted = False
                wcfh_reject_reason = ""

                if user_step == 0:  # 初始状态
                    user.get_curr_location(user_step)  # 获取用户当前位置
                    # debug(f"User initial position: ({user.curr_x}, {user.curr_y})")
                    min_distance = AREA_SCALE_X * AREA_SCALE_Y  # 初始化最小距离
                    min_distance_bs_idx = None  # 初始化最小距离基站索引
                    for bs_idx in range(len(bs_list)):
                        distance = (
                            (user.curr_x - bs_list[bs_idx].bs_x) ** 2
                            + (user.curr_y - bs_list[bs_idx].bs_y) ** 2
                        ) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            min_distance_bs_idx = bs_idx
                    debug(f"min_distance_bs_idx: {min_distance_bs_idx} from source bs")
                    user.set_source_bs(
                        bs_list[min_distance_bs_idx], bs_list
                    )  # 设定服务基站
                    debug(
                        f"source_bs_id: {user.source_bs_id}, "
                        f"source_bs_rsrp: {user.source_bs_rsrp}, "
                        f"source_bs_distance: {user.source_bs_distance}, "
                        f"source_bs_pl: {user.source_bs_pl}, "
                        f"source_bs_sinr: {user.source_bs_sinr}"
                    )
                    min_distance = AREA_SCALE_X * AREA_SCALE_Y  # 初始化最小距离
                    min_distance_bs_idx = None  # 初始化最小距离基站索引
                    for bs_idx in range(len(bs_list)):
                        if bs_list[bs_idx].bs_id == user.source_bs_id:
                            continue
                        distance = (
                            (user.curr_x - bs_list[bs_idx].bs_x) ** 2
                            + (user.curr_y - bs_list[bs_idx].bs_y) ** 2
                        ) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            min_distance_bs_idx = bs_idx
                    debug(f"min_distance_bs_idx: {min_distance_bs_idx} from target bs")
                    user.set_target_bs(
                        bs_list[min_distance_bs_idx], bs_list
                    )  # 设定目标基站
                    debug(
                        f"target_bs_id: {user.target_bs_id}, "
                        f"target_bs_rsrp: {user.target_bs_rsrp}, "
                        f"target_bs_distance: {user.target_bs_distance}, "
                        f"target_bs_pl: {user.target_bs_pl}, "
                        f"target_bs_sinr: {user.target_bs_sinr}"
                    )

                # 检查HOM并更新TTT
                if user.source_bs_rsrp < FORCED_HANDOVER_RSRP_THRESHOLD:
                    weak_coverage_dwell_ticks += 1
                else:
                    weak_coverage_dwell_ticks = 0

                standard_a3_triggered = (
                    user.target_bs_rsrp - user.source_bs_rsrp > user.hom_determined
                )
                if variant_config["uses_wcfh"] and not standard_a3_triggered:
                    wcfh_accepted, wcfh_reject_reason = classify_wcfh_guard(
                        user, weak_coverage_dwell_ticks
                    )
                    raw_wcfh_candidate = (
                        weak_coverage_dwell_ticks >= WCFH_DWELL_TICKS
                        and user.source_bs_rsrp < FORCED_HANDOVER_RSRP_THRESHOLD
                    )
                    if raw_wcfh_candidate:
                        wcfh_raw_candidate_count += 1
                        if wcfh_accepted:
                            wcfh_accepted_count += 1
                        elif wcfh_reject_reason == "margin":
                            wcfh_rejected_margin_count += 1
                        elif wcfh_reject_reason == "target_min":
                            wcfh_rejected_target_min_count += 1
                if standard_a3_triggered or (
                    variant_config["uses_wcfh"] and wcfh_accepted
                ):
                    flag_force_ttt_to_zero = False  # 是否强制将TTT置零标记
                    if standard_a3_triggered:
                        debug(
                            f"HOM met, updating ttt_countdown from {user.ttt_countdown}"
                        )
                    elif wcfh_accepted:
                        user.ttt_before = user.ttt_determined
                        user.hom_before = user.hom_determined
                        user.ttt_countdown = 0  # 强制将TTT计数器置零
                        flag_force_ttt_to_zero = True
                        debug(f"Triggered forced setting ttt_countdown to 0")

                    user.dl_source_bs_rsrp_list.append(user.source_bs_rsrp)
                    if user.ttt_countdown > 0:
                        user.ttt_countdown -= 100  # 每个时间步为100ms
                        if user.ttt_countdown == 0:
                            just_reached_zero_flag = True

                    else:
                        debug(f"ttt_countdown is 0, triggering {variant} policy")
                        flag_dqn_triggered = True  # 触发控制策略
                        dummy_dl_source_bs_rsrp_list = deepcopy(
                            user.dl_source_bs_rsrp_list
                        )
                        user.avg_rsrp_in_dbm = average_rsrp_calculation(
                            user.dl_source_bs_rsrp_list
                        )

                        objective_label_before_handover = objective_handover_label(
                            dummy_dl_source_bs_rsrp_list,
                            user.source_bs_rsrp,
                        )
                        if flag_force_ttt_to_zero and variant_config["uses_wcfh"]:
                            dl_predicted_label_before_handover = 1
                            label_source = "wcfh_late_cue"
                        elif variant_config["uses_catboost_label"]:
                            dl_predicted_label_before_handover = predict_catboost_label(
                                user, user_idx, user_step
                            )
                            label_source = "catboost"
                        else:
                            dl_predicted_label_before_handover = NEUTRAL_LABEL
                            label_source = "masked_neutral"
                        dl_predicted = np.array([[dl_predicted_label_before_handover]])

                        # 构建当前状态.仅当TTT为0时,才需要调用DQN模型
                        curr_state.clear()
                        curr_state += BS_LOCATION_LIST
                        curr_state += USER_TRAJECTORY_LIST[user_idx][user_step]
                        curr_state += [
                            user.source_bs_id,
                            user.source_bs_distance,
                            user.source_bs_sinr,
                            user.source_bs_rsrp,
                            user.target_bs_id,
                            user.target_bs_distance,
                            user.target_bs_sinr,
                            user.target_bs_rsrp,
                            INIT_TTT,
                            INIT_HOM,
                            dl_predicted[0][0],
                        ]
                        debug(f"curr_state: {curr_state}")

                        action_prior_metadata = {
                            "pre_prior_action": ("", ""),
                            "post_prior_action": ("", ""),
                            "action_prior_applied": 0,
                            "action_prior_beta": 0.0,
                        }
                        ppo_metadata = None
                        if variant_config["uses_dqn"] and variant_config.get("uses_ppo", False):
                            action, ppo_metadata = choose_ppo_action(
                                dqn,
                                curr_state,
                                RL_ACTION_SPACE,
                                device,
                                action_prior_label=(
                                    dl_predicted_label_before_handover
                                    if variant_config.get("uses_action_prior", False)
                                    else None
                                ),
                                action_prior_beta=(
                                    ACTION_PRIOR_BETA
                                    if variant_config.get("uses_action_prior", False)
                                    else 0.0
                                ),
                                reference_action=(user.ttt_determined, user.hom_determined),
                            )
                            q_estimate = None
                            action_prior_metadata = ppo_metadata
                        elif variant_config["uses_dqn"]:
                            action, q_estimate, action_prior_metadata = choose_action(
                                dqn,
                                curr_state,
                                epsilon,  # Phase 7W fixed epsilon-greedy exploration rate.
                                RL_ACTION_SPACE,
                                device,
                                action_prior_label=(
                                    dl_predicted_label_before_handover
                                    if variant_config.get("uses_action_prior", False)
                                    else None
                                ),
                                action_prior_beta=(
                                    ACTION_PRIOR_BETA
                                    if variant_config.get("uses_action_prior", False)
                                    else 0.0
                                ),
                                reference_action=(user.ttt_determined, user.hom_determined),
                            )
                        elif variant == "adaptive_heuristic":
                            action = adaptive_heuristic_action(user)
                            q_estimate = None
                        else:
                            action = (user.ttt_determined, user.hom_determined)
                            q_estimate = None
                        if variant_config["uses_dqn"]:
                            action_index_for_update = (
                                int(ppo_metadata["action_index"])
                                if ppo_metadata is not None and "action_index" in ppo_metadata
                                else ACTION_INDEX_BY_VALUE[tuple(action)]
                            )
                        else:
                            action_index_for_update = -1
                        debug(f"action: {action}")
                        debug(f"q_estimate: {q_estimate}")

                        dqn_log_list.append(
                            [
                                variant,
                                label_source,
                                objective_label_before_handover,
                                USER_TRAJECTORY_LIST[user_idx][user_step][0],  # time
                                user.curr_x,
                                user.curr_y,
                                user.source_bs_id,
                                user.source_bs_distance,
                                user.source_bs_sinr,
                                user.source_bs_rsrp,
                                user.target_bs_id,
                                user.target_bs_distance,
                                user.target_bs_sinr,
                                user.target_bs_rsrp,
                                user.ttt_determined,
                                user.hom_determined,
                                dl_predicted[0][0],
                                action[0],
                                action[1],
                                action_prior_metadata["pre_prior_action"][0],
                                action_prior_metadata["pre_prior_action"][1],
                                action_prior_metadata["post_prior_action"][0],
                                action_prior_metadata["post_prior_action"][1],
                                action_prior_metadata["action_prior_applied"],
                                action_prior_metadata["action_prior_beta"],
                                0,  # 先占位 reward，稍后替换
                                "",
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                int(replay_buffer is not None),
                                len(replay_buffer) if replay_buffer is not None else 0,
                                0,
                                dqn_update_count,
                                "",
                                "",
                                VALUE_TD_LOSS if value_based_variant else "",
                                TARGET_UPDATE_MODE if value_based_variant else "",
                            ]
                        )

                        # 在更新动作之前，保存当前值（用于 Handover_Pre）
                        user.ttt_before = user.ttt_determined
                        user.hom_before = user.hom_determined
                        previous_action = (user.ttt_before, user.hom_before)

                        # 此处需根据动作确定是否进行切换
                        user.ttt_determined, user.hom_determined = action
                        debug(
                            f"ttt_determined: {user.ttt_determined}, hom_determined: {user.hom_determined}"
                        )

                        # 判断A3条件是否满足
                        a3_satisfied = (
                            user.target_bs_rsrp - user.source_bs_rsrp
                        ) > user.hom_determined

                        # 得到新的HOM后,如果RSRP差异不满足条件,则更新TTT,不进行切换
                        if (
                            user.target_bs_rsrp - user.source_bs_rsrp
                            <= user.hom_determined
                        ):
                            user.ttt_countdown = user.ttt_determined
                        # 如果RSRP差异满足条件,即差异较大,则更新TTT,如果TTT为0,则进行切换
                        else:
                            user.ttt_countdown = max(
                                0, user.ttt_determined - user.ttt_backup
                            )

                            if user.ttt_countdown == 0:
                                user.dl_source_bs_rsrp_list.clear()  # 当发生切换时,清空最近RSRP列表

                        user.ttt_backup = (
                            user.ttt_determined
                        )  # 对本轮确定的新TTT进行备份
                elif user.target_bs_rsrp - user.source_bs_rsrp <= user.hom_determined:
                    # debug(
                    #     f"HOM not met, resetting ttt_countdown from {user.ttt_countdown} to {user.ttt_determined}"
                    # )
                    user.ttt_countdown = (
                        user.ttt_determined
                    )  # 当不满足HOM条件时,重置TTT
                    user.dl_source_bs_rsrp_list.clear()

                # # 满足强制切换条件
                # if user.source_bs_rsrp < FORCED_HANDOVER_RSRP_THRESHOLD:
                #     if (
                #         user.target_bs_rsrp > user.source_bs_rsrp
                #         and user.target_bs_rsrp - user.source_bs_rsrp
                #         <= user.hom_determined
                #     ):
                #         user.ttt_before = user.ttt_determined
                #         user.hom_before = user.hom_determined
                #         user.ttt_countdown = 0  # 只清零 countdown，不改参数
                #         debug("Triggered forced handover condition.")

                # if user_step % 10000 == 0:
                #     debug_print(
                #         f"ttt_countdown: {user.ttt_countdown}, hom_determined: {user.hom_determined}"
                #     )

                # 用户前进一步
                user.get_next_location(user_step)

                # 判断是否真的切换（当前帧切换完成）
                handover_executed = (user.ttt_countdown == 0) and (
                    not just_reached_zero_flag
                )
                pre_handover_time = USER_TRAJECTORY_LIST[user_idx][user_step][0]
                pre_handover_source_bs_id = user.source_bs_id
                pre_handover_source_rsrp = user.source_bs_rsrp
                handover_rsrp_gain = None
                ppho_event = False

                # 记录SINR用于吞吐量计算
                segment_sinr_sum += user.source_bs_sinr
                segment_step_count += 1

                if (user_step + 1) % segment_step == 0:
                    avg_sinr = segment_sinr_sum / segment_step_count
                    seg_thpt = estimate_throughput(avg_sinr) / 1e6  # Mbps
                    segment_list.append(round(seg_thpt, 3))
                    segment_sinr_sum = 0.0
                    segment_step_count = 0

                # 即未触发切换 新增just_reached_zero_flag逻辑，避免当计数器为100时会递减为0在此处引起错误切换
                if user.ttt_countdown > 0 or just_reached_zero_flag:
                    user.next_step_calculation_no_handover(bs_list)
                else:  # 即触发了切换

                    # 切换前状态记录
                    ho_log_list.append(
                        [
                            "Handover_Pre",
                            USER_TRAJECTORY_LIST[user_idx][user_step][0],
                            user.curr_x,
                            user.curr_y,
                            user.source_bs_id,
                            user.source_bs_distance,
                            user.source_bs_sinr,
                            user.source_bs_rsrp,
                            user.target_bs_id,
                            user.target_bs_distance,
                            user.target_bs_sinr,
                            user.target_bs_rsrp,
                            user.ttt_before,
                            user.hom_before,
                        ]
                    )

                    # 更新后
                    user.curr_x = user.next_x
                    user.curr_y = user.next_y
                    user.next_step_calculation_with_handover(bs_list)
                    handover_rsrp_gain = user.source_bs_rsrp - pre_handover_source_rsrp
                    post_handover_time = USER_TRAJECTORY_LIST[user_idx][user_step + 1][0]
                    if (
                        last_departed_bs_id is not None
                        and user.source_bs_id == last_departed_bs_id
                        and last_handover_time is not None
                        and (post_handover_time - last_handover_time) <= ppho_window_ms()
                    ):
                        ppho_event = True
                    last_departed_bs_id = pre_handover_source_bs_id
                    last_handover_time = post_handover_time

                    ho_log_list.append(
                        [
                            "Handover_Post",
                            USER_TRAJECTORY_LIST[user_idx][user_step + 1][0],
                            user.curr_x,
                            user.curr_y,
                            user.source_bs_id,
                            user.source_bs_distance,
                            user.source_bs_sinr,
                            user.source_bs_rsrp,
                            user.target_bs_id,
                            user.target_bs_distance,
                            user.target_bs_sinr,
                            user.target_bs_rsrp,
                            user.ttt_determined,
                            user.hom_determined,
                        ]
                    )

                # 如果没有触发DQN,则直接进入下一步
                if not flag_dqn_triggered:
                    user.curr_x = user.next_x
                    user.curr_y = user.next_y
                # 如果触发了DQN,则计算目标Q值并进行DQN优化
                else:
                    dummy_dl_source_bs_rsrp_list.append(user.source_bs_rsrp)
                    user.avg_rsrp_in_dbm = average_rsrp_calculation(
                        dummy_dl_source_bs_rsrp_list
                    )

                    prev_rsrp_for_reward = (
                        dummy_dl_source_bs_rsrp_list[-2]
                        if len(dummy_dl_source_bs_rsrp_list) >= 2
                        else pre_handover_source_rsrp
                    )
                    if variant_config["uses_dqn"]:
                        if variant_config["uses_catboost_label"]:
                            dl_predicted_after = predict_catboost_label(
                                user, user_idx, user_step + 1
                            )
                        else:
                            dl_predicted_after = NEUTRAL_LABEL

                        next_state.clear()
                        next_state += BS_LOCATION_LIST
                        next_state += USER_TRAJECTORY_LIST[user_idx][user_step + 1]
                        next_state += [
                            user.source_bs_id,
                            user.source_bs_distance,
                            user.source_bs_sinr,
                            user.source_bs_rsrp,
                            user.target_bs_id,
                            user.target_bs_distance,
                            user.target_bs_sinr,
                            user.target_bs_rsrp,
                            user.ttt_determined,
                            user.hom_determined,
                            dl_predicted_after,
                        ]
                        debug(f"next_state: {next_state}")

                        reward_family = variant_config.get("reward_family", REWARD_FAMILY_RSRP)
                        if reward_family == REWARD_FAMILY_LEGACY_LABEL_SHAPED:
                            reward = calculate_reward(
                                prev_rsrp=prev_rsrp_for_reward,
                                post_rsrp=user.source_bs_rsrp,
                                predicted_label=dl_predicted_label_before_handover,
                                previous_action=previous_action,
                                action=(user.ttt_determined, user.hom_determined),
                                a3_satisfied=a3_satisfied,
                                handover_executed=handover_executed,
                                flag_reward_rsrp=FLAG_REWARD_RSRP,
                            )
                        elif reward_family == REWARD_FAMILY_OUTCOME:
                            reward_components = outcome_based_reward(
                                prev_rsrp=prev_rsrp_for_reward,
                                post_rsrp=user.source_bs_rsrp,
                                handover_executed=handover_executed,
                                ppho_event=ppho_event,
                                post_sinr=user.source_bs_sinr,
                                rlf_sinr_threshold=RLF_SINR_THRESHOLD_DB,
                                return_components=True,
                            )
                            reward = reward_components["reward"]
                        else:
                            reward = rsrp_control_reward(
                                prev_rsrp_for_reward,
                                user.source_bs_rsrp,
                                handover_executed,
                            )

                        dqn_log_list[-1][DQN_LOG_REWARD_INDEX] = reward
                        if reward_family == REWARD_FAMILY_OUTCOME:
                            dqn_log_list[-1][
                                DQN_LOG_REWARD_COMPONENT_START_INDEX:DQN_LOG_REWARD_COMPONENT_START_INDEX + 10
                            ] = [
                                reward_components["profile"],
                                reward_components["rsrp"],
                                reward_components["sinr"],
                                reward_components["execution"],
                                reward_components["ppho"],
                                reward_components["raw"],
                                reward_components["handover_executed"],
                                reward_components["ppho_event"],
                                reward_components["rsrp_gain_db"],
                                reward,
                            ]

                        curr_state_tensor = torch.tensor(
                            curr_state, dtype=torch.float32
                        ).to(device)
                        next_state_tensor = torch.tensor(
                            next_state, dtype=torch.float32
                        ).to(device)
                        replay_warmup_active = 0
                        replay_batch_loss = ""
                        replay_batch_reward_mean = ""
                        if variant_config.get("uses_ppo", False):
                            with torch.no_grad():
                                _, next_value = dqn(next_state_tensor)
                                value_target = torch.tensor(
                                    reward, dtype=torch.float32, device=device
                                ) + RL_GAMMA * next_value.detach()

                            logits, current_value = dqn(curr_state_tensor)
                            if variant_config.get("uses_action_prior", False):
                                from ActionChooser import action_prior_bias

                                prior_bias = torch.tensor(
                                    action_prior_bias(
                                        RL_ACTION_SPACE,
                                        previous_action,
                                        dl_predicted_label_before_handover,
                                    ),
                                    dtype=torch.float32,
                                    device=device,
                                )
                                logits = logits + ACTION_PRIOR_BETA * prior_bias
                            distribution = torch.distributions.Categorical(logits=logits)
                            action_index = torch.tensor(
                                ppo_metadata["action_index"], dtype=torch.long, device=device
                            )
                            new_log_prob = distribution.log_prob(action_index)
                            old_log_prob = ppo_metadata["log_prob"].detach()
                            advantage = (value_target - current_value).detach()
                            ratio = torch.exp(new_log_prob - old_log_prob)
                            clipped_ratio = torch.clamp(
                                ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS
                            )
                            policy_loss = -torch.min(
                                ratio * advantage, clipped_ratio * advantage
                            )
                            value_loss = torch.nn.functional.mse_loss(
                                current_value, value_target.detach()
                            )
                            entropy_loss = -distribution.entropy()
                            dqn_loss = (
                                policy_loss
                                + PPO_VALUE_COEF * value_loss
                                + PPO_ENTROPY_COEF * entropy_loss
                            )
                            rl_optimizer.zero_grad()
                            dqn_loss.backward()
                            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
                            rl_optimizer.step()
                            dqn_update_count += 1
                        else:
                            if replay_buffer is not None:
                                replay_buffer.append(
                                    ReplayTransition(
                                        state=[float(value) for value in curr_state],
                                        action_index=int(action_index_for_update),
                                        reward=float(reward),
                                        next_state=[float(value) for value in next_state],
                                    )
                                )
                                replay_warmup_active = int(
                                    epoch < REPLAY_WARMUP_EPOCHS
                                    or len(replay_buffer) < REPLAY_WARMUP_TRANSITIONS
                                )
                                if (
                                    not replay_warmup_active
                                    and len(replay_buffer) >= REPLAY_BATCH_SIZE
                                    and REPLAY_UPDATES_PER_STEP > 0
                                ):
                                    loss_values = []
                                    last_batch_rewards = []
                                    for _ in range(REPLAY_UPDATES_PER_STEP):
                                        batch = replay_buffer.sample(REPLAY_BATCH_SIZE)
                                        dqn_loss_tensor = optimize_value_batch(
                                            dqn=dqn,
                                            target_dqn=target_dqn,
                                            optimizer=rl_optimizer,
                                            transitions=batch,
                                            uses_double_dqn=variant_config.get(
                                                "uses_double_dqn", False
                                            ),
                                            loss_fn=value_loss_fn,
                                            device=device,
                                        )
                                        loss_values.append(float(dqn_loss_tensor.item()))
                                        last_batch_rewards = [
                                            float(transition.reward) for transition in batch
                                        ]
                                        dqn_update_count += 1
                                        if dqn_update_count % DDQN_TARGET_UPDATE_STEPS == 0:
                                            target_dqn.load_state_dict(dqn.state_dict())
                                    dqn_loss = float(np.mean(loss_values))
                                    replay_batch_loss = dqn_loss
                                    replay_batch_reward_mean = float(np.mean(last_batch_rewards))
                                else:
                                    dqn_loss = 0.0
                            elif variant_config.get("uses_double_dqn", False):
                                next_action_idx = torch.argmax(dqn(next_state_tensor)).item()
                                next_q = target_dqn(next_state_tensor)[next_action_idx].detach()
                                q_target = reward + RL_GAMMA * next_q
                                q_target = q_target.unsqueeze(0)
                                dqn_loss = value_loss_fn(q_estimate, q_target)
                                rl_optimizer.zero_grad()
                                dqn_loss.backward()
                                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
                                rl_optimizer.step()
                                dqn_update_count += 1
                                if dqn_update_count % DDQN_TARGET_UPDATE_STEPS == 0:
                                    target_dqn.load_state_dict(dqn.state_dict())
                            else:
                                next_q = torch.max(target_dqn(next_state_tensor).detach())
                                q_target = reward + RL_GAMMA * next_q
                                q_target = q_target.unsqueeze(0)
                                dqn_loss = value_loss_fn(q_estimate, q_target)
                                rl_optimizer.zero_grad()
                                dqn_loss.backward()
                                torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
                                rl_optimizer.step()
                                dqn_update_count += 1
                                if dqn_update_count % DDQN_TARGET_UPDATE_STEPS == 0:
                                    target_dqn.load_state_dict(dqn.state_dict())
                        dqn_log_list[-1][DQN_LOG_REPLAY_START_INDEX:DQN_LOG_REPLAY_START_INDEX + 8] = [
                            int(replay_buffer is not None),
                            len(replay_buffer) if replay_buffer is not None else 0,
                            replay_warmup_active,
                            dqn_update_count,
                            replay_batch_loss,
                            replay_batch_reward_mean,
                            VALUE_TD_LOSS if value_based_variant else "",
                            TARGET_UPDATE_MODE if value_based_variant else "",
                        ]
                    else:
                        reward = rsrp_control_reward(
                            prev_rsrp_for_reward,
                            user.source_bs_rsrp,
                            handover_executed,
                        )
                        dqn_log_list[-1][DQN_LOG_REWARD_INDEX] = reward

                    # ==== 客观切换类型与失败统计 ====
                    total_ho_attempts += 1

                    if objective_label_before_handover == 0:
                        early_ho_count += 1
                    elif objective_label_before_handover == 1:
                        late_ho_count += 1
                    else:
                        ideal_ho_count += 1
                    hof_count = early_ho_count + late_ho_count

                    # ==== RLF 判定逻辑：只要切换后 SINR 仍低于门限则视为链路失败 ====
                    if handover_executed:
                        completed_ho_count += 1
                        if handover_rsrp_gain is not None:
                            rsrp_gain_sum += handover_rsrp_gain
                        if ppho_event:
                            ppho_count += 1
                        if user.source_bs_sinr < RLF_SINR_THRESHOLD_DB:
                            rlf_count += 1

                if radio_link_trace_enabled(epoch):
                    append_radio_link_trace(
                        radio_link_trace_list,
                        epoch=epoch,
                        ue_id=ue_id,
                        variant=variant,
                        sample_time=USER_TRAJECTORY_LIST[user_idx][user_step + 1][0],
                        user_step=user_step + 1,
                        pos_x=user.next_x,
                        pos_y=user.next_y,
                        user=user,
                        standard_a3_triggered=standard_a3_triggered,
                        raw_wcfh_candidate=raw_wcfh_candidate,
                        wcfh_accepted=wcfh_accepted,
                        wcfh_reject_reason=wcfh_reject_reason,
                        handover_executed=handover_executed,
                    )

                user_step += 1  # 更新计步器

                # # 仅用于校验DQN调用时是否存在问题
                # if flag_dqn_triggered:
                #     break

            user_segment_throughput[ue_id] = segment_list

            # ==== 日志写入 ====
            if dqn_log_list:
                # 插入 Epoch 和 UserId
                for entry in dqn_log_list:
                    entry.insert(0, epoch)
                    entry.insert(1, ue_id)

                dqn_log_path = os.path.join(log_base_dir, "DQN_log_all.csv")
                write_header = not os.path.exists(dqn_log_path)

                with open(dqn_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(
                            [
                                "Epoch",
                                "UserId",
                                "Variant",
                                "LabelSource",
                                "ObjectiveLabel",
                                "Time",
                                "PosX",
                                "PosY",
                                "SourceBsId",
                                "SourceDis",
                                "SourceSinr",
                                "SourceRsrp",
                                "TargetBsId",
                                "TargetDis",
                                "TargetSinr",
                                "TargetRsrp",
                                "TTT",
                                "HOM",
                                "dl_predicted",
                                "Action_TTT",
                                "Action_HOM",
                                "PrePriorAction_TTT",
                                "PrePriorAction_HOM",
                                "PostPriorAction_TTT",
                                "PostPriorAction_HOM",
                                "PriorShiftApplied",
                                "ActionPriorBeta",
                                "Reward",
                                "RewardProfile",
                                "RewardRsrpComponent",
                                "RewardSinrComponent",
                                "RewardExecComponent",
                                "RewardPphoComponent",
                                "RewardRaw",
                                "HandoverExecuted",
                                "PPHOEvent",
                                "RewardRsrpGainDb",
                                "RewardFinal",
                                "ReplayEnabled",
                                "ReplayBufferSize",
                                "ReplayWarmupActive",
                                "ReplayUpdateCount",
                                "ReplayBatchLoss",
                                "ReplayBatchRewardMean",
                                "ValueTdLoss",
                                "TargetUpdateMode",
                            ]
                        )
                    writer.writerows(dqn_log_list)

            if ho_log_list:
                # 插入 Epoch 和 UserId
                for entry in ho_log_list:
                    entry.insert(0, epoch)
                    entry.insert(1, ue_id)

                ho_log_path = os.path.join(log_base_dir, "HO_log_all.csv")
                write_header = not os.path.exists(ho_log_path)

                with open(ho_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(
                            [
                                "Epoch",
                                "UserId",
                                "EventType",
                                "Time",
                                "PosX",
                                "PosY",
                                "SourceBsId",
                                "SourceDis",
                                "SourceSinr",
                                "SourceRsrp",
                                "TargetBsId",
                                "TargetDis",
                                "TargetSinr",
                                "TargetRsrp",
                                "TTT",
                                "HOM",
                            ]
                        )
                    writer.writerows(ho_log_list)

            if radio_link_trace_list:
                trace_log_path = os.path.join(log_base_dir, "radio_link_trace.csv")
                write_header = not os.path.exists(trace_log_path)

                with open(trace_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(RADIO_LINK_TRACE_HEADER)
                    writer.writerows(radio_link_trace_list)

        # 在每次触发DQN后，记录当前epoch内的所有loss
        if isinstance(dqn_loss, torch.Tensor):
            epoch_dqn_loss_list.append(dqn_loss.item())
        elif dqn_loss != 0.0:
            epoch_dqn_loss_list.append(dqn_loss)

        # 所有用户轨迹完成后：
        if epoch_dqn_loss_list:
            list_dqn_loss.append(
                np.mean(epoch_dqn_loss_list)
            )  # 计算当前epoch内所有用户轨迹的平均损失

            # 保存所有 loss 到 CSV（每行一条）20250519修改
            dqn_loss_path = os.path.join(log_base_dir, "DQN_loss.csv")
            np.savetxt(
                dqn_loss_path,
                np.array(list_dqn_loss),
                delimiter=",",
                header="AverageDQN_Loss",
                comments="",
            )

        # ==== 保存综合统计 ====
        stats_log_path = os.path.join(log_base_dir, "RL_HO_stats.csv")
        write_header = not os.path.exists(stats_log_path)

        all_segments = [seg for lst in user_segment_throughput.values() for seg in lst]
        avg_throughput = np.mean(all_segments) if all_segments else 0
        avg_rsrp_gain = (
            round(rsrp_gain_sum / completed_ho_count, 4)
            if completed_ho_count
            else 0
        )

        with open(stats_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Epoch",
                        "Variant",
                        "RandomSeed",
                        "Avg_Throughput_Mbps",
                        "TotalHOs",
                        "CompletedHOs",
                        "EarlyHO",
                        "LateHO",
                        "IdealHO",
                        "HOF",
                        "PPHO",
                        "RLF",
                        "Avg_RSRP_Gain_dB",
                        "WCFHRawCandidates",
                        "WCFHAccepted",
                        "WCFHRejectedByMargin",
                        "WCFHRejectedByTargetMin",
                        "EarlyHO_Rate",
                        "LateHO_Rate",
                        "HOF_Rate",
                        "PPHO_Rate",
                        "RLF_Rate",
                    ]
                )
            writer.writerow(
                [
                    epoch,
                    variant,
                    random_seed,
                    round(avg_throughput, 3),
                    total_ho_attempts,
                    completed_ho_count,
                    early_ho_count,
                    late_ho_count,
                    ideal_ho_count,
                    hof_count,
                    ppho_count,
                    rlf_count,
                    avg_rsrp_gain,
                    wcfh_raw_candidate_count,
                    wcfh_accepted_count,
                    wcfh_rejected_margin_count,
                    wcfh_rejected_target_min_count,
                    round(early_ho_count / total_ho_attempts, 4)
                    if total_ho_attempts
                    else 0,
                    round(late_ho_count / total_ho_attempts, 4)
                    if total_ho_attempts
                    else 0,
                    round(hof_count / total_ho_attempts, 4)
                    if total_ho_attempts
                    else 0,
                    round(ppho_count / completed_ho_count, 4)
                    if completed_ho_count
                    else 0,
                    round(rlf_count / completed_ho_count, 4)
                    if completed_ho_count
                    else 0,
                ]
            )

        # ==== 保存分段吞吐量 ====
        segment_log_path = os.path.join(log_base_dir, "Throughput_segments.csv")
        write_seg_header = not os.path.exists(segment_log_path)

        with open(segment_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_seg_header:
                max_seg = (
                    max(len(v) for v in user_segment_throughput.values())
                    if user_segment_throughput
                    else 0
                )
                writer.writerow(
                    ["Epoch", "UserId"] + [f"Seg{i}_Thpt" for i in range(max_seg)]
                )
            for uid, segs in user_segment_throughput.items():
                row = [epoch, uid] + segs
                writer.writerow(row)

        if (not DISABLE_LOSS_EARLY_STOP) and len(list_dqn_loss) > 10:
            avg_recent_loss = np.mean(list_dqn_loss[-10:])
            debug_print(f"Recent DQN Loss avg = {avg_recent_loss:.4f}")
            if avg_recent_loss < 0.01:
                debug_print(f"Converged at epoch {epoch}, loss = {avg_recent_loss:.4f}")
                break

        # 强制终止条件
        if epoch >= RL_MAX_EPOCHS:
            debug_print(f"Forced stop at epoch {epoch}")
            break

        # Phase 7W keeps epsilon fixed at RL_EPSILON for value-based variants.

        epoch += 1  # 更新主循环计数器


if __name__ == "__main__":
    debug_print(f"device: {device}")
    debug_print(f"dl_model: {dl_model}")

    # 强化学习主函数
    rl()
