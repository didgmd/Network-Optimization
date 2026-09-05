# -*- coding: utf-8 -*-
import numpy as np
import torch
from DebugPrint import debug
from Parameters import UAV_ACTION_SPACE

CLOUD_OUTPUT_DIM = 9


def _normalize_cloud_available_mask(available_mask, action_dim):
    if action_dim != CLOUD_OUTPUT_DIM:
        raise RuntimeError(
            f"Cloud action dimension contract failed: expected {CLOUD_OUTPUT_DIM}, "
            f"got {action_dim}."
        )
    if not isinstance(available_mask, (list, tuple, np.ndarray)):
        raise TypeError("available_mask must be a list, tuple, or NumPy array.")
    mask = np.asarray(available_mask)
    if mask.ndim != 1:
        raise ValueError("available_mask must be one-dimensional.")
    if mask.dtype != np.bool_:
        raise TypeError("available_mask entries must be boolean values.")
    if len(mask) != action_dim:
        raise ValueError(
            f"available_mask length {len(mask)} does not match action dimension "
            f"{action_dim}."
        )
    if not bool(mask.any()):
        raise ValueError("available_mask must contain at least one available action.")
    return mask


def cloud_choose_action(
    agent, curr_state, device, epsilon=0.1, available_mask=None
):
    action_tensor = agent(torch.tensor(curr_state, dtype=torch.float32).to(device))
    action_dim = int(action_tensor.shape[0])
    output_dim = getattr(getattr(agent, "fc2", None), "out_features", None)
    if output_dim != CLOUD_OUTPUT_DIM or action_dim != CLOUD_OUTPUT_DIM:
        raise RuntimeError(
            "Cloud action dimension contract failed: agent output layer and runtime "
            f"Q vector must both equal {CLOUD_OUTPUT_DIM}; got {output_dim} and "
            f"{action_dim}."
        )

    if available_mask is None:
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, action_tensor.shape[0])
        else:
            action = torch.argmax(action_tensor).item()
    else:
        mask = _normalize_cloud_available_mask(available_mask, action_dim)
        if np.random.uniform() < epsilon:
            action = int(np.random.choice(np.flatnonzero(mask)))
        else:
            torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=action_tensor.device)
            masked_q = action_tensor.masked_fill(~torch_mask, -torch.inf)
            action = torch.argmax(masked_q).item()
    q_estimate = action_tensor[action]
    debug(f"Cloud action: {action}, Q: {q_estimate:.4f}, epsilon: {epsilon:.4f}")
    return action, q_estimate


def bs_choose_action(
    agent, curr_state, device, epsilon, num_of_sc, active_count, capacity
):
    state_tensor = (
        torch.tensor(curr_state, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    action_tensor = agent(state_tensor).squeeze(0)
    half_len = len(action_tensor) // 2
    sc_scores = action_tensor[:half_len]
    power_scores = action_tensor[half_len:]

    if np.random.uniform() < epsilon:
        sc_logits = torch.rand_like(sc_scores)
    else:
        sc_logits = sc_scores

    sc_weights = torch.softmax(sc_logits, dim=0)
    power_weights = torch.softmax(power_scores, dim=0)
    if active_count > 0:
        active_power = power_weights[:active_count]
        active_sum = torch.sum(active_power)
        if active_sum.item() > 0:
            power_weights = torch.cat(
                [active_power / active_sum, power_weights[active_count:]]
            )
        else:
            uniform = torch.full_like(active_power, 1.0 / active_count)
            power_weights = torch.cat([uniform, power_weights[active_count:]])

    sc_allocations = []
    for u in range(active_count):
        sc_allocations.append(int(np.floor(num_of_sc * sc_weights[u].item())))

    remainder = num_of_sc - sum(sc_allocations)
    if active_count > 0 and remainder != 0:
        max_idx = torch.argmax(sc_weights[:active_count]).item()
        sc_allocations[max_idx] = max(sc_allocations[max_idx] + remainder, 0)

    action_list = []
    for u in range(capacity):
        if u < active_count:
            action_list.append((sc_allocations[u], power_weights[u].item()))
        else:
            action_list.append((0, 0.0))

    probs_sc = torch.softmax(sc_scores, dim=0)
    probs_power = torch.softmax(power_scores, dim=0)
    val_sc = torch.sum(probs_sc * sc_scores)
    val_power = torch.sum(probs_power * power_scores)
    q_estimate = 0.5 * (val_sc + val_power)

    debug(f"BS actions: {action_list}, Q: {q_estimate:.4f}")
    return action_list, q_estimate


def uav_choose_action(agent, curr_state, device, epsilon=0.1):
    action_tensor = agent(torch.tensor(curr_state, dtype=torch.float32).to(device))
    if np.random.uniform() < epsilon:
        action_index = np.random.randint(0, action_tensor.shape[0])
    else:
        action_index = torch.argmax(action_tensor).item()
    action = UAV_ACTION_SPACE[action_index]
    q_estimate = action_tensor[action_index]
    debug(f"UAV action: {action}, Q: {q_estimate:.4f}, epsilon: {epsilon:.4f}")
    return action, q_estimate
