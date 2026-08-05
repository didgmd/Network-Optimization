# -*- coding: utf-8 -*-
import numpy as np
import torch


def action_prior_bias(action_space, reference_action, predicted_label):
    if predicted_label is None or reference_action is None:
        return np.zeros(len(action_space), dtype=np.float32)

    ref_ttt, ref_hom = reference_action
    scores = []
    for ttt, hom in action_space:
        ttt_delta = np.sign(ttt - ref_ttt)
        hom_delta = np.sign(hom - ref_hom)
        if int(predicted_label) == 0:
            score = 0.5 * (ttt_delta + hom_delta)
        elif int(predicted_label) == 1:
            score = -0.5 * (ttt_delta + hom_delta)
        elif int(predicted_label) == 2:
            same_action = ttt_delta == 0 and hom_delta == 0
            small_move = abs(ttt - ref_ttt) <= 500 and abs(hom - ref_hom) <= 0.5
            score = 1.0 if same_action else 0.25 if small_move else -0.5
        else:
            score = 0.0
        scores.append(float(np.clip(score, -1.0, 1.0)))
    return np.asarray(scores, dtype=np.float32)


def choose_action(
    dqn,
    curr_state,
    epsilon,
    action_space,
    device,
    action_prior_label=None,
    action_prior_beta=0.0,
    reference_action=None,
):
    raw_action_tensor = dqn(torch.tensor(curr_state, dtype=torch.float32).to(device))
    action_tensor = raw_action_tensor.clone()
    pre_prior_action_index = torch.argmax(raw_action_tensor).item()
    action_prior_applied = False

    if action_prior_label is not None and action_prior_beta > 0:
        bias = torch.tensor(
            action_prior_bias(action_space, reference_action, action_prior_label),
            dtype=torch.float32,
            device=device,
        )
        action_tensor = action_tensor + float(action_prior_beta) * bias
        action_prior_applied = True

    if np.random.uniform() < epsilon:
        action_index = np.random.choice(range(len(action_space)))  # 随机探索
    else:
        action_index = torch.argmax(action_tensor).item()  # 贪婪利用

    action = action_space[action_index]
    q_estimate = raw_action_tensor[action_index].unsqueeze(
        0
    )  # 增加一个维度，保证q_estimate是标量tensor; prior only biases selection.

    metadata = {
        "pre_prior_action": action_space[pre_prior_action_index],
        "post_prior_action": action,
        "action_prior_applied": int(action_prior_applied),
        "action_prior_beta": float(action_prior_beta),
    }

    return action, q_estimate, metadata


def choose_ppo_action(
    actor_critic,
    curr_state,
    action_space,
    device,
    action_prior_label=None,
    action_prior_beta=0.0,
    reference_action=None,
):
    state_tensor = torch.tensor(curr_state, dtype=torch.float32).to(device)
    logits, value = actor_critic(state_tensor)
    pre_prior_action_index = torch.argmax(logits).item()
    action_prior_applied = False

    if action_prior_label is not None and action_prior_beta > 0:
        bias = torch.tensor(
            action_prior_bias(action_space, reference_action, action_prior_label),
            dtype=torch.float32,
            device=device,
        )
        logits = logits + float(action_prior_beta) * bias
        action_prior_applied = True

    distribution = torch.distributions.Categorical(logits=logits)
    action_index = distribution.sample()
    action = action_space[action_index.item()]
    metadata = {
        "pre_prior_action": action_space[pre_prior_action_index],
        "post_prior_action": action,
        "action_prior_applied": int(action_prior_applied),
        "action_prior_beta": float(action_prior_beta),
        "log_prob": distribution.log_prob(action_index),
        "entropy": distribution.entropy(),
        "value": value,
        "action_index": action_index.item(),
    }
    return action, metadata
