from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayTransition:
    state: list[float]
    action_index: int
    reward: float
    next_state: list[float]


class ExperienceReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self._items: deque[ReplayTransition] = deque(maxlen=capacity)

    def append(self, transition: ReplayTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError("Replay batch size must be positive.")
        if len(self._items) < batch_size:
            raise ValueError("Replay buffer does not contain enough transitions.")
        return random.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)
