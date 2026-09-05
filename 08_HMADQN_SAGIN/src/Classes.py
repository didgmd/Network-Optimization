# -*- coding: utf-8 -*-
import hashlib
import math

import numpy as np
import torch
from Parameters import (
    SAT_ORBIT_HEIGHT_M,
    SAT_FREQUENCY,
    SAT_TX_POWER,
    AREA_SIZE_X,
    AREA_SIZE_Y,
    USER_STEP_SPEED_X,
    USER_STEP_SPEED_Y,
    USER_HEIGHT,
    UAV_MIN_HEIGHT,
    UAV_MAX_HEIGHT,
    UAV_STEP_SPEED,
    DELTA_MACRO_dB,
    DELTA_SMALL_dB,
    DELTA_UAV_dB,
    DELTA_SAT_dB,
)


class BaseStation:
    def __init__(self, index, bs_type):
        self.index = index
        self.bs_type = bs_type
        self.serv_user_list = []


class SatelliteBS(BaseStation):
    def __init__(self, index):
        super().__init__(index, "satellite")
        self.orbit_height_m = SAT_ORBIT_HEIGHT_M
        self.tx_power = SAT_TX_POWER
        self.frequency = SAT_FREQUENCY
        self.x = AREA_SIZE_X / 2
        self.y = AREA_SIZE_Y / 2

    def calculate_distance(self, user, curr_or_next="curr"):
        user_x = user.x if curr_or_next == "curr" else user.next_x
        user_y = user.y if curr_or_next == "curr" else user.next_y
        user_z = user.z if curr_or_next == "curr" else user.next_z
        dx = self.x - user_x
        dy = self.y - user_y
        dz = self.orbit_height_m - user_z
        return (dx**2 + dy**2 + dz**2) ** 0.5

    def __repr__(self):
        return f"SatelliteBS(index={self.index}, height_m={self.orbit_height_m})"


class MacroBS(BaseStation):
    def __init__(self, index, x, y, z=0.0):
        super().__init__(index, "macro")
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"MacroBS(index={self.index}, x={self.x:.0f}, y={self.y:.0f})"


class SmallBS(BaseStation):
    def __init__(self, index, x, y, z=0.0):
        super().__init__(index, "small")
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"SmallBS(index={self.index}, x={self.x:.0f}, y={self.y:.0f})"


class User:
    def __init__(self, global_index, x, y, user_type):
        self.global_index = global_index
        self.x = x
        self.y = y
        self.z = USER_HEIGHT
        self.user_type = user_type
        self.rsrp_list = []
        self.serv_bs_id_by_cloud_agent = None
        self.id_in_serv_bs = None
        self.previous_bs = None
        self.next_x = x
        self.next_y = y
        self.next_z = USER_HEIGHT
        self.shadowing_sigma_db = 0.0
        self.shadowing_seed = 0

        if user_type == "eMBB":
            self.qos = {"rate": 25}
        elif user_type == "uRLLC":
            self.qos = {"rate": 10}
        elif user_type == "mMTC":
            self.qos = {"rate": 1}
        else:
            self.qos = {"rate": 10}

    def __repr__(self):
        return (
            f"User(global_index={self.global_index}, x={self.x:.0f}, y={self.y:.0f}, "
            f"user_type={self.user_type})"
        )

    def calculate_rsrp(self, curr_or_next, bs, frequency, tx_power):
        if isinstance(bs, SatelliteBS):
            distance = bs.calculate_distance(self, curr_or_next)
            delta = DELTA_SAT_dB
        else:
            user_x = self.x if curr_or_next == "curr" else self.next_x
            user_y = self.y if curr_or_next == "curr" else self.next_y
            user_z = self.z if curr_or_next == "curr" else self.next_z

            if hasattr(bs, "curr_z"):
                dz = bs.curr_z - user_z
                distance = ((user_x - bs.x) ** 2 + (user_y - bs.y) ** 2 + dz**2) ** 0.5
                delta = DELTA_UAV_dB
            else:
                bs_z = getattr(bs, "z", 0.0)
                dz = bs_z - user_z
                distance = ((user_x - bs.x) ** 2 + (user_y - bs.y) ** 2 + dz**2) ** 0.5
                delta = DELTA_MACRO_dB if bs.bs_type == "macro" else DELTA_SMALL_dB

        if distance <= 0:
            distance = 1.0

        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        pl = fspl + delta
        rsrp = tx_power - pl
        return rsrp + self._shadowing_db(curr_or_next, bs)

    def configure_shadowing(self, sigma_db=0.0, seed=0):
        self.shadowing_sigma_db = max(float(sigma_db or 0.0), 0.0)
        self.shadowing_seed = int(seed or 0)

    def _shadowing_db(self, curr_or_next, bs):
        if self.shadowing_sigma_db <= 0.0:
            return 0.0

        user_x = self.x if curr_or_next == "curr" else self.next_x
        user_y = self.y if curr_or_next == "curr" else self.next_y
        user_z = self.z if curr_or_next == "curr" else self.next_z
        bs_x = getattr(bs, "curr_x", getattr(bs, "x", 0.0))
        bs_y = getattr(bs, "curr_y", getattr(bs, "y", 0.0))
        bs_z = getattr(bs, "curr_z", getattr(bs, "z", getattr(bs, "orbit_height_m", 0.0)))
        key = "|".join(
            [
                str(self.shadowing_seed),
                f"{self.shadowing_sigma_db:.6f}",
                str(self.global_index),
                getattr(bs, "bs_type", bs.__class__.__name__),
                str(getattr(bs, "index", "")),
                str(curr_or_next),
                f"{user_x:.3f}",
                f"{user_y:.3f}",
                f"{user_z:.3f}",
                f"{bs_x:.3f}",
                f"{bs_y:.3f}",
                f"{bs_z:.3f}",
            ]
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        u1 = (int.from_bytes(digest[:8], "big") + 0.5) / (2**64)
        u2 = (int.from_bytes(digest[8:16], "big") + 0.5) / (2**64)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return self.shadowing_sigma_db * z

    def calculate_next_step(self):
        self.next_x = self.x + USER_STEP_SPEED_X * np.random.choice([-1, 1])
        self.next_y = self.y + USER_STEP_SPEED_Y * np.random.choice([-1, 1])
        self.next_z = self.z
        self.next_x = max(0, min(self.next_x, AREA_SIZE_X))
        self.next_y = max(0, min(self.next_y, AREA_SIZE_Y))

    def move(self):
        self.x = self.next_x
        self.y = self.next_y


class UavBS(BaseStation):
    def __init__(self, index, curr_x, curr_y, curr_z):
        super().__init__(index, "uav")
        self.curr_x = curr_x
        self.curr_y = curr_y
        self.curr_z = curr_z
        self.x = curr_x
        self.y = curr_y
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_z = curr_z

    def move(self, action_tuple):
        dx, dy, dz = action_tuple
        self.prev_x = self.curr_x
        self.prev_y = self.curr_y
        self.prev_z = self.curr_z
        self.curr_x += dx * UAV_STEP_SPEED
        self.curr_y += dy * UAV_STEP_SPEED
        self.curr_z += dz * UAV_STEP_SPEED
        self.curr_x = np.clip(self.curr_x, 0, AREA_SIZE_X)
        self.curr_y = np.clip(self.curr_y, 0, AREA_SIZE_Y)
        self.curr_z = np.clip(self.curr_z, UAV_MIN_HEIGHT, UAV_MAX_HEIGHT)
        self.x = self.curr_x
        self.y = self.curr_y

    def __repr__(self):
        return f"UavBS(idx={self.index}, x={self.curr_x:.1f}, y={self.curr_y:.1f}, z={self.curr_z:.1f})"


class CloudAgent(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BsAgent(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        input_size = input_dim[0] * input_dim[1]
        self.fc_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc_net(x)


class UavAgent(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
