from DebugPrint import *
import numpy as np


class Anchor:
    def __init__(self, index, x, y, radius, num_bs):
        self.index = index
        self.x = x
        self.y = y
        self.radius = radius
        self.hoSuccessRateMatrix = np.full((num_bs, num_bs), 0.0)

    def __repr__(self):
        return f"{self.index}"

    def __str__(self):
        return f"Anchor {self.index} at ({self.x}, {self.y}) with radius {self.radius}"


def anchor_list_initialization(anchor_radius, x_max, y_max, num_bs):
    anchor_list = []
    anchor_index = 0
    anchor_x = anchor_radius
    anchor_y = anchor_radius

    while True:
        anchor_list.append(
            Anchor(anchor_index, anchor_x, anchor_y, anchor_radius, num_bs)
        )
        anchor_index += 1
        if anchor_x + 2 * anchor_radius <= x_max:
            anchor_x += 2 * anchor_radius
            continue
        else:
            if anchor_y + 2 * anchor_radius <= y_max:
                anchor_x = anchor_radius
                anchor_y += 2 * anchor_radius
                continue
            else:
                debug_print(
                    f"Anchor list initialization complete. {anchor_index} anchors created."
                )
                break

    return anchor_list
