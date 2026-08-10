import numpy as np


def levy_walk(
    levy_steps,
    levy_alpha,
    levy_scale,
    levy_position_list,
    levy_velocity_list,
    levy_step_x_list,
    levy_step_y_list,
    x_max,
    y_max,
):
    for _ in range(levy_steps):
        step_length = np.random.pareto(levy_alpha) * levy_scale  # 使用帕累托分布采样步长
        step_angle = np.random.uniform(0, 2 * np.pi)  # 随机选择步长方向

        step_x = step_length * np.cos(step_angle)
        step_y = step_length * np.sin(step_angle)

        new_step_x = levy_step_x_list[-1] + step_x
        new_step_y = levy_step_y_list[-1] + step_y

        # 边界检查，确保横纵坐标在0到100之间
        if new_step_x > x_max:
            new_step_x = x_max
        elif new_step_x < 0:
            new_step_x = 0

        if new_step_y > y_max:
            new_step_y = y_max
        elif new_step_y < 0:
            new_step_y = 0

        # print(f"new_step_x: {new_step_x}, new_step_y: {new_step_y}")

        new_position = np.array([float(new_step_x), float(new_step_y)])

        levy_step_x_list.append(new_step_x)
        levy_step_y_list.append(new_step_y)
        levy_position_list.append(new_position)

    for i in range(1, levy_steps + 1):
        velocity = levy_position_list[i] - levy_position_list[i - 1]  # 根据位置差计算速度
        levy_velocity_list.append(velocity)

    return levy_velocity_list, levy_position_list, levy_step_x_list, levy_step_y_list
