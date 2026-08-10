# 在这里写聚类
import random
import numpy as np
import math


# (-1,1)归一化
def normalize_to_range(data, new_min=-1, new_max=1):
    old_min = min(data)
    old_max = max(data)

    normalized_data = [
        (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        for x in data
    ]
    return normalized_data


# (0,1)归一化
def normalize_to_01(data):
    min_val = min(data)
    max_val = max(data)

    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data


def clustering():
    # distance
    d = []
    for _ in range(100):
        random_number1 = random.randint(1, 100)
        d.append(random_number1)
    normalize_d = normalize_to_01(d)

    # rsrp
    r = []
    for _ in range(100):
        random_number2 = random.randint(1, 100)
        r.append(random_number2)
    normalize_r = normalize_to_range(r)

    # sinr
    s = []
    for _ in range(100):
        random_number3 = random.randint(1, 100)
        s.append(random_number3)
    normalize_s = normalize_to_range(s)

    # 凑成向量
    x = []
    for i in range(100):
        xi = np.array([normalize_d[i], normalize_r[i], normalize_s[i]])
        x.append(xi)

    # 开始减聚
    # 第一个公式
    p = []
    for o in range(100):
        pp = 0
        for i in range(100):
            euclidean_distance = np.sum((x[o] - x[i]) ** 2)
            new_p = math.exp(-3 * euclidean_distance)
            pp += new_p
        p.append(pp)

    # 第一个公式的最大值
    first_max_p = max(p)
    max_index = p.index(first_max_p)
    first_centre_p = first_max_p
    first_centre_x = x[max_index]
    choose_max_index = [max_index]

    x_centre = [first_centre_x]
    max_indices = [max_index]

    # 第二个公式
    while True:
        for i in range(100):
            if i in choose_max_index:
                continue
            else:
                p[i] = p[i] - p[max_index] * math.exp(
                    -2 * np.sum((x[i] - x[max_index]) ** 2)
                )
        other_numbers = []
        for i, value in enumerate(p):
            if i not in choose_max_index:
                other_numbers.append(value)
        max_p = max(other_numbers)
        max_index = p.index(max_p)
        max_indices.append(max_index)
        choose_max_index.append(max_index)
        x_centre.append(x[max_index])
        if max_p < 0.005 * first_centre_p:
            break

    # 分类聚类
    categories = [[] for _ in range(len(max_indices))]
    for num in list(range(100)):
        differences = [abs(num - ref) for ref in max_indices]
        nearest_max_indices_index = differences.index(min(differences))
        categories[nearest_max_indices_index].append(num)
    print(categories)
    print(len(categories))

    # 初始化一个字典来存储各个状态对应的小列表
    result_dict = {i: [] for i in range(1, len(categories) + 1)}

    # 按照规则将元素分类
    for index, num in enumerate(categories):
        state = (index % len(categories)) + 1
        result_dict[state].append(num)

    print(result_dict)
    # 打印每个状态对应的小列表
    for state, small_list in result_dict.items():
        print(f"状态 {state}: {small_list}")


if __name__ == "__main__":
    # 在这里调用
    clustering()
