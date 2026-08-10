def min_max_normalization(node_list, state):
    stateNorm = state.copy()
    for i in range(len(node_list)):
        if node_list[i].nodeType == "enb":
            stateNorm[i] = state[i] / 503
        elif node_list[i].nodeType == "gnb":
            stateNorm[i] = state[i] / 1007

    return stateNorm
