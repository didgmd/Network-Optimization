from DebugPrint import *

# 初始化乒乓次数列表
lPPHOAllEpoch = []


def ppho_calculation(user_list, epoch):
    for i in range(epoch):
        n_ppho_per_epoch = 0
        for user in user_list:
            s6_epoch_list = user.s6TotalList[i]
            for j in range(2, len(s6_epoch_list)):
                if (
                    s6_epoch_list[j - 2] == s6_epoch_list[j]
                    and s6_epoch_list[j - 1] != s6_epoch_list[j]
                ):
                    n_ppho_per_epoch += 1

        lPPHOAllEpoch.append(n_ppho_per_epoch)

    debug_print(f"lPPHOAllEpoch {lPPHOAllEpoch}")
