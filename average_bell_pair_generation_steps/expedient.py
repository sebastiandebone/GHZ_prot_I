"""
2020 David Elkouss (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
_____________________________________________
"""
import random
import matplotlib.pyplot as plt
import statistics

prob = [0.7346, 0.7506, 0.8619, 0.8550, 0.8651, 0.8619, 0.8550, 0.8654]
steps = [7, 6, 4, 3, 2, 4, 3, 2]


def level(probability, time_steps):
    if random.random() <= probability:
        return time_steps
    return time_steps + level(probability, time_steps)


def l2():
    n1 = level(prob[0], steps[0])
    if random.random() <= prob[1]:
        return n1 + steps[1]
    return n1 + steps[1] + l2()


def l4():
    n3 = level(prob[2], steps[2])
    if random.random() <= prob[3]:
        return n3 + steps[3]
    return n3 + steps[3] + l4()


def l5():
    n2 = max(l2(), l2())
    n4 = max(l4(), l4())
    if random.random() <= prob[4]:
        return n2 + n4 + steps[4]
    return n2 + n4 + steps[4] + l5()


def l7():
    n6 = level(prob[5], steps[5])
    if random.random() <= prob[6]:
        return n6 + steps[6]
    return n6 + steps[6] + l7()


def l8():
    n5 = l5()
    n7 = max(l7(), l7())
    if random.random() <= prob[7]:
        return n5 + n7 + steps[7]
    return n5 + n7 + steps[7] + l8()


# print(statistics.mean(nts_bins))                # We take the mean of all the Monte Carlo iterations
# plt.hist(nts_bins, bins=[33, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71])
# plt.show()

# pb = [[0.7346, 0.7506], [0.8619, 0.8550], [0.8651], [0.8619, 0.8550], [0.8654]]
# ts = [[7,      6],      [4,      3],      [2],      [4,      3],      [2]]
# fr = [0,                1,                0,        3,                0]
# pr = [2,                2,                1,        2,                1]
#
# nr_levels = len(pb)
#
# def level(lvl):
#     n = 0
#     for i in range(len(pb[lvl])):
#         if random.random() <= pb[lvl][i]:
#             n = n + ts[lvl][i]
#         else:
#             return n =
#
#             return time_steps + standardprog(probability, time_steps)
#             npar[j] = npar[j] + standardprog(pb[lvl][i], ts[lvl][i])
#     n = n + max(npar)
#
#
# def standardprog(probability, time_steps):
#     if random.random() <= probability:
#         return time_steps
#     return time_steps + standardprog(probability, time_steps)

