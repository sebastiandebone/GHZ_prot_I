"""
2020 David Elkouss, Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
_____________________________________________
"""
import random
import matplotlib.pyplot as plt
import statistics

prob = [0.7277, 0.7429, 0.8586, 0.8509, 0.8019, 0.8586, 0.8509, 0.8043, 0.8586, 0.8509, 0.6588, 0.8586, 0.8509, 0.6454]
steps = [7, 6, 4, 3, 5, 4, 3, 5, 4, 3, 5, 4, 3, 5]


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
    n2 = l2()
    n4 = l4()
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
    n7 = l7()
    if random.random() <= prob[7]:
        return n5 + n7 + steps[7]
    return n5 + n7 + steps[7] + l8()


def l10():
    n9 = level(prob[8], steps[8])
    if random.random() <= prob[9]:
        return n9 + steps[9]
    return n9 + steps[9] + l10()


def l11():
    n8 = max(l8(), l8())
    n10 = max(l10(), l10())
    if random.random() <= prob[10]:
        return n8 + n10 + steps[10]
    return n8 + n10 + steps[10] + l11()


def l13():
    n12 = level(prob[11], steps[11])
    if random.random() <= prob[12]:
        return n12 + steps[12]
    return n12 + steps[12] + l13()


def l14():
    n11 = l11()
    n13 = max(l13(), l13())
    if random.random() <= prob[13]:
        return n11 + n13 + steps[13]
    return n11 + n13 + steps[13] + l14()


# print(statistics.mean(nts_bins))                # We take the mean of all the Monte Carlo iterations
# plt.hist(nts_bins, bins=[33, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71])
# plt.show()