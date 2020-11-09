"""
2020 Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
_____________________________________________
"""
import average_bell_pair_generation_steps.average_calculator as ac
import average_bell_pair_generation_steps.expedient as expedient
import average_bell_pair_generation_steps.stringent as stringent
import statistics
import matplotlib.pyplot as plt
import operations as op
# import known_protocols as kp
# import da_protocols as dap
# import pickle
# import plot_protocol as pprot


def Expedient(F=0.9):
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)

    AB2, p1 = op.purification(AB2, AB1, 1, True)
    AB3, p2 = op.purification(AB3, AB2, 1, True)

    AB2 = op.set_isotropic_state(F)
    AB2, p3 = op.purification(AB2, AB1, 1, True)
    AB3, p4 = op.purification(AB3, AB2, 2, True)

    CD3 = AB3

    BC1 = op.set_isotropic_state(F)
    BC2 = op.set_isotropic_state(F)
    BC2, p5 = op.purification(BC2, BC1, 2, True)
    BC2, p6 = op.purification(BC2, BC1, 1, True)

    AD2 = BC2

    ABCD = op.fuse_GHZ_ancilla(AB3, CD3, BC2)
    ABCD, p7 = op.purification(ABCD, AD2, 7, True)
    p8 = p5
    p9 = p6
    ABCD, p10 = op.purification(ABCD, BC2, 2, True)
    ABCD, p11 = op.purification(ABCD, AD2, 7, True)

    pb = [[[p1, p2, p3, p4], [p1, p2, p3, p4]], [[p5, p6], [p5, p6]], [[p7]], [[p8, p9], [p8, p9]], [[p10]], [[p11]]]
    ts = [[[2,  1,  2,  0],  [2,  1,  2,  0]],  [[2,  1],  [2,  1]],  [[0]],  [[2,  1],  [2,  1]],  [[0]],   [[0]]]
    fr = [0,                                    1,                    0,      3,                    0,       0]
    fl = [[[0,  0,  2,  0],  [0,  0,  2,  0]],  [[0,  0],  [0,  0]],  [[0]],  [[0,  0],  [0,  0]],  [[0]],   [[0]]]

    return ABCD, pb, ts, fr, fl


def Stringent(F=0.9):
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)

    AB2, p1a = op.purification(AB2, AB1, 1, True)
    AB3, p1b = op.purification(AB3, AB2, 1, True)

    AB2 = op.set_isotropic_state(F)
    AB2, p2a = op.purification(AB2, AB1, 1, True)
    AB3, p2b = op.purification(AB3, AB2, 2, True)

    AB2 = op.set_isotropic_state(F)
    AB2, p3 = op.purification(AB2, AB1, 2, True)
    AB2, p4 = op.purification(AB2, AB1, 1, True)

    AB2, p5a = op.purification(AB2, AB1, 1, True)
    AB3, p5b = op.purification(AB3, AB2, 1, True)

    AB2 = op.set_isotropic_state(F)
    AB2, p6 = op.purification(AB2, AB1, 2, True)
    AB2, p7 = op.purification(AB2, AB1, 1, True)

    AB2, p8a = op.purification(AB2, AB1, 1, True)
    AB3, p8b = op.purification(AB3, AB2, 2, True)

    CD3 = AB3

    BC1 = op.set_isotropic_state(F)
    BC2 = op.set_isotropic_state(F)
    BC2, p9 = op.purification(BC2, BC1, 2, True)
    BC2, p10 = op.purification(BC2, BC1, 1, True)
    BC2, p11 = op.purification(BC2, BC1, 1, True)

    AD2 = BC2

    ABCD = op.fuse_GHZ_ancilla(AB3, CD3, BC2)
    ABCD, p12 = op.purification(ABCD, AD2, 7, True)

    BC2 = op.set_isotropic_state(F)
    BC2, p13 = op.purification(BC2, BC1, 2, True)
    BC2, p14 = op.purification(BC2, BC1, 1, True)
    BC2, p15 = op.purification(BC2, BC1, 1, True)
    ABCD, p16a = op.purification(ABCD, BC2, 2, True)
    ABCD, p16b = op.purification(ABCD, AD2, 7, True)

    pb = [[[p1a, p1b, p2a, p2b, p3, p4, p5a, p5b, p6, p7, p8a, p8b], [p1a, p1b, p2a, p2b, p3, p4, p5a, p5b, p6, p7, p8a, p8b]], [[p9, p10, p11], [p9, p10, p11]], [[p12]], [[p13, p14, p15], [p13, p14, p15]], [[p16a]],  [[p16b]]]
    ts = [[[2,   1,   2,   0,   2,  1,  1,   0,   2,  1,  1,   0],   [2,   1,   2,   0,   2,  1,  1,   0,   2,  1,  1,   0]],   [[2,  1,   1],   [2,  1,   1]],   [[0]],   [[2,   1,   1],   [2,   1,   1]],   [[0]],     [[0]]]
    fr = [0,                                                                                                                    1,                                0,       3,                                  0,         0]
    fl = [[[0,   0,   2,   0,   4,  4,  4,   0,   8,  8,  8,   0],   [0,   0,   2,   0,   4,  4,  4,   0,   8,  8,  8,   0]],   [[0,  0,   0],   [0,  0,   0]],   [[0]],   [[0,   0,   0],   [0,   0,   0]],   [[0]],     [[0]]]

    return ABCD, pb, ts, fr, fl


def protocol414(F=0.9):
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)
    AB3, p1 = op.purification(AB3, AB2, 1, True)
    AB2, p2a = op.purification(AB2, AB1, 2, True)
    AB3, p2b = op.purification(AB3, AB2, 3, True)

    CD1 = op.set_isotropic_state(F)
    CD2 = op.set_isotropic_state(F)
    CD3 = op.set_isotropic_state(F)
    CD3, q1 = op.purification(CD3, CD2, 1, True)
    CD2, q2a = op.purification(CD2, CD1, 1, True)
    CD3, q2b = op.purification(CD3, CD2, 2, True)

    BC1 = op.set_isotropic_state(F)
    BC2 = op.set_isotropic_state(F)
    BC3 = op.set_isotropic_state(F)
    BC3, p3 = op.purification(BC3, BC2, 1, True)
    BC2, p4a = op.purification(BC2, BC1, 2, True)
    BC3, p4b = op.purification(BC3, BC2, 3, True)

    AD1 = op.set_isotropic_state(F)
    AD2 = op.set_isotropic_state(F)
    AD2, q3 = op.purification(AD2, AD1, 1, True)

    ABC = op.fuse_GHZ_local(AB3, BC3, 0, 0)
    ABCD = op.fuse_GHZ_local(ABC, CD3, 0, 0)
    ABCD, p5 = op.purification(ABCD, AD2, 7, True)

    pb = [[[p1, p2a, p2b], [q1, q2a, q2b]], [[p3, p4a, p4b], [q3]], [[p5]]]
    ts = [[[2,  2,   0],   [2,  2,   0]],   [[2,  2,   0],   [2]],  [[0]]]
    fr = [0,                                1,                      0]
    fl = [[[0,  1,   0],   [0,  1,   0]],   [[0,  1,   0],   [0]],  [[0]]]

    return ABCD, pb, ts, fr, fl


def protocol422(F=0.9):
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)
    AB3, p1 = op.purification(AB3, AB2, 3, True)
    AB2, p2a = op.purification(AB2, AB1, 1, True)
    AB3, p2b = op.purification(AB3, AB2, 2, True)

    CD1 = op.set_isotropic_state(F)
    CD2 = op.set_isotropic_state(F)
    CD3 = op.set_isotropic_state(F)
    CD4 = op.set_isotropic_state(F)
    CD4, q1 = op.purification(CD4, CD3, 3, True)
    CD4, q2 = op.purification(CD4, CD3, 1, True)
    CD3, q3a = op.purification(CD3, CD2, 3, True)
    CD4, q3b = op.purification(CD4, CD3, 2, True)
    CD3 = op.set_isotropic_state(F)
    CD3, q4 = op.purification(CD3, CD2, 2, True)
    CD2, q5a = op.purification(CD2, CD1, 3, True)
    CD3, q5b = op.purification(CD3, CD2, 1, True)
    CD4, q5c = op.purification(CD4, CD3, 1, True)

    BC1 = op.set_isotropic_state(F)
    BC2 = op.set_isotropic_state(F)
    BC3 = op.set_isotropic_state(F)
    BC3, p6 = op.purification(BC3, BC2, 3, True)
    BC2, p7a = op.purification(BC2, BC1, 1, True)
    BC3, p7b = op.purification(BC3, BC2, 2, True)

    AD1 = op.set_isotropic_state(F)
    AD2 = op.set_isotropic_state(F)
    AD3 = op.set_isotropic_state(F)
    AD4 = op.set_isotropic_state(F)
    AD4, q6 = op.purification(AD4, AD3, 2, True)
    AD2, q7 = op.purification(AD2, AD1, 1, True)
    AD3, q8a = op.purification(AD3, AD2, 1, True)
    AD4, q8b = op.purification(AD4, AD3, 3, True)

    ABC = op.fuse_GHZ_local(AB3, BC3, 0, 0)
    ABCD = op.fuse_GHZ_local(ABC, AD4, 2, 0)
    ABCD, p9 = op.purification(ABCD, CD4, 1, True)

    pb = [[[p1, p2a, p2b], [q1, q2, q3a, q3b, q4, q5a, q5b, q5c]], [[p6, p7a, p7b], [q6, q7, q8a, q8b]], [[p9]]]
    ts = [[[2,  2,   0],   [2,  1,  2,   0,   2,  2,   0,   0]],   [[2,  2,   0],   [2,  2,  1,   0]],   [[0]]]
    fr = [0,                                                       1,                                    0]
    fl = [[[0,  1,   0],   [0,  0,  2,   0,   4,  5,   4,   0]],   [[0,  1,   0],   [0,  1,  1,   0]],   [[0]]]

    return ABCD, pb, ts, fr, fl


def protocol442(F=0.9):
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)
    AB4 = op.set_isotropic_state(F)
    AB5 = op.set_isotropic_state(F)
    AB4, p1a = op.purification(AB4, AB3, 3, True)
    AB5, p1b = op.purification(AB5, AB4, 2, True)
    AB4 = op.set_isotropic_state(F)
    AB4, p2a = op.purification(AB4, AB3, 3, True)
    AB5, p2b = op.purification(AB5, AB4, 1, True)
    AB4 = op.set_isotropic_state(F)
    AB3, p3a = op.purification(AB3, AB2, 1, True)
    AB4, p3b = op.purification(AB4, AB3, 1, True)
    AB3 = op.set_isotropic_state(F)
    AB3, p4 = op.purification(AB3, AB2, 3, True)
    AB2, p5a = op.purification(AB2, AB1, 3, True)
    AB3, p5b = op.purification(AB3, AB2, 2, True)
    AB4, p5c = op.purification(AB4, AB3, 3, True)
    AB5, p5d = op.purification(AB5, AB4, 2, True)

    CD1 = op.set_isotropic_state(F)
    CD2 = op.set_isotropic_state(F)
    CD3 = op.set_isotropic_state(F)
    CD4 = op.set_isotropic_state(F)
    CD4, q1 = op.purification(CD4, CD3, 2, True)
    CD3, q2a = op.purification(CD3, CD2, 3, True)
    CD4, q2b = op.purification(CD4, CD3, 1, True)
    CD3 = op.set_isotropic_state(F)
    CD3, q3 = op.purification(CD3, CD2, 3, True)
    CD3, q4 = op.purification(CD3, CD2, 1, True)
    CD2, q5a = op.purification(CD2, CD1, 2, True)
    CD3, q5b = op.purification(CD3, CD2, 3, True)
    CD2 = op.set_isotropic_state(F)
    CD2, q6a = op.purification(CD2, CD1, 3, True)
    CD3, q6b = op.purification(CD3, CD2, 1, True)
    CD4, q6c = op.purification(CD4, CD3, 2, True)

    AC1 = op.set_isotropic_state(F)
    AC2 = op.set_isotropic_state(F)
    AC3 = op.set_isotropic_state(F)
    AC4 = op.set_isotropic_state(F)
    AC4, p7 = op.purification(AC4, AC3, 1, True)
    AC3, p8 = op.purification(AC3, AC2, 2, True)
    AC2, p9a = op.purification(AC2, AC1, 3, True)
    AC3, p9b = op.purification(AC3, AC2, 1, True)
    AC4, p9c = op.purification(AC4, AC3, 2, True)
    AC2 = op.set_isotropic_state(F)
    AC3 = op.set_isotropic_state(F)
    AC2, p10a = op.purification(AC2, AC1, 3, True)
    AC3, p10b = op.purification(AC3, AC2, 2, True)
    AC2 = op.set_isotropic_state(F)
    AC2, p11a = op.purification(AC2, AC1, 1, True)
    AC3, p11b = op.purification(AC3, AC2, 1, True)
    AC4, p11c = op.purification(AC4, AC3, 3, True)

    BD1 = op.set_isotropic_state(F)
    BD2 = op.set_isotropic_state(F)
    BD3 = op.set_isotropic_state(F)
    BD4 = op.set_isotropic_state(F)
    BD4, q7 = op.purification(BD4, BD3, 3, True)
    BD3, q8a = op.purification(BD3, BD2, 3, True)
    BD4, q8b = op.purification(BD4, BD3, 2, True)
    BD3 = op.set_isotropic_state(F)
    BD3, q9 = op.purification(BD3, BD2, 2, True)
    BD2, q10a = op.purification(BD2, BD1, 1, True)
    BD3, q10b = op.purification(BD3, BD2, 1, True)
    BD4, q10c = op.purification(BD4, BD3, 3, True)

    # ACD = op.fuse_GHZ_local(AC4, CD4, 0, 0)
    # ACDB = op.fuse_GHZ_local(ACD, AB5, 2, 0)
    ABC = op.fuse_GHZ_local(AB5, AC4, 1, 0)
    ABCD = op.fuse_GHZ_local(ABC, CD4, 0, 0)
    ABCD, p12 = op.purification(ABCD, BD4, 3, True)

    pb = [[[p1a, p1b, p2a, p2b, p3a, p3b, p4, p5a, p5b, p5c, p5d], [q1, q2a, q2b, q3, q4, q5a, q5b, q6a, q6b, q6c]], [[p7, p8, p9a, p9b, p9c, p10a, p10b, p11a, p11b, p11c], [q7, q8a, q8b, q9, q10a, q10b, q10c]], [[p12]]]
    ts = [[[2,   1,   2,   0,   2,   1,   2,  2,   0,   0,   0],   [2,  2,   0,   2,  1,  2,   0,   2,   0,   0]],   [[2,  2,  2,   0,   0,   2,    1,    2,    0,    0],    [2,  2,   0,   2,  2,    0,    0]],    [[0]]]
    fr = [0,                                                                                                         1,                                                                                             0]
    fl = [[[0,   0,   2,   0,   4,   4,   6,  7,   6,   4,   0],   [0,  1,   0,   3,  3,  5,   3,   7,   3,   0]],   [[0,  1,  2,   1,   0,   5,    5,    7,    5,    0],    [0,  1,   0,   3,  4,    3,    0]],    [[0]]]

    return ABCD, pb, ts, fr, fl


# # RECURSIVE LOGIC:
# nr_iter = 10000000  # Number of iterations for the Monte Carlo
# total = 0
# # for i_n in range(nr_iter):
# #     total = total + stringent.l14()
# # print(total * 1.0 / nr_iter + 2)
# for i_n in range(nr_iter):
#     total = total + expedient.l8()
# print(total * 1.0 / nr_iter + 2)

# PARAMETERS FROM APPENDIX D.2 OF NAOMI'S THESIS:
# # Expedient:
# pb = [[[0.7346, 0.7506], [0.7346, 0.7506]], [[0.8619, 0.8550], [0.8619, 0.8550]], [[0.8651]], [[0.8619, 0.8550], [0.8619, 0.8550]], [[0.8654]]]
# ts = [[[7,      6],      [7,      6]],      [[4,      3],      [4,      3]],      [[2]],      [[4,      3],      [4,      3]],      [[2]]]
# fr = [0,                                    1,                                    0,          3,                                    0]
# fl = [[[0,      0],      [0,      0]],      [[0,      0],      [0,      0]],      [[0]],      [[0,      0],      [0,      0]],      [[0]]]
# # Stringent:
# pb = [[[0.7277, 0.7429, 0.8586, 0.8509, 0.8019, 0.8586, 0.8509, 0.8043], [0.7277, 0.7429, 0.8586, 0.8509, 0.8019, 0.8586, 0.8509, 0.8043]], [[0.8586, 0.8509], [0.8586, 0.8509]], [[0.6588]], [[0.8586, 0.8509], [0.8586, 0.8509]], [[0.6454]]]
# ts = [[[7,      6,      4,      3,      5,      4,      3,      5],      [7,      6,      4,      3,      5,      4,      3,      5]],      [[4,      3],      [4,      3]],      [[5]],      [[4,      3],      [4,      3]],      [[5]]]
# fr = [0,                                                                                                                                    1,                                    0,          3,                                    0]
# fl = [[[0,      0,      2,      2,      0,      5,      5,      0],      [0,      0,      2,      2,      0,      5,      5,      0]],      [[0,      0],      [0,      0]],      [[0]],      [[0,      0],      [0,      0]],      [[0]]]
#
# print(statistics.mean(ac.av_calc(pb, ts, fr, fl, 1000000, False)))

# print_results = False   # Prints whether steps have succeeded or have failed
# nr_iter = 10000         # Number of iterations for the Monte Carlo
# F = 0.9                 # Fidelity of input isotropic Bell diagonal states
#
# state414, pb414, ts414, fr414, fl414 = protocol414(F)
# state422, pb422, ts422, fr422, fl422 = protocol422(F)
# state442, pb442, ts442, fr442, fl442 = protocol442(F)
# stateExp, pbExp, tsExp, frExp, flExp = Expedient(F)
# stateStr, pbStr, tsStr, frStr, flStr = Stringent(F)
#
# print("Dynamic program (4,14) : approximately {} Bell pair generation steps on average".format(statistics.mean(ac.av_calc(pb414, ts414, fr414, fl414, nr_iter, print_results))))
# print("Dynamic program (4,22) : approximately {} Bell pair generation steps on average".format(statistics.mean(ac.av_calc(pb422, ts422, fr422, fl422, nr_iter, print_results))))
# print("Dynamic program (4,42) : approximately {} Bell pair generation steps on average".format(statistics.mean(ac.av_calc(pb442, ts442, fr442, fl442, nr_iter, print_results))))
# print("Expedient              : approximately {} Bell pair generation steps on average".format(statistics.mean(ac.av_calc(pbExp, tsExp, frExp, flExp, nr_iter, print_results))))
# print("Stringent              : approximately {} Bell pair generation steps on average".format(statistics.mean(ac.av_calc(pbStr, tsStr, frStr, flStr, nr_iter, print_results))))
# # plt.hist(nts_bins, bins=[33, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71])
# # plt.show()
