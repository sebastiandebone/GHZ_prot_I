"""
2020 Runsheng Ouyang, Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I/
"""
import sys
import os.path
import numpy as np
import operations as op
import ancilla_rotations as ar
from math import floor
import known_protocols as kp
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import da_search as das
import da_protocols as dap
import os
import average_bell_pair_generation_steps.protocols_plus_probabilities as avg_bell_gs
import average_bell_pair_generation_steps.average_calculator as ac
import statistics


def plot_settings():
    # plt.rc('font',family='serif')
    # plt.rc('text', usetex=True)
    plt.rc('mathtext', fontset='stix')
    # plt.rc('font', family='STIXGeneral')
    plt.rc('font', family='Times New Roman')
    # plt.rcParams['figure.figsize'] = 8.5/2.54, 7/2.54

    # plt.rcParams["font.family"] = "Calibri"
    # plt.rcParams["font.style"] = "normal"
    # plt.rcParams["font.weight"] = "100"
    # plt.rcParams["font.stretch"] = "normal"
    plt.rcParams["font.size"] = 8
    plt.rcParams["lines.linewidth"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.3
    plt.rcParams["grid.linewidth"] = 0.3
    # plt.rcParams.update({'figure.autolayout': True})


def cm2inch(*tupl):
    """
    Function that converts a tuple from cm to inch.
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def get_random_seeds(size, max_number):
    """"
    Function that produces a list of random seed numbers.

    Parameters
    ----------
    size : int
        integer that determines how long the list of random seeds should be
    max_number : int
        integer that determines what the maximum number of the seeds in the list should be

    Returns
    ----------
    seed_list : list of integers of length size
        randomly generated list of seeds
    """
    seed_list = []
    for i in range(size):
        seed_list.append(random.randint(1, max_number))
    return seed_list


# FIGURES PAPER
def plot_base_program_diff_buffer_sizes():
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.
    """
    F_min = 0.8
    F_max = 1
    seg = 100
    n_max = 4
    k_max = 42
    F = np.arange(F_min, F_max, (F_max - F_min) / seg)
    outc = np.zeros([20, seg])

    if not(os.path.isfile('calc_data/sp_4_42_intF_0.8_1_100_1_0.txt')):
        das.store_data_intF(n_max, k_max, 'sp', F_min, F_max, seg, 1, 0)
    if not (os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_1_0.txt')):
        das.store_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 1, 0)
    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_2_0.txt')):
        das.store_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 2, 0)
    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_3_0.txt')):
        das.store_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 3, 0)
    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_4_0.txt')):
        das.store_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 4, 0)
    dataF0 = das.import_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 1, 0)
    dataF1 = das.import_data_intF(n_max, k_max, 'sp', F_min, F_max, seg, 1, 0)
    dataF2 = das.import_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 2, 0)
    dataF3 = das.import_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 3, 0)
    dataF4 = das.import_data_intF(n_max, k_max, 'mpF', F_min, F_max, seg, 4, 0)
    i_spike = das.spike(n_max, k_max, 'sp', F_min, F_max, seg, 1, 0, 0.85)
    # print(dataF2[1][2][4][0].t1)
    # print(dataF2[1][2][2][1].state[0])
    # print(dataF2[1][2][4][0].t2)
    # protocol = dap.identify_protocol(dataF2[1], 4, 42, 0)
    # print(dap.operate_protocol(protocol, 2, 0.85))
    # return
    # pprot.plot_protocol(protocol, 1)
    i_spike3 = das.spike(n_max, k_max, 'mpF', F_min, F_max, seg, 3, 0, 0.85)

    for i in range(seg):
        outc[5, i] = dataF0[i][n_max][k_max][0].state[0]
        outc[0, i] = dataF1[i][n_max][k_max][0].state[0]
        outc[1, i] = dap.operate_protocol(dap.identify_protocol(dataF1[i_spike], n_max, k_max, 0), 1, F[i])[0]
        outc[2, i] = dataF2[i][n_max][k_max][0].state[0]
        outc[6, i] = dap.operate_protocol(dap.identify_protocol(dataF3[i_spike3], n_max, k_max, 0), 3, F[i])[0]
        outc[3, i] = dataF3[i][n_max][k_max][0].state[0]
        outc[4, i] = dataF4[i][n_max][k_max][0].state[0]

        outc[15, i] = 1 - dataF0[i][n_max][k_max][0].state[0]
        outc[10, i] = 1 - dataF1[i][n_max][k_max][0].state[0]
        outc[11, i] = 1 - dap.operate_protocol(dap.identify_protocol(dataF1[i_spike], n_max, k_max, 0), 1, F[i])[0]
        outc[12, i] = 1 - dataF2[i][n_max][k_max][0].state[0]
        outc[16, i] = 1 - dap.operate_protocol(dap.identify_protocol(dataF3[i_spike3], n_max, k_max, 0), 3, F[i])[0]
        outc[13, i] = 1 - dataF3[i][n_max][k_max][0].state[0]
        outc[14, i] = 1 - dataF4[i][n_max][k_max][0].state[0]

    # print(dataF0[50][4][42][0].state[0])
    # print(dataF1[50][4][42][0].state[0])
    # print(dataF2[50][4][42][0].state[0])
    # print("")

    plot_settings()
    fig, ax1 = plt.subplots()
    ax1.plot(F, outc[1, :], color='red', linestyle='--', label='Protocol found for $b=1$ at $F_\mathrm{Bell}=' + str(round(F[i_spike], 3)) + '$')
    ax1.plot(F, outc[6, :], color='blue', linestyle=':', label='Protocol found for $b=3$ at $F_\mathrm{Bell}=' + str(round(F[i_spike3], 3)) + '$')
    ax1.plot(F, outc[0, :], color='k', label='Dynamic program (' + str(n_max) + ', ' + str(k_max) + '), $b=1$')
    ax1.plot(F, outc[2, :], color='0.7', label='Dynamic program (' + str(n_max) + ', ' + str(k_max) + '), $b=2$')
    ax1.plot(F, outc[3, :], color='k', linestyle='--', label='Dynamic program (' + str(n_max) + ', ' + str(k_max) + '), $b=3$')
    # ax1.plot(F, outc[4, :], color='k', linestyle=':', label='Dynamic program (' + str(n_max) + ', ' + str(k_max) + '), $b=4$')
    # ax1.plot(F, outc[5, :], color='purple', linestyle='--', label='Confirm $b=1$')

    # ax2.plot(F, outc[11, :], color='red', linestyle='--')
    # ax2.plot(F, outc[16, :], color='blue', linestyle=':')
    # ax2.plot(F, outc[10, :], color='k')
    # ax2.plot(F, outc[12, :], color='0.7')
    # ax2.plot(F, outc[13, :], color='k', linestyle='--')
    # # ax2.plot(F, outc[14, :], color='k', linestyle=':')

    # ax2.set_yscale('log')

    plt.sca(ax1)
    plt.legend(loc='lower right', ncol=1, handleheight=1.2, labelspacing=0.03) #, bbox_to_anchor=(1, 0.5))
    ax1.set(xlabel=r'Fidelity $F_\mathrm{Bell}$ of isotropic Bell pairs used',
            ylabel=r'Fidelity $F_\mathrm{GHZ}$ of final GHZ state')
    ax1.grid()
    # ax2.set(xlabel=r'Fidelity $F_\mathrm{Bell}$ of isotropic Bell pairs used',
    #        ylabel=r'Infidelity $1 - F_\mathrm{GHZ}$ of final GHZ state')
    ax1.tick_params(direction='inout')
    # ax2.tick_params(direction='inout')
    ax1.grid(which='minor', alpha=0.2)
    ax1.grid(which='major', alpha=0.5)
    # ax2.grid(which='minor', alpha=0.2)
    # ax2.grid(which='major', alpha=0.5)
    # plt.sca(ax2)
    # plt.grid(True, which="both")
    plt.sca(ax1)
    plt.xticks([0.8, 0.85, 0.9, 0.95, 1], ['0.8', '0.85', '0.9', '0.95', '1'])
    # plt.sca(ax2)
    # plt.xticks([0.8, 0.85, 0.9, 0.95, 1], ['0.8', '0.85', '0.9', '0.95', '1'])

    fig.set_size_inches(cm2inch(8.5, 6.5))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig('figures/plot_base_program_diff_buffer_sizes.pdf', pad_inches=0, bbox_inches='tight')
    return


def plot_base_program_diff_buffer_sizes_additional_figure():
    """
    Function that creates an additional figure shining more light on one of the figures in the paper (specifically for
    input fidelity F = 0.825). Can use a pre-calculated file with data: if it is not there it will be created for future
    use of the function.
    """
    F = 0.825
    if not (os.path.isfile('calc_data/data0.p')):
        data0 = das.dynamic_algorithm(4, 42, F, 'mpF', 1, 0, 0)
        pickle.dump(data0, open("calc_data/data0.p", "wb"))
    if not (os.path.isfile('calc_data/data0_rot.p')):
        data0_rot = das.dynamic_algorithm(4, 42, F, 'mpF', 1, 1, 0)
        pickle.dump(data0_rot, open("calc_data/data0_rot.p", "wb"))
    if not (os.path.isfile('calc_data/data2.p')):
        data2 = das.dynamic_algorithm(4, 42, F, 'mpF', 2, 0, 0)
        pickle.dump(data2, open("calc_data/data2.p", "wb"))
    if not (os.path.isfile('calc_data/data3.p')):
        data3 = das.dynamic_algorithm(4, 42, F, 'mpF', 3, 0, 0)
        pickle.dump(data3, open("calc_data/data3.p", "wb"))
    if not (os.path.isfile('calc_data/data4.p')):
        data4 = das.dynamic_algorithm(4, 42, F, 'mpF', 4, 0, 0)
        pickle.dump(data4, open("calc_data/data4.p", "wb"))
    if not (os.path.isfile('calc_data/data5.p')):
        data5 = das.dynamic_algorithm(4, 42, F, 'mpF', 5, 0, 0)
        pickle.dump(data5, open("calc_data/data5.p", "wb"))
    if not (os.path.isfile('calc_data/data6.p')):
        data6 = das.dynamic_algorithm(4, 42, F, 'mpF', 6, 0, 0)
        pickle.dump(data6, open("calc_data/data6.p", "wb"))
    if not (os.path.isfile('calc_data/data8.p')):
        data8 = das.dynamic_algorithm(4, 42, F, 'mpF', 8, 0, 0)
        pickle.dump(data8, open("calc_data/data8.p", "wb"))
    if not (os.path.isfile('calc_data/data40.p')):
        data40 = das.dynamic_algorithm(4, 42, F, 'mpF', 40, 0, 0)
        pickle.dump(data40, open("calc_data/data40.p", "wb"))

    data0 = pickle.load(open("calc_data/data0.p", "rb"))
    data2 = pickle.load(open("calc_data/data2.p", "rb"))
    data3 = pickle.load(open("calc_data/data3.p", "rb"))
    data4 = pickle.load(open("calc_data/data4.p", "rb"))
    data5 = pickle.load(open("calc_data/data5.p", "rb"))
    data6 = pickle.load(open("calc_data/data6.p", "rb"))
    data8 = pickle.load(open("calc_data/data8.p", "rb"))
    data40 = pickle.load(open("calc_data/data40.p", "rb"))
    data0_rot = pickle.load(open("calc_data/data0_rot.p", "rb"))

    outc = np.zeros([21, 42])
    outc56 = np.zeros([6, 42])

    for n in range(2, 5):
        for k in range(n - 1, 43):
            outc[0 + (n-2)*7, k - (n - 1)] = 1 - data0[n][k][0].state[0]
            outc[1 + (n-2)*7, k - (n - 1)] = 1 - data0_rot[n][k][0].state[0]
            outc[2 + (n-2)*7, k - (n - 1)] = 1 - data2[n][k][0].state[0]    # 1 overlaps with this one
            outc[3 + (n-2)*7, k - (n - 1)] = 1 - data3[n][k][0].state[0]    # 1 overlaps
            outc[4 + (n-2)*7, k - (n - 1)] = 1 - data3[n][k][2].state[0]
            outc[5 + (n-2)*7, k - (n - 1)] = 1 - data4[n][k][0].state[0]    # 1 and 2 both overlap
            outc[6 + (n-2)*7, k - (n - 1)] = 1 - data4[n][k][3].state[0]
            outc56[0 + (n-2)*2, k - (n - 1)] = 1 - data8[n][k][0].state[0]
            outc56[1 + (n-2)*2, k - (n - 1)] = 1 - data40[n][k][0].state[0]

    x_list = np.ones(42)
    x_list2 = np.ones(41)
    x_list3 = np.ones(40)
    for i in range(1, 43):
        x_list[i - 1] = i
    for i in range(2, 43):
        x_list2[i - 2] = i
    for i in range(3, 43):
        x_list3[i - 3] = i

    plot_settings()
    fig, ax = plt.subplots()
    # ax.plot(x_list, outc[0, :], color='k', linestyle='-', label=r'$b=1$')
    # ax.plot(x_list, outc[1, :], color='0.7', linestyle='-', label=r'$b=1$, w/ rot.')
    ax.plot(x_list, outc[2, :], color='red', linestyle='--', label=r'$b=2$, pos. 1, 2')
    ax.plot(x_list, outc[3, :], color='blue', linestyle=':', label=r'$b=3$, pos. 1, 2')
    # ax.plot(x_list, outc[4, :], color='tab:blue', linestyle='--', label=r'$b=3$, pos. 3')
    ax.plot(x_list, outc[5, :], color='purple', linestyle=':', label='$b=4$, pos. 1, 2, 3')
    # ax.plot(x_list, outc[6, :], color='tab:purple', linestyle='--', label='$b=4$, pos. 4')
    ax.plot(x_list, outc56[0, :], color='yellow', linestyle='--', label='$b=5$, pos. 1')
    ax.plot(x_list, outc56[1, :], color='green', linestyle='--', label='$b=6$, pos. 1')

    # ax.plot(x_list2, outc[7, 0:41], color='k', linestyle='-')
    # ax.plot(x_list2, outc[8, 0:41], color='0.7', linestyle='-')
    ax.plot(x_list2, outc[9, 0:41], color='red', linestyle='--')
    ax.plot(x_list2, outc[10, 0:41], color='blue', linestyle=':')
    # ax.plot(x_list2, outc[11, 0:41], color='tab:blue', linestyle='--')
    ax.plot(x_list2, outc[12, 0:41], color='purple', linestyle=':')
    # ax.plot(x_list2, outc[13, 0:41], color='tab:purple', linestyle='--')
    ax.plot(x_list2, outc56[2, 0:41], color='yellow', linestyle='--')
    ax.plot(x_list2, outc56[3, 0:41], color='green', linestyle='--')

    # ax.plot(x_list3, outc[14, 0:40], color='k', linestyle='-')
    # ax.plot(x_list3, outc[15, 0:40], color='0.7', linestyle='-')
    ax.plot(x_list3, outc[16, 0:40], color='red', linestyle='--')
    ax.plot(x_list3, outc[17, 0:40], color='blue', linestyle=':')
    # ax.plot(x_list3, outc[18, 0:40], color='tab:blue', linestyle='--')
    ax.plot(x_list3, outc[19, 0:40], color='purple', linestyle=':')
    # ax.plot(x_list3, outc[20, 0:40], color='tab:purple', linestyle='--')
    ax.plot(x_list3, outc56[4, 0:40], color='yellow', linestyle='--')
    ax.plot(x_list3, outc56[5, 0:40], color='green', linestyle='--')

    ax.set_yscale('log')

    plt.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03) #, bbox_to_anchor=(1, 0.5))
    ax.set(xlabel=r'Number of isotropic input Bell pairs used with $F_\mathrm{Bell}=0.825$',
           ylabel=r'Infidelity $1-F$ of final state')
    ax.grid()
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xticks(range(1, 43), minor=True)
    # ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    fig.set_size_inches(cm2inch(8.5, 6.5))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    plt.savefig('figures/plot_base_program_diff_buffer_sizes_additional_figure.pdf', pad_inches=0, bbox_inches='tight')

    return


def plot_rand_diff_temperatures(use_precalculated_seeds=True):
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.

    Parameters
    ----------
    use_precalculated_seeds : Boolean
        Determines whether or not a set of precalculated seeds should be used at each input fidelity (i.e., the set of
        seeds used in the paper.
    """
    F_min = 0.8
    F_max = 1
    seg = 100
    n_max = 4
    k_max = 42
    F = np.arange(F_min, F_max, (F_max - F_min) / seg)
    outc = np.zeros([12, seg])
    T1 = 1e-05
    T2 = 0.1
    T3 = 1

    if use_precalculated_seeds and seg == 100:
        seed_list = [7616, 2787, 4471, 1985, 8927, 7603, 5827, 8346, 6187, 6378, 6753, 1873, 4251, 2811, 456, 3035,
                     9801, 3964, 8553, 1782, 473, 7010, 9414, 8607, 2076, 3375, 6996, 9099, 3623, 6142, 9150, 9399,
                     2653, 4744, 1249, 2549, 2261, 732, 8252, 7633, 7356, 9670, 9019, 6009, 5780, 9430, 1099, 8757,
                     2377, 6416, 4936, 9394, 3304, 3278, 7916, 8367, 8284, 5650, 2890, 1912, 5950, 9888, 7604, 2659,
                     9451, 7224, 228, 5669, 1345, 9167, 2763, 9128, 3542, 8714, 6631, 4920, 9341, 3059, 8014, 6726,
                     9371, 2599, 9891, 7465, 5968, 8040, 1291, 9416, 9558, 9271, 2217, 3950, 9245, 5720, 1067, 3559,
                     1431, 5065, 2456, 3243, 896]
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_1_0_7616-896_{}.txt'.format(T1))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T1)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_10_0_7616-896_{}.txt'.format(T1))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T1)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_50_0_7616-896_{}.txt'.format(T1))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T1)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_200_0_7616-896_{}.txt'.format(T1))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T2)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_1_0_7616-896_{}.txt'.format(T2))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T2)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_10_0_7616-896_{}.txt'.format(T2))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T2)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_50_0_7616-896_{}.txt'.format(T2))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T2)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_200_0_7616-896_{}.txt'.format(T2))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T2)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_1_0_7616-896_{}.txt'.format(T3))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T3)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_10_0_7616-896_{}.txt'.format(T3))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T3)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_50_0_7616-896_{}.txt'.format(T3))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T3)
        if not (os.path.isfile('calc_data/random_4_42_intF_0.8_1_100_200_0_7616-896_{}.txt'.format(T3))):
            das.store_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T3)
        dataF1 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T1)
        dataF2 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T1)
        dataF3 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T1)
        dataF4 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T1)
        dataF5 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T2)
        dataF6 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T2)
        dataF7 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T2)
        dataF8 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T2)
        dataF9 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 1, 0, seed_list, T3)
        dataF10 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 10, 0, seed_list, T3)
        dataF11 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 50, 0, seed_list, T3)
        dataF12 = das.import_data_intF(n_max, k_max, 'random', F_min, F_max, seg, 200, 0, seed_list, T3)

    else:
        seed_list = get_random_seeds(seg + 1, 10000)
        print(seed_list)
        dataF1 = [None] * seg
        dataF2 = [None] * seg
        dataF3 = [None] * seg
        dataF4 = [None] * seg
        dataF5 = [None] * seg
        dataF6 = [None] * seg
        dataF7 = [None] * seg
        dataF8 = [None] * seg
        dataF9 = [None] * seg
        dataF10 = [None] * seg
        dataF11 = [None] * seg
        dataF12 = [None] * seg
        for i in range(seg):
            current_time = time.ctime(time.time())
            print(i, current_time, F[i])
            dataF1[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 1, 0, 0, seed_list[i], T1)
            dataF2[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 10, 0, 0, seed_list[i], T1)
            dataF3[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 50, 0, 0, seed_list[i], T1)
            dataF4[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 200, 0, 0, seed_list[i], T1)
            dataF5[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 1, 0, 0, seed_list[i], T2)
            dataF6[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 10, 0, 0, seed_list[i], T2)
            dataF7[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 50, 0, 0, seed_list[i], T2)
            dataF8[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 200, 0, 0, seed_list[i], T2)
            dataF9[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 1, 0, 0, seed_list[i], T3)
            dataF10[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 10, 0, 0, seed_list[i], T3)
            dataF11[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 50, 0, 0, seed_list[i], T3)
            dataF12[i] = das.dynamic_algorithm(n_max, k_max, F[i], 'random', 200, 0, 0, seed_list[i], T3)

    for i in range(seg):
        outc[0, i] = 1 - dataF1[i][n_max][k_max][0].state[0]
        outc[1, i] = 1 - dataF2[i][n_max][k_max][0].state[0]
        outc[2, i] = 1 - dataF3[i][n_max][k_max][0].state[0]
        outc[3, i] = 1 - dataF4[i][n_max][k_max][0].state[0]
        outc[4, i] = 1 - dataF5[i][n_max][k_max][0].state[0]
        outc[5, i] = 1 - dataF6[i][n_max][k_max][0].state[0]
        outc[6, i] = 1 - dataF7[i][n_max][k_max][0].state[0]
        outc[7, i] = 1 - dataF8[i][n_max][k_max][0].state[0]
        outc[8, i] = 1 - dataF9[i][n_max][k_max][0].state[0]
        outc[9, i] = 1 - dataF10[i][n_max][k_max][0].state[0]
        outc[10, i] = 1 - dataF11[i][n_max][k_max][0].state[0]
        outc[11, i] = 1 - dataF12[i][n_max][k_max][0].state[0]

    plot_settings()
    # color_ba = '0.5'
    fig, (ax3, ax, ax2) = plt.subplots(3, 1, sharex='col', gridspec_kw={'hspace': 0, 'wspace': 0})

    ax3.plot(F, outc[0, :], color='0.7', label='Random program, $b=1$')
    ax3.plot(F, outc[1, :], color='black', linestyle='--', label='Random program, $b=10$')
    ax3.plot(F, outc[2, :], color='black', linestyle=':', label='Random program, $b=50$')
    ax3.plot(F, outc[3, :], color='black', label='Random program, $b=200$')
    ax.plot(F, outc[4, :], color='0.7', label='Random program, $b=1$')
    ax.plot(F, outc[5, :], color='black', linestyle='--', label='Random program, $b=10$')
    ax.plot(F, outc[6, :], color='black', linestyle=':', label='Random program, $b=50$')
    ax.plot(F, outc[7, :], color='black', label='Random program, $b=200$')
    ax2.plot(F, outc[8, :], color='0.7', label='Random program, $b=1$')
    ax2.plot(F, outc[9, :], color='black', linestyle='--', label='Random program, $b=10$')
    ax2.plot(F, outc[10, :], color='black', linestyle=':', label='Random program, $b=50$')
    ax2.plot(F, outc[11, :], color='black', label='Random program, $b=200$')

    ax3.text(0.81, 1e-5, r'$T={}$'.format(T1))
    ax.text(0.81, 7e-5, r'$T={}$'.format(T2))
    ax2.text(0.81, 2.5e-4, r'$T={}$'.format(T3))

    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax2.set(xlabel=r'Fidelity $F_\mathrm{Bell}$ of isotropic Bell pairs used')
    ax.set(ylabel=r'Infidelity $1-F_\mathrm{GHZ}$ of final $(n,k)=(4,42)$ GHZ state')

    # plt.xlabel(r'Number of Bell pairs used ($F_\mathrm{Bell}=0.9$)')
    # plt.ylabel(r'$1-F_\mathrm{GHZ}$ of the final GHZ state')
    plt.sca(ax)
    plt.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03)
    plt.sca(ax2)
    plt.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03)
    plt.sca(ax3)
    plt.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03)

    # plt.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03) #, bbox_to_anchor=(1, 0.5))

    ax.tick_params(direction='inout')
    ax2.tick_params(direction='inout')
    ax3.tick_params(direction='inout')

    # plt.sca(ax)
    # plt.ylim(0.00000001, 1.5)
    # plt.sca(ax2)
    # plt.ylim(0.00000001, 1.5)
    # plt.sca(ax3)
    # plt.ylim(0.00000001, 1.5)

    # plt.sca(ax)
    # plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])#np.arange(min(x_list), max(x_list), 6.0))
    # plt.sca(ax)
    # plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])
    # plt.sca(ax)
    # plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])
    # # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # # ax.grid()

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(locmin)
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.yaxis.set_minor_locator(locmin)
    ax3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.sca(ax)
    plt.grid(True, which="both")
    plt.sca(ax2)
    plt.grid(True, which="both")
    plt.sca(ax3)
    plt.grid(True, which="both")
    fig.set_size_inches(cm2inch(8.5, 15.5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/plot_rand_diff_temperatures_{}.pdf'.format(seed_list[0]), pad_inches=0, bbox_inches='tight')
    return


def best_random_protocols(n_max=8, k_max=80, F_input=0.9, nstate=200, inc_rot=0):
    seed_list = [3316, 7876, 570, 1264, 9343, 6959, 1162, 2100, 5177, 8559, 5454, 8917, 6232, 2994, 9603, 9296, 8193,
                 9321, 4319, 4239, 4010, 7355, 9398, 9047, 273, 9697, 6637, 8965, 2599, 5148, 6372, 5911, 3844, 17,
                 5263, 200, 4720, 787, 5339, 7157, 8184, 5289, 9342, 9304, 3409, 4122, 2967, 1789, 3048, 4734, 4831,
                 6272, 6897, 8397, 3360, 1109, 8164, 1361, 9541, 5428, 6766, 1837, 8560, 1043, 6328, 701, 1082, 3725,
                 852, 6029, 7106, 8174, 2556, 7533, 6013, 9076, 7502, 4950, 8562, 4164, 561, 6941, 1464, 4496, 4230,
                 8111, 9981, 5976, 9707, 8695, 2589, 3038, 1126, 7144, 6165, 845, 1555, 8660, 9783, 6466, 9075, 9674,
                 1526, 1222, 4328, 4231, 1820, 6602, 6091, 1714, 2421]
    T_list = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003,
              0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    nT = np.size(T_list)

    best_states = np.empty((n_max + 1, k_max + 1, 3), dtype=object)
    for i1 in range(n_max + 1):
        for i2 in range(k_max + 1):
            for i3 in range(3):
                best_states[i1][i2][i3] = 0

    for i in range(44):
        dataT = das.import_data_varT(n_max, k_max, 'random', F_input, nstate, inc_rot, seed_list[i])
        for n in range(2, n_max + 1):
            for k in range(n - 1, k_max + 1):
                for i3 in range(nT):
                    if dataT[i3][n][k][0] > best_states[n][k][0]:
                        best_states[n][k][0] = dataT[i3][n][k][0]
                        best_states[n][k][1] = T_list[i3]
                        best_states[n][k][2] = seed_list[i]

    return best_states


def plot_comparison_programs():
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.
    """
    outc2 = np.ones([10, 42])
    outc3 = np.ones([10, 41])
    outc4 = np.ones([10, 40])

    # The fidelities of the best states at F_Bell = 0.9 found with 44 seeds (and 18 temperatures) with the random algorithm:
    best_states = best_random_protocols()

    if not(os.path.isfile('calc_data/mpc_4_42_intF_0.8_1_100_2_0.txt')):
        das.store_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 0)
    msda_no_rot = das.import_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 0)

    if not(os.path.isfile('calc_data/mpc_4_42_intF_0.8_1_100_2_1.txt')):
        das.store_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 1)
    msda_inc_rot = das.import_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 1)

    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_2_0.txt')):
        das.store_data_intF(4, 42, 'mpF', 0.8, 1, 100, 2, 0)
    msdabf2 = das.import_data_intF(4, 42, 'mpF', 0.8, 1, 100, 2, 0)

    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_3_0.txt')):
        das.store_data_intF(4, 42, 'mpF', 0.8, 1, 100, 3, 0)
    msdabf3 = das.import_data_intF(4, 42, 'mpF', 0.8, 1, 100, 3, 0)

    if not(os.path.isfile('calc_data/mpF_4_42_intF_0.8_1_100_4_0.txt')):
        das.store_data_intF(4, 42, 'mpF', 0.8, 1, 100, 4, 0)
    msdabf4 = das.import_data_intF(4, 42, 'mpF', 0.8, 1, 100, 4, 0)

    if not(os.path.isfile('calc_data/sp_4_42_intF_0.8_1_100_1_0.txt')):
        das.store_data_intF(4, 42, 'sp', 0.8, 1, 100, 1, 0)
    ssda = das.import_data_intF(4, 42, 'sp', 0.8, 1, 100, 1, 0)

    for i in range(3):  # n = 2, 3, 4
        if i == 0:
            for j in range(42):  # now 42
                n = 2
                k = j + 1
                i_spike_no_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_inc_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 1, 0.9)
                F_no_rot = dap.operate_protocol(dap.identify_protocol(msda_no_rot[i_spike_no_rot], n, k, 0), 2, 0.9)[0]
                F_inc_rot = dap.operate_protocol(dap.identify_protocol(msda_inc_rot[i_spike_inc_rot], n, k, 0), 2, 0.9)[0]
                outc2[0, j] = 1 - F_no_rot
                outc2[1, j] = 1 - F_inc_rot
                outc2[2, j] = 1 - best_states[n][k][0]
                i_spike_msdabf2 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_msdabf3 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 3, 0, 0.9)
                i_spike_msdabf4 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 4, 0, 0.9)
                i_spike_ssda = das.spike(4, 42, 'sp', 0.8, 1, 100, 1, 0, 0.9)
                F_msdabf2 = dap.operate_protocol(dap.identify_protocol(msdabf2[i_spike_msdabf2], n, k, 0), 2, 0.9)[0]
                F_msdabf3 = dap.operate_protocol(dap.identify_protocol(msdabf3[i_spike_msdabf3], n, k, 0), 3, 0.9)[0]
                F_msdabf4 = dap.operate_protocol(dap.identify_protocol(msdabf4[i_spike_msdabf4], n, k, 0), 4, 0.9)[0]
                F_ssda = dap.operate_protocol(dap.identify_protocol(ssda[i_spike_ssda], n, k, 0), 1, 0.9)[0]
                outc2[6, j] = 1 - F_msdabf2
                outc2[5, j] = 1 - F_msdabf3
                outc2[3, j] = 1 - F_msdabf4
                outc2[4, j] = 1 - F_ssda
                print(n, k)
        elif i == 1:
            for j in range(41):  # now 41
                n = 3
                k = j + 2
                i_spike_no_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_inc_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 1, 0.9)
                F_no_rot = dap.operate_protocol(dap.identify_protocol(msda_no_rot[i_spike_no_rot], n, k, 0), 2, 0.9)[0]
                F_inc_rot = dap.operate_protocol(dap.identify_protocol(msda_inc_rot[i_spike_inc_rot], n, k, 0), 2, 0.9)[0]
                outc3[0, j] = 1 - F_no_rot  # now 42T
                outc3[1, j] = 1 - F_inc_rot  # now 42
                outc3[2, j] = 1 - best_states[n][k][0]  # now 42
                i_spike_msdabf2 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_msdabf3 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 3, 0, 0.9)
                i_spike_msdabf4 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 4, 0, 0.9)
                i_spike_ssda = das.spike(4, 42, 'sp', 0.8, 1, 100, 1, 0, 0.9)
                F_msdabf2 = dap.operate_protocol(dap.identify_protocol(msdabf2[i_spike_msdabf2], n, k, 0), 2, 0.9)[0]
                F_msdabf3 = dap.operate_protocol(dap.identify_protocol(msdabf3[i_spike_msdabf3], n, k, 0), 3, 0.9)[0]
                F_msdabf4 = dap.operate_protocol(dap.identify_protocol(msdabf4[i_spike_msdabf4], n, k, 0), 4, 0.9)[0]
                F_ssda = dap.operate_protocol(dap.identify_protocol(ssda[i_spike_ssda], n, k, 0), 1, 0.9)[0]
                outc3[6, j] = 1 - F_msdabf2
                outc3[5, j] = 1 - F_msdabf3
                outc3[3, j] = 1 - F_msdabf4
                outc3[4, j] = 1 - F_ssda
                print(n, k)
        elif i == 2:
            for j in range(40):  # now 40
                n = 4
                k = j + 3
                i_spike_no_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_inc_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 1, 0.9)
                F_no_rot = dap.operate_protocol(dap.identify_protocol(msda_no_rot[i_spike_no_rot], n, k, 0), 2, 0.9)[0]
                F_inc_rot = dap.operate_protocol(dap.identify_protocol(msda_inc_rot[i_spike_inc_rot], n, k, 0), 2, 0.9)[0]
                outc4[0, j] = 1 - F_no_rot  # now 42+41
                outc4[1, j] = 1 - F_inc_rot  # now 42+41
                outc4[2, j] = 1 - best_states[n][k][0]  # now 42+41
                i_spike_msdabf2 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 2, 0, 0.9)
                i_spike_msdabf3 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 3, 0, 0.9)
                i_spike_msdabf4 = das.spike(4, 42, 'mpF', 0.8, 1, 100, 4, 0, 0.9)
                i_spike_ssda = das.spike(4, 42, 'sp', 0.8, 1, 100, 1, 0, 0.9)
                F_msdabf2 = dap.operate_protocol(dap.identify_protocol(msdabf2[i_spike_msdabf2], n, k, 0), 2, 0.9)[0]
                F_msdabf3 = dap.operate_protocol(dap.identify_protocol(msdabf3[i_spike_msdabf3], n, k, 0), 3, 0.9)[0]
                F_msdabf4 = dap.operate_protocol(dap.identify_protocol(msdabf4[i_spike_msdabf4], n, k, 0), 4, 0.9)[0]
                F_ssda = dap.operate_protocol(dap.identify_protocol(ssda[i_spike_ssda], n, k, 0), 1, 0.9)[0]
                outc4[6, j] = 1 - F_msdabf2
                outc4[5, j] = 1 - F_msdabf3
                outc4[3, j] = 1 - F_msdabf4
                outc4[4, j] = 1 - F_ssda
                print(n, k)

    x_list = np.empty((42), dtype = int) # now 42+41+40
    # list for x axis
    for i in range(42): # now 42+41+30
        x_list[i] = i + 1

    outc2tot = np.zeros([1, 42])  # now 42+41+40
    outc3tot = np.zeros([1, 41])  # now 42+41+40
    outc4tot = np.zeros([1, 40])  # now 42+41+40

    for i in range(42):
        outc2tot[0, i] = min([outc2[0,i], outc2[1,i], outc2[2,i], outc2[3,i], outc2[4,i], outc2[5,i], outc2[6,i]])
    for i in range(41):
        outc3tot[0, i] = min([outc3[0,i], outc3[1,i], outc3[2,i], outc3[3,i], outc3[4,i], outc3[5,i], outc3[6,i]])
    for i in range(40):
        outc4tot[0, i] = min([outc4[0,i], outc4[1,i], outc4[2,i], outc4[3,i], outc4[4,i], outc4[5,i], outc4[6,i]])

    plot_settings()
    color_ba = '0.5'
    # fig, (ax3, ax, ax2) = plt.subplots(3, 1, sharex='col', gridspec_kw={'hspace': 0, 'wspace': 0})
    fig, ax3 = plt.subplots()

    # ax.plot(x_list, outc2tot[0, :], color=color_ba, label='Best protocols found', linestyle='--')
    # ax2.plot(x_list, outc2tot[0, :], color=color_ba, label='Best protocols found', linestyle='--')
    # ax3.plot(x_list, outc2tot[0, :], color=color_ba, label='Best protocols found', linestyle='--')
    # ax.plot(x_list, outc2[0, :], color='k', label='Diff. criteria, b=2')
    # ax.plot(x_list, outc2[1, :], color='k', label='Diff. criteria, w/ rot.', linestyle=':', linewidth=0.8)
    ax3.plot(x_list, outc2[2, :], color='k', label=r'Random program')
    ax3.plot(x_list, outc2[4, :], color='k', label=r'Base program, $b=1$', linestyle=':', linewidth=0.8)
    ax3.plot(x_list, outc2[6, :], color='0.7', label=r'Base program, $b=2$')
    ax3.plot(x_list, outc2[5, :], color='0.5', linestyle='--', label=r'Base program, $b=3$')
    # ax3.plot(x_list, outc2[3, :], color='k', linestyle=':', label='b=4')


    # ax.plot(x_list[1:42], outc3tot[0, :], color=color_ba, linestyle='--')
    # ax2.plot(x_list[1:42], outc3tot[0, :], color=color_ba, linestyle='--')
    # ax3.plot(x_list[1:42], outc3tot[0, :], color=color_ba, linestyle='--')
    # ax.plot(x_list[1:42], outc3[0, :], color='k')
    # ax.plot(x_list[1:42], outc3[1, :], color='k', linestyle=':', linewidth=0.8)
    ax3.plot(x_list[1:42], outc3[2, :], color='k')
    ax3.plot(x_list[1:42], outc3[4, :], color='k', linestyle=':', linewidth=0.8)
    ax3.plot(x_list[1:42], outc3[6, :], color='0.7')
    ax3.plot(x_list[1:42], outc3[5, :], color='0.5', linestyle='--')
    # ax3.plot(x_list[1:42], outc3[3, :], color='k', linestyle=':')

    # ax.plot(x_list[2:42], outc4tot[0, :], color=color_ba, linestyle='--')
    # ax2.plot(x_list[2:42], outc4tot[0, :], color=color_ba, linestyle='--')
    # ax3.plot(x_list[2:42], outc4tot[0, :], color=color_ba, linestyle='--')
    # ax.plot(x_list[2:42], outc4[0, :], color='k')
    # ax.plot(x_list[2:42], outc4[1, :], color='k', linestyle=':', linewidth=0.8)
    ax3.plot(x_list[2:42], outc4[2, :], color='k')
    ax3.plot(x_list[2:42], outc4[4, :], color='k', linestyle=':', linewidth=0.8)
    ax3.plot(x_list[2:42], outc4[6, :], color='0.7')
    ax3.plot(x_list[2:42], outc4[5, :], color='0.5', linestyle='--')
    # ax3.plot(x_list[2:42], outc4[3, :], color='k', linestyle=':')

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax3.set(xlabel=r'Number of isotropic Bell pairs used ($F_\mathrm{Bell}=0.9$)',
            ylabel=r'Infidelity $1-F_\mathrm{GHZ}$ of the final GHZ state')

    # plt.xlabel(r'Number of Bell pairs used ($F_\mathrm{Bell}=0.9$)')
    # plt.ylabel(r'$1-F_\mathrm{GHZ}$ of the final GHZ state')
    # plt.sca(ax)
    # plt.legend(loc='upper right', ncol=1, handleheight=1.2, labelspacing=0.03)
    # plt.sca(ax2)
    # plt.legend(loc='upper right', ncol=1, handleheight=1.2, labelspacing=0.03)
    plt.sca(ax3)
    plt.legend(loc='upper right', ncol=1, handleheight=1.2, labelspacing=0.03)

    # ax.tick_params(direction='inout')
    # ax2.tick_params(direction='inout')
    ax3.tick_params(direction='inout')

    # plt.sca(ax)
    # plt.ylim(0.00000001, 1.5)
    # plt.sca(ax2)
    # plt.ylim(0.00000001, 1.5)
    plt.sca(ax3)
    plt.xlim(0, 47)

    ax3.text(42.5, 1e-4, r'$n=4$')
    ax3.text(42.5, 1e-5, r'$n=3$')
    ax3.text(42.5, 1e-7, r'$n=2$')

    # plt.sca(ax)
    # plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])#np.arange(min(x_list), max(x_list), 6.0))
    # plt.sca(ax)
    # plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])
    plt.sca(ax3)
    plt.xticks([1, 2, 3, 8, 15, 22, 29, 36, 42])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.grid()

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(), numticks=12)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax2.yaxis.set_minor_locator(locmin)
    # ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.yaxis.set_minor_locator(locmin)
    ax3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # plt.sca(ax)
    # plt.grid(True, which="both")
    # plt.sca(ax2)
    # plt.grid(True, which="both")
    plt.sca(ax3)
    plt.grid(True, which="both")
    fig.set_size_inches(cm2inch(8.5, 6.5))
    plt.savefig('figures/plot_comparison_programs.pdf', pad_inches=0, bbox_inches='tight')

    return


def plot_infidelity_with_80_Bell_pairs_for_diff_number_of_parties():
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.
    """
    best_states = best_random_protocols()
    IF_list = np.empty(7, dtype=float)  # infidelity
    for i in range(7):
        IF_list[i] = 1 - best_states[i + 2][80][0]
    print(IF_list)
    n_list = [2, 3, 4, 5, 6, 7, 8]

    plot_settings()
    fig, ax = plt.subplots()
    ax.plot(n_list, IF_list, marker='.', color='k')
    # legend = ax.legend(loc='center left', ncol=1, handleheight=1.2, labelspacing=0.03, bbox_to_anchor=(1, 0.5))

    ax.set_yscale('log')

    ax.set(xlabel=r'Number of parties $n$ of final GHZ state',
           ylabel=r'$1-F_\mathrm{GHZ}$ of best protocol with 80 pairs')
    ax.tick_params(direction='inout')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    fig.set_size_inches(cm2inch(8.5, 6.5))
    plt.savefig('figures/plot_infidelity_with_80_Bell_pairs_for_diff_number_of_parties.pdf', pad_inches=0, bbox_inches='tight')
    return


def plot_Bell_pairs_needed_for_diff_number_of_parties():
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.
    """
    best_states = best_random_protocols()
    k_list = np.empty(7, dtype=int)
    th = best_states[8][80][0]
    print(th)
    k_list[6] = 80
    for n in range(2, 8):
        flag = 0
        for k in range(n - 1, 80):
            if best_states[n][k][0] > th:
                k_list[n - 2] = k
                break
    n_list = [2, 3, 4, 5, 6, 7, 8]

    plot_settings()
    fig, ax = plt.subplots()
    ax.plot(n_list, k_list, marker='.', color='k')
    # legend = ax.legend(loc='center left', ncol=1, handleheight=1.2, labelspacing=0.03, bbox_to_anchor=(1, 0.5))
    ax.set(xlabel=r'Number of parties $n$ of final GHZ state',
           ylabel=r'$k$ needed to reach target fidelity $F_\mathrm{GHZ}$')
    ax.tick_params(direction='inout')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    fig.set_size_inches(cm2inch(8.5, 6.5))
    plt.savefig('figures/plot_Bell_pairs_needed_for_diff_number_of_parties.pdf', pad_inches=0, bbox_inches='tight')
    return


def export_protocols():
    # # Argument list new dynamic algorithm:
    # # n_max, k_max, F, da_type='mpc', nstate=2, inc_rot=0, show_or_not=0, seed=10, T=0.0009
    data_random_410 = das.dynamic_algorithm(8, 80, 0.9, 'random', 200, 0, 1, 7876, 0.00007)
    data_random_422 = das.dynamic_algorithm(8, 80, 0.9, 'random', 200, 0, 1, 273, 0.0004)
    data_random_442 = das.dynamic_algorithm(8, 80, 0.9, 'random', 200, 0, 1, 2100, 0.0005)
    data_random_414 = das.dynamic_algorithm(8, 80, 0.9, 'random', 200, 0, 1, 9343, 0.0003)

    # protocolsfrom410 = dict()
    # protocolsfrom422 = dict()
    # protocolsfrom442 = dict()
    # protocolsfrom414 = dict()
    # for i in range(7):
    #     n = i + 2
    #     for j in range(80):
    #         k = j + 1
    #         if k > n - 2:
    #             for t in range(200):
    #                 protocolsfrom410[(n, k, t)] = dap.protocol_add_id_nrs(dap.identify_protocol(data_random_410, n, k, t))
    #                 protocolsfrom422[(n, k, t)] = dap.protocol_add_id_nrs(dap.identify_protocol(data_random_422, n, k, t))
    #                 protocolsfrom442[(n, k, t)] = dap.protocol_add_id_nrs(dap.identify_protocol(data_random_442, n, k, t))
    #                 protocolsfrom414[(n, k, t)] = dap.protocol_add_id_nrs(dap.identify_protocol(data_random_414, n, k, t))

    # pickle.dump(protocolsfrom410, open("protocols/prots_with_best_prot_at_4_10.p", "wb"))
    # pickle.dump(protocolsfrom422, open("protocols/prots_with_best_prot_at_4_22.p", "wb"))
    # pickle.dump(protocolsfrom442, open("protocols/prots_with_best_prot_at_4_42.p", "wb"))
    # pickle.dump(protocolsfrom414, open("protocols/prots_with_best_prot_at_4_14.p", "wb"))

    pickle.dump(dap.protocol_add_id_nrs(dap.identify_protocol(data_random_410, 4, 10, 0)), open("protocols/best_prot_at_4_10.p", "wb"))
    pickle.dump(dap.protocol_add_id_nrs(dap.identify_protocol(data_random_422, 4, 22, 0)), open("protocols/best_prot_at_4_22.p", "wb"))
    pickle.dump(dap.protocol_add_id_nrs(dap.identify_protocol(data_random_442, 4, 42, 0)), open("protocols/best_prot_at_4_42.p", "wb"))
    pickle.dump(dap.protocol_add_id_nrs(dap.identify_protocol(data_random_414, 4, 14, 0)), open("protocols/best_prot_at_4_14.p", "wb"))
    return


def plot_comparison_best_prots_plus_averages():
    """
    Function that creates one of the plots in the paper. Can use a pre-calculated file with data: if it is not there it
    will be created for future use of the function.
    """
    F_min = 0.8
    F_max = 1
    F = np.arange(F_min, F_max, (F_max - F_min) / 100)
    outc = np.zeros([17, 100])

    if not(os.path.isfile("protocols/best_prot_at_4_10.p") and os.path.isfile("protocols/best_prot_at_4_14.p")
           and os.path.isfile("protocols/best_prot_at_4_22.p") and os.path.isfile("protocols/best_prot_at_4_42.p")):
        export_protocols()
    # protocol410 = pickle.load(open("protocols/best_prot_at_4_10.p", "rb"))
    protocol414 = pickle.load(open("protocols/best_prot_at_4_14.p", "rb"))
    protocol422 = pickle.load(open("protocols/best_prot_at_4_22.p", "rb"))
    protocol442 = pickle.load(open("protocols/best_prot_at_4_42.p", "rb"))

    # if not(os.path.isfile('calc_data/mpc_4_42_intF_0.8_1_100_2_0.txt')):
    #     das.store_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 0)
    # msda_no_rot = das.import_data_intF(4, 42, 'mpc', 0.8, 1, 100, 2, 0)

    for i in range(100):
        outc[0, i] = 1 - kp.Expedient(F[i])[0]
        outc[1, i] = 1 - kp.Stringent(F[i])[0]
        # outc[2, i] = 1 - dap.operate_protocol(protocol410, 200, F[i])[0]
        outc[3, i] = 1 - dap.operate_protocol(protocol422, 200, F[i])[0]
        outc[4, i] = 1 - dap.operate_protocol(protocol442, 200, F[i])[0]
        # outc[5, i] = 1 - kp.Minimize4X_40(F[i])[0]
        # outc[6, i] = 1 - kp.Minimize4X_22(F[i])[0]
        outc[16, i] = 1 - dap.operate_protocol(protocol414, 200, F[i])[0]

    # for j in range(4):
    #     k = j + 39
    #     i_spike_no_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 0, 0.9)
    #     for i in range(100):
    #         F_no_rot = dap.operate_protocol(dap.identify_protocol(msda_no_rot[i_spike_no_rot], 4, k, 0), 2, F[i])[0]
    #         outc[7+j, i] = 1 - F_no_rot
    # for i in range(100):
    #     outc[11, i] = min(outc[4, i], outc[10, i])
    # for j in range(4):
    #     k = j + 12
    #     i_spike_no_rot = das.spike(4, 42, 'mpc', 0.8, 1, 100, 2, 0, 0.9)
    #     for i in range(100):
    #         F_no_rot = dap.operate_protocol(dap.identify_protocol(msda_no_rot[i_spike_no_rot], 4, k, 0), 2, F[i])[0]
    #         outc[12+j, i] = 1 - F_no_rot

    plot_settings()
    fig, ax = plt.subplots()

    nr_iter = 100000
    print_results = False
    outy = np.zeros([5, 100])
    for i in range(100):
        state414, pb414, ts414, fr414, fl414 = avg_bell_gs.protocol414(F[i])
        outy[0, i] = statistics.mean(ac.av_calc(pb414, ts414, fr414, fl414, nr_iter, print_results))
        state422, pb422, ts422, fr422, fl422 = avg_bell_gs.protocol422(F[i])
        outy[1, i] = statistics.mean(ac.av_calc(pb422, ts422, fr422, fl422, nr_iter, print_results))
        state442, pb442, ts442, fr442, fl442 = avg_bell_gs.protocol442(F[i])
        outy[2, i] = statistics.mean(ac.av_calc(pb442, ts442, fr442, fl442, nr_iter, print_results))
        stateExp, pbExp, tsExp, frExp, flExp = avg_bell_gs.Expedient(F[i])
        outy[3, i] = statistics.mean(ac.av_calc(pbExp, tsExp, frExp, flExp, nr_iter, print_results))
        stateStr, pbStr, tsStr, frStr, flStr = avg_bell_gs.Stringent(F[i])
        outy[4, i] = statistics.mean(ac.av_calc(pbStr, tsStr, frStr, flStr, nr_iter, print_results))

    color = 'tab:blue'
    color2 = 'xkcd:light blue'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Average number of Bell pair generation steps', color=color)  # we already handled the x-label with ax1
    ax2.plot(F, outy[0, :], color=color2, linestyle=':')
    ax2.plot(F, outy[1, :], color=color2)
    ax2.plot(F, outy[2, :], color=color)
    ax2.plot(F, outy[3, :], color=color2, linestyle='--')
    ax2.plot(F, outy[4, :], color=color, linestyle='--')

    ax2.tick_params(axis='y', labelcolor=color)

    # ax.plot(F, outc[2, :], label='Random algorithm (4,10)', color='0.5', linestyle='-.') #label='DA_random_880_7876_0.00007 (4, 10)')
    ax.plot(F, outc[0, :], label=r'Expedient $(4,22)$', color='0.5', linestyle='--')
    ax.plot(F, outc[16, :], label=r'Random program $(4,14)$', color='0.5', linestyle=':')
    # ax.plot(F, outc[15, :], label='Diff. criteria, w/o rot. (4,15)', color='k')
    # ax.plot(F, outc[6, :], label='Minimize4X [2] (4,22)', linestyle='--')
    # ax.plot(F, outc[3, :], label='Random algorithm (4,22)', color='0.5', linestyle='-') #label='DA_random_880_273_0.0004 (4, 22)')
    ax.plot(F, outc[3, :], label=r'Random program $(4,22)$', color='0.5')  # label='DA_random_880_273_0.0004 (4, 22)')
    ax.plot(F, outc[1, :], label=r'Stringent $(4,42)$', color='k', linestyle='--')
    # ax.plot(F, outc[5, :], label='Minimize4X [2] (4,40)')
    ax.plot(F, outc[4, :], label=r'Random program $(4,42)$', color='k') #label='DA_random_880_2100_0.0005 (4, 42)')
    # ax.plot(F, outc[7, :], label='Diff. criteria, w/o rot. (4,39)')
    # ax.plot(F, outc[8, :], label='Diff. criteria, w/o rot. (4,40)')
    # ax.plot(F, outc[9, :], label='Diff. criteria, w/o rot. (4,41)')
    # ax.plot(F, outc[10, :], label='Diff. criteria, w/o rot. (4,42)', color='k')
    # ax.plot(F, outc[11, :], label='Smallest between diff. crit. w/o rot. and dynamic algorithm (4,42)', color='k')
    # ax.plot(F, outc[12, :], label='Diff. criteria, w/o rot. (4,12)', color='k')
    # ax.plot(F, outc[13, :], label='Diff. criteria, w/o rot. (4,13)', color='k')
    # ax.plot(F, outc[14, :], label='Diff. criteria, w/o rot. (4,14)', color='0.5')

    ax.set_zorder(1)  # default zorder is 0 for ax and ax2
    ax.patch.set_visible(False)  # prevents ax from hiding ax2

    ax.legend(loc='lower left', ncol=1, handleheight=1.2, labelspacing=0.03, fancybox=True, bbox_to_anchor=(0, 1)) #, bbox_to_anchor=(1, 0.5))
    ax.set_yscale('log')
    ax.set(xlabel=r'Fidelity $F_\mathrm{Bell}$ of isotropic Bell pairs used',
           ylabel=r'$1-F_\mathrm{GHZ}$ of final 4-qubit GHZ state')
    # plt.xticks(range(N))
    ax.tick_params(direction='inout')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.yticks([1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])  # np.arange(min(x_list), max(x_list), 6.0))
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # plt.grid(True, which="both")
    plt.xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    # ax2.set_yticks([0, 20, 40, 60, 80, 100, 120])
    # plt.sca(ax2)
    # plt.ylim(0, 120)
    ax.grid()
    fig.set_size_inches(cm2inch(8.5, 8))
    plt.savefig('figures/plot_comparison_best_prots_plus_averages.pdf', pad_inches=0, bbox_inches='tight')

    return
