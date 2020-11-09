"""
2020 Runsheng Ouyang, Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
"""
import sys
import numpy as np
import operations as op
import ancilla_rotations as ar
import da_protocols as dap
from math import floor
import time
import pickle
import random
import math
import matplotlib.pyplot as plt
import multiprocessing
import os

plt.rcParams["font.family"] = "Calibri"
plt.rcParams["font.style"] = "normal"
# plt.rcParams["font.weight"] = "100"
plt.rcParams["font.stretch"] = "normal"
plt.rcParams["font.size"] = 11
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["axes.linewidth"] = 0.4
plt.rcParams["grid.linewidth"] = 0.4
plt.rcParams.update({'figure.autolayout': True})


class Path:
    """
    A class used to capture the structure of the dynamic algorithm that store multiple states per value of n and k;
    for each combination [n][k] (where n is the number of parties of the n-qubit GHZ diagonal state and k is the number
    of isotropic Bell diagonal states used) a new object of this class is created.

    ...

    Attributes
    ----------
    p_or_f : 0, 1 or -1
        0 for purification, 1 for fusion, and -1 if (n, k) = (2, 1), which is an elementary isotropic Bell
        diagonal state
    z_or_xy : 0, 1 or -1
        0 for z purification, 1 for xy purification, -1 for fusion or (n, k) = (2, 1)
    n : positive integer
        number of parties over which the n-qubit GHZ diagonal state is created
    k : positive integer
        number of Bell diagonal states used to create  the n-qubit GHZ diagonal state
    n2 : positive integer, or -1
        number of the previous n (size of the ancillary state), with n2 = -1 if (n, k) = (2, 1)
    k2 : positive integer, or -1
        number of the previous k (number of states used to create the ancillary state), with k2 = -1 if (n, k) = (2, 1)
    dec : integer, -1 or bigger
        describes the type of stablizer used in purification, with dec = -1 if fusion or if (n, k) = (2, 1)
    i : integer, -1 or bigger
        describes which qubit of the first state is used in the fusion process, with i = -1 if purification or if
        (n, k) = (2, 1)
    j : integer, -1 or bigger
        describes which qubit of the second state is used in the fusion process, with j = -1 if purification or if
        (n, k) = (2, 1)
    state : one-dimensional numpy-vector of length 2**n
        the best n-qubit GHZ diagonal state found for these values of n and k
    t1 : non-negative integer
        describes the memory position of the first state that is used to create the current state (at data[n1][k1][t1])
    t2 : non-negative integer
        describes the memory position of the second state that is used to create the current state (at data[n2][k2][t2])
    r1 : non-negative integer
        describes the ancillary state rotation applied to the first state at data[n1][k1][t1]
    r2 : non-negative integer
        describes the ancillary state rotation applied to the second state at data[n2][k2][t2]

    Methods
    -------
    created(self, n, k)
        Prints status information about the dynamic program in case this is requested by the user
    """
    def __init__(self, p_or_f, z_or_xy, n, k, t, n2, k2, dec, i, j, state, t1, t2, r1, r2):
        self.p_or_f = p_or_f
        self.z_or_xy = z_or_xy
        self.n = n
        self.k = k
        self.t = t
        self.n2 = n2
        self.k2 = k2
        self.dec = dec
        self.i = i
        self.j = j
        self.state = state
        self.t1 = t1
        self.t2 = t2
        self.r1 = r1
        self.r2 = r2

    # noinspection PyMethodMayBeStatic
    def created(self, n, k):
        print('(', n, ',', k, ')')


def compare_F(A, B):
    """
    This function compares the fidelity between GHZ diagonal states A and B

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has higher fidelity, 0 means A and B have equal F, -1 means B has higher F
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension")
    if A[0] > B[0]:
        best = 1
    elif A[0] == B[0]:
        best = 0
    elif A[0] < B[0]:
        best = -1
    return best


def compare_min(A, B):
    """
    This function compares which of the two GHZ diagonal states A and B has the smallest value for the smallest
    coefficient

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has the smallest value for the smallest coefficient, 0 means A and B have equal F, -1 means B has
        the smallest values for the smallest coefficient
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension")
    if min(A) < min(B):
        best = 1
    elif min(A) == min(B):
        best = 0
    elif min(A) > min(B):
        best = -1
    return best


def compare_gap(A, B):
    """
    This function compares which of the two GHZ diagonal states A and B has the biggest gap between the biggest
    coefficient and the smallest coefficient

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has the biggest coefficient gap, 0 means A and B have equal F, -1 means B has higher gap
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension")
    if (max(A) - min(A)) > (max(B) - min(B)):
        best = 1
    elif (max(A) - min(A)) == (max(B) - min(B)):
        best = 0
    elif (max(A) - min(A)) < (max(B) - min(B)):
        best = -1
    return best


def compare_entropy(A, B):
    """
    This function compares which of the two GHZ diagonal states A and B has the lowest entropy

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has the lowest entropy, 0 means A and B have equal F, -1 means B has the lowest entropy
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension")
    S_A = 0
    S_B = 0
    for i in range(np.size(A)):
        if A[i] != 0:
            S_A = S_A - A[i] * np.log2(A[i])
        if B[i] != 0:
            S_B = S_B - B[i] * np.log2(B[i])
    if S_A < S_B:
        best = 1
    elif S_A == S_B:
        best = 0
    elif S_A > S_B:
        best = -1
    return best


def compare_gap_coeff_sum(A, B):
    """
    This function compares which of the two GHZ diagonal states A and B has the biggest gap between the biggest
    coefficient and the smallest coefficient sum configuration.

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has the biggest gap, 0 means A and B have equal F, -1 means B has the biggest gap
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension.")
    if (max(A) - ar.smallest_coefficient_sum(A)) > (max(B) - ar.smallest_coefficient_sum(B)):
        best = 1
    elif (max(A) - ar.smallest_coefficient_sum(A)) == (max(B) - ar.smallest_coefficient_sum(B)):
        best = 0
    elif (max(A) - ar.smallest_coefficient_sum(A)) < (max(B) - ar.smallest_coefficient_sum(B)):
        best = -1
    return best


def compare_min_coeff_sum(A, B):
    """
    This function compares which of the two GHZ diagonal states A and B has the smallest value for the coefficient
    sum

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B

    Returns
    -------
    best : int
        +1 means A has the smallest coefficient sum, 0 means A and B have equal F, -1 means B has the smallest
        coefficient sum
    """
    if np.size(A) != np.size(B):
        sys.exit("The input states in compare function should have the same dimension.")
    if ar.smallest_coefficient_sum(A) < ar.smallest_coefficient_sum(B):
        best = 1
    elif ar.smallest_coefficient_sum(A) == ar.smallest_coefficient_sum(B):
        best = 0
    elif ar.smallest_coefficient_sum(A) > ar.smallest_coefficient_sum(B):
        best = -1
    return best


def compare(A, B, t):
    """
    This function compares which of the two GHZ diagonal states A and B has the best statistics based on a certain
    comparison determined by the input parameter t.

    Parameters
    ----------
    A : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as B
        GHZ diagonal state  A
    B : one-dimensional numpy object containing the coefficients of a GHZ diagonal state of the same size as A
        GHZ diagonal state  B
    t : integer in range [0,4]
        determines which is of the comparisons is used

    Returns
    -------
    best : int
        +1 means A has the best stats, 0 means A and B have equal F, -1 means B has the best stats
    """
    if t == 0:
        best = compare_F(A, B)
    elif t == 2:
        best = compare_min(A, B)
    elif t == 1:
        best = compare_min_coeff_sum(A, B)
    elif t == 3:
        best = compare_gap_coeff_sum(A, B)
    elif t == 4:
        best = compare_entropy(A, B)
    return best


def create_storage(n_max, k_max, ntype):
    """
    This function creates a matrix of size (n_max+1, k_max+1, ntype), and fills it on places that are feasible elements
    in the dynamic program (e.g., n=4 and k=2 is infeasible, because it is impossible to make an 4-qubit GHZ state with
    only 2 Bell pairs).

    Parameters
    ----------
    n_max : positive integer
        Maximum number of qubits of the GHZ diagonal state that will be created
    k_max : positive integer
        Maximum number of isotropic Bell diagonal states used to create and distill the final GHZ diagonal state
    ntype : positive integer
        Parameter that determines how many protocols are stored per value of n and k (each of them based on the a
        different comparison; the first ntype comparisons of the compare function above)

    Returns
    -------
    data : (n_max+1, k_max+1, ntype) matrix with objects of class Path
        Each element of this matrix is used to store information about its element itself and how it's made
    """
    data = np.empty((n_max + 1, k_max + 1, ntype), dtype=object)
    for n in range(n_max + 1):
        for k in range(k_max + 1):
            for t in range(ntype):
                if (n == 0) | (n == 1) | (k < n - 1):
                    data[n][k][t] = None
                else:
                    data[n][k][t] = Path(p_or_f=-1, z_or_xy=-1, n=n, k=k, t=t, n2=-1, k2=-1, dec=-1, i=-1, j=-1,
                                         state=op.set_isotropic_state(1 / 2 ** n, n), t1=-1, t2=-1, r1=-1, r2=-1)
    return data


def update_m_prot(data, n, k, F, show_or_not, nstate, inc_rot, da_type='sp'):
    """
    Function that updates data[n][k][t] in the deterministic version of the algorithm.

    Parameters
    ----------
    data : matrix with objects of class Path
    n : positive integer smaller than or equal to n_max
        Number of parties for which we want to update the element in data
    k : positive integer smaller than or equal to k_max
        Number of Bell diagonal states for which we want to update the element in data
    n_max : positive integer
        Maximum number of qubits of the GHZ diagonal state stored in data
    k_max : positive integer
        Maximum number of isotropic Bell diagonal states stored in data
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states used
    show_or_not : Boolean
        Determines if status update about a matrix entry being update is printed in the console
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    da_type : string
        String describing what version of the deterministic random the algorithm should carry out: 'sp' is a single
        protocol per value of n and k, 'mpc' is multiple protocols per value of n and k based on different conditions,
        'mpF' is multiple protocols per n and k based on the highest fidelity)

    Returns
    -------
    data : (n_max+1, k_max+1, nstate) matrix with objects of class Path
        Each element (n, k, t) of this matrix is used to store information about (n, k, t) itself and how it's made
    """
    if da_type == 'sp':
        nstate = 1
        cp = 1
    elif da_type == 'mpc':
        cp = 1
    elif da_type == 'mpF':
        cp = 0  # This parameter makes sure for 'mpF', states are only compared based on fidelity
    else:
        sys.exit("Dynamic algorithm type unknown.")

    if n == 2 and k == 1:
        state = op.set_isotropic_state(F, 2)
        for t in range(nstate):
            data[2][1][t] = Path(p_or_f=-1, z_or_xy=-1, n=2, k=1, t=t, n2=-1, k2=-1, dec=-1, i=-1, j=-1,
                                 state=state, t1=-1, t2=-1, r1=-1, r2=-1)
    else:
        # if n == 4 and k == 42:
        #     inc_rot = 1
        # else:
        #     inc_rot = 0

        # Create a buffer for the states that will be found
        buffer = np.empty(nstate, dtype=object)
        for t in range(nstate):
            buffer[t] = data[n][k][t]

        # Try purification:
        for dec in range(1, 2 ** n):
            if dec < 2 ** (n - 1):  # Z_purification
                n2 = np.size(op.transform_list_ind(dec, n, 'Z'))
            else:  # XY_purification
                n2 = n
            for k2 in range(n2 - 1, k - n + 1 + 1):
                for a in range(nstate):
                    for b in range(nstate):
                        for r in range((2 ** (n2 - 1) + floor(2 / n2) * 2) ** inc_rot):
                            to_compare = op.purification(data[n][k - k2][a].state,
                                                         ar.ancilla_rotation(data[n2][k2][b].state, r),
                                                         dec)
                            # print('')
                            # print(to_compare[0])
                            for t in range(nstate):
                                if compare(to_compare, buffer[nstate - 1].state, 0) == -1 and da_type == 'mpF':
                                    break   # Break if the new state is not better than the last one
                                if compare(to_compare, buffer[t].state, t*cp) == 1 or \
                                        (compare(to_compare, buffer[t].state, t*cp) == 0 and a + b == 0):
                                    # print('')
                                    # print('n, k =', n, k, 'k - k2, a, k2, b, dec, r =', k - k2, a, k2, b, dec, r)
                                    # print('buffer[', t, '].state =', buffer[t].state, ", to_compare =", to_compare)
                                    # print('compare =', compare(to_compare, buffer[t].state, t*cp))
                                    if da_type == 'mpF':                    # Move up the existing states to the right,
                                        for v in range(nstate - t - 1):     # from the last to the number t+1
                                            u = nstate - 1 - v
                                            buffer[u].p_or_f = buffer[u - 1].p_or_f
                                            buffer[u].z_or_xy = buffer[u - 1].z_or_xy
                                            buffer[u].n = buffer[u - 1].n
                                            buffer[u].k = buffer[u - 1].k
                                            buffer[u].t = u
                                            buffer[u].n2 = buffer[u - 1].n2
                                            buffer[u].k2 = buffer[u - 1].k2
                                            buffer[u].dec = buffer[u - 1].dec
                                            buffer[u].i = buffer[u - 1].i
                                            buffer[u].j = buffer[u - 1].j
                                            buffer[u].state = buffer[u - 1].state
                                            buffer[u].t1 = buffer[u - 1].t1
                                            buffer[u].t2 = buffer[u - 1].t2
                                            buffer[u].r1 = buffer[u - 1].r1
                                            buffer[u].r2 = buffer[u - 1].r2
                                            # print('buffer[ u =', u, '].state =', buffer[u].state)
                                    buffer[t].p_or_f = 0
                                    buffer[t].z_or_xy = dec >> (n - 1)
                                    buffer[t].n = n
                                    buffer[t].k = k
                                    buffer[t].t = t
                                    buffer[t].n2 = n2
                                    buffer[t].k2 = k2
                                    buffer[t].dec = dec
                                    buffer[t].i = -1
                                    buffer[t].j = -1
                                    buffer[t].state = to_compare
                                    buffer[t].t1 = a
                                    buffer[t].t2 = b
                                    buffer[t].r1 = -1
                                    buffer[t].r2 = r
                                    # print('buffer[ t =', t, '].state[0] =', buffer[t].state[0])
                                    if da_type == 'mpF':    # Breaks the "for t in range(nstate)" loop at the point
                                        break               # where the first better state is found

        # Try fusion:
        for n2 in range(2, n - 1 + 1):
            for k2 in range(n2 - 1, k - n + n2 + 1):
                n1 = n + 1 - n2
                for i in range(n1):
                    for j in range(n2):
                        for a in range(nstate):
                            for b in range(nstate):
                                for r1 in range((2 ** (n1 - 1) + floor(2 / n1) * 2) ** inc_rot):
                                    for r2 in range((2 ** (n2 - 1) + floor(2 / n2) * 2) ** inc_rot):
                                        to_compare = op.fuse_GHZ_local(
                                            ar.ancilla_rotation(data[n1][k - k2][a].state, r1),
                                            ar.ancilla_rotation(data[n2][k2][b].state, r2),
                                            i, j)
                                        for t in range(nstate):
                                            if compare(to_compare, buffer[nstate-1].state, 0) == -1 and da_type == 'mpF':
                                                break
                                            if compare(to_compare, buffer[t].state, t*cp) == 1 or \
                                                    (compare(to_compare, buffer[t].state, t*cp) != -1
                                                     and a + b == 0):
                                                if da_type == 'mpF':         # Move up the existing states to the right,
                                                    for v in range(nstate - t - 1):  # from the last to the number t+1
                                                        u = nstate - 1 - v
                                                        buffer[u].p_or_f = buffer[u - 1].p_or_f
                                                        buffer[u].z_or_xy = buffer[u - 1].z_or_xy
                                                        buffer[u].n = buffer[u - 1].n
                                                        buffer[u].k = buffer[u - 1].k
                                                        buffer[u].t = u
                                                        buffer[u].n2 = buffer[u - 1].n2
                                                        buffer[u].k2 = buffer[u - 1].k2
                                                        buffer[u].dec = buffer[u - 1].dec
                                                        buffer[u].i = buffer[u - 1].i
                                                        buffer[u].j = buffer[u - 1].j
                                                        buffer[u].state = buffer[u - 1].state
                                                        buffer[u].t1 = buffer[u - 1].t1
                                                        buffer[u].t2 = buffer[u - 1].t2
                                                        buffer[u].r1 = buffer[u - 1].r1
                                                        buffer[u].r2 = buffer[u - 1].r2
                                                buffer[t].p_or_f = 1
                                                buffer[t].z_or_xy = -1
                                                buffer[t].n = n
                                                buffer[t].k = k
                                                buffer[t].t = t
                                                buffer[t].n2 = n2
                                                buffer[t].k2 = k2
                                                buffer[t].dec = -1
                                                buffer[t].i = i
                                                buffer[t].j = j
                                                buffer[t].state = to_compare
                                                buffer[t].t1 = a
                                                buffer[t].t2 = b
                                                buffer[t].r1 = r1
                                                buffer[t].r2 = r2
                                                if da_type == 'mpF':   # Breaks the "for t in range(nstate)" loop at
                                                    break              # the point where the first better state is found
        for t in range(nstate):
            data[n][k][t] = buffer[t]
            # if k == 6:
            #     print('buffer[', t, ' ].t2 = ', buffer[t].t2)
    if show_or_not == 1:
        Path.created(data[n][k], n, k)
    return data


def update_random(data, n, k, F, show_or_not, nstate, inc_rot, T):
    """
    Function that updates data[n][k][t] in the situation where we want a randomized version of the dynamic algorithm.

    Parameters
    ----------
    data : matrix with objects of class Path
    n : positive integer smaller than or equal to n_max
        Number of parties for which we want to update the element in data
    k : positive integer smaller than or equal to k_max
        Number of Bell diagonal states for which we want to update the element in data
    n_max : positive integer
        Maximum number of qubits of the GHZ diagonal state stored in data
    k_max : positive integer
        Maximum number of isotropic Bell diagonal states stored in data
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states used
    show_or_not : Boolean
        Determines if status update about a matrix entry being update is printed in the console
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    T : float
        temperature in exp**(delta_F/T). A higher T will allow this function accept more states with low fidelity

    Returns
    -------
    data : (n_max+1, k_max+1, nstate) matrix with objects of class Path
        Each element (n, k, t) of this matrix is used to store information about (n, k, t) itself and how it's made
    """

    if n == 2 and k == 1:   # At the (n, k) = (2, 1) spots in the memory we add isotropic Bell pairs
        state = op.set_isotropic_state(F, 2)
        for t in range(nstate):
            data[2][1][t] = Path(p_or_f=-1, z_or_xy=-1, n=2, k=1, t=t, n2=-1, k2=-1, dec=-1, i=-1, j=-1,
                                 state=state, t1=-1, t2=-1, r1=-1, r2=-1)
    else:
        # Create a buffer for the states that will be found
        buffer = np.empty(nstate, dtype=object)
        for t in range(nstate):
            buffer[t] = data[n][k][t]

        # Create first state to compare with:
        prev_state = op.set_isotropic_state(1 / 2 ** n, n)

        for t in range(nstate):
            if n == 2:  # In this situation we can only do purification
                p_or_f = 0
            elif k == n - 1:    # In this situation we can only do fusion
                p_or_f = 1
            else:
                p_or_f = random.randint(0, 1)

            if p_or_f == 0:     # Purification
                n2 = 0
                dec = 0
                while (n2 == 0) | ((n2 - 1) > (k - n + 1)):  # to satisfy randint input range
                    dec = random.randint(1, 2 ** n - 1)
                    if dec < 2 ** (n - 1):  # Z_purification
                        n2 = np.size(op.transform_list_ind(dec, n, 'Z'))
                    else:  # XY_purification
                        n2 = n
                z_or_xy = dec >> (n - 1)
                k2 = random.randint(n2 - 1, k - n + 1)
                i = -1
                j = -1
                a = random.randint(0, nstate - 1)
                b = random.randint(0, nstate - 1)
                r1 = -1
                r2 = random.randint(0, (2 ** (n2 - 1) + floor(2 / n2) * 2) ** inc_rot - 1)
                to_compare = op.purification(data[n][k - k2][a].state,
                                             ar.ancilla_rotation(data[n2][k2][b].state, r2),
                                             dec)
            else:   # Fusion
                z_or_xy = -1
                dec = -1
                n2 = random.randint(2, n - 1)
                n1 = n + 1 - n2
                if k == n - 1:
                    k2 = n2 - 1
                else:
                    k2 = random.randint(n2 - 1, k - n + n2)
                i = random.randint(0, n1 - 1)
                j = random.randint(0, n2 - 1)
                a = random.randint(0, nstate - 1)
                b = random.randint(0, nstate - 1)
                r1 = random.randint(0, (2 ** (n1 - 1) + floor(2 / n1) * 2) ** inc_rot - 1)
                r2 = random.randint(0, (2 ** (n2 - 1) + floor(2 / n2) * 2) ** inc_rot - 1)
                to_compare = op.fuse_GHZ_local(ar.ancilla_rotation(data[n1][k - k2][a].state, r1),
                                               ar.ancilla_rotation(data[n2][k2][b].state, r2),
                                               i, j)

            if compare(to_compare, prev_state, 0) == -1:    # Triggers if the previous state was better
                rate = math.exp((to_compare[0] - prev_state[0]) / T)  # Accept rate
                psi = random.uniform(0, 1)  # Throw a dice to see if the state is accepted
                if psi < rate:
                    accept_state = 1
                else:
                    accept_state = 0
            else:   # Triggers if the new state is better or the same as the previous state
                accept_state = 1

            if accept_state == 1:  # Accept the new state in the memory
                buffer[t].p_or_f = p_or_f
                buffer[t].z_or_xy = z_or_xy
                buffer[t].n = n
                buffer[t].k = k
                buffer[t].t = t
                buffer[t].n2 = n2
                buffer[t].k2 = k2
                buffer[t].dec = dec
                buffer[t].i = i
                buffer[t].j = j
                buffer[t].state = to_compare
                buffer[t].t1 = a
                buffer[t].t2 = b
                buffer[t].r1 = r1
                buffer[t].r2 = r2
                prev_state = to_compare
            else:   # Use the previous state in the memory
                buffer[t].p_or_f = buffer[t - 1].p_or_f
                buffer[t].z_or_xy = buffer[t - 1].z_or_xy
                buffer[t].n = buffer[t - 1].n
                buffer[t].k = buffer[t - 1].k
                buffer[t].t = buffer[t - 1].t
                buffer[t].n2 = buffer[t - 1].n2
                buffer[t].k2 = buffer[t - 1].k2
                buffer[t].dec = buffer[t - 1].dec
                buffer[t].i = buffer[t - 1].i
                buffer[t].j = buffer[t - 1].j
                buffer[t].state = buffer[t - 1].state
                buffer[t].t1 = buffer[t - 1].t1
                buffer[t].t2 = buffer[t - 1].t2
                buffer[t].r1 = buffer[t - 1].r1
                buffer[t].r2 = buffer[t - 1].r2
                prev_state = prev_state

        # Sort buffer from highest fidelity to lowest
        to_sort = np.empty(nstate, dtype=object)
        for t in range(nstate):
            to_sort[t] = buffer[t].state[0]
        sorted_buf = np.argsort(to_sort)
        # np.argsort() return to a list with element from 0 to nstate-1, from smallest F to the biggest
        for t in range(nstate):
            u = nstate - 1 - t  # From biggest F to smallest F
            data[n][k][t] = buffer[sorted_buf[u]]
            data[n][k][t].t = t

    if show_or_not == 1:
        Path.created(data[n][k], n, k)
    return data


def dynamic_algorithm(n_max, k_max, F, da_type='mpc', nstate=2, inc_rot=0, show_or_not=0, seed=10, T=0.0009):
    """
    Function that carries out the dynamic algorithm and updates the object data in this process (using the update
    function).

    Parameters
    ----------
    n_max : positive integer
        Maximum number of qubits of the GHZ diagonal state stored in data
    k_max : positive integer
        Maximum number of isotropic Bell diagonal states stored in data
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states used
    da_type : string
        String describing what version of the deterministic random the algorithm should carry out: 'sp' is a single
        protocol per value of n and k, 'mpc' is multiple protocols per value of n and k based on different conditions,
        'mpF' is multiple protocols per n and k based on the highest fidelity, and 'random' is the randomized version
        of the dynamic program
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    show_or_not : Boolean
        Determines if status update about a matrix entry being update is printed in the console
    seed : integer
        Integer that determines which should be used in the randomized version of the dynamic program
    T : float
        temperature in exp**(delta_F/T). A higher T will allow this function accept more states with low fidelity

    Returns
    -------
    data : (n_max+1, k_max+1) matrix with objects of class Path
        Each element (n, k) of this matrix is used to store information about (n, k) itself and how it's made
    """
    random.seed(seed)
    data = create_storage(n_max, k_max, nstate)
    # dynamic algorithm by using update function
    for n in range(2, n_max + 1):
        for k in range(n - 1, k_max + 1):
            if da_type == 'random':
                update_random(data, n, k, F, show_or_not, nstate, inc_rot, T)
            else:
                update_m_prot(data, n, k, F, show_or_not, nstate, inc_rot, da_type)
    return data


def store_data_intF(n, k, da_type='mpc', F_min=0, F_max=1, seg=100, nstate=2, inc_rot=0, seed=10, T=0.0009):
    """
    Function that executes the dynamic program over a range of values F for the isotropic Bell diagonal states used,
    and stores the data file in a .txt file so that it can be loaded later again.

    Parameters
    ----------
    n : positive integer
        Contains the maximum number of parties for which we want to apply the dynamic program
    k : positive integer
        Contains the maximum number of isotropic Bell diagonal states for which we want to apply the dynamic program
    da_type : string
        String describing what version of the deterministic random the algorithm should carry out: 'sp' is a single
        protocol per value of n and k, 'mpc' is multiple protocols per value of n and k based on different conditions,
        'mpF' is multiple protocols per n and k based on the highest fidelity, and 'random' is the randomized version
        of the dynamic program
    F_min : float between 0 and 1, smaller than F_max
        Minimum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    F_max : float between 0 and 1
        Maximum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    seg : positive integer
        number of fidelity values used in the data structure stored in the .txt file
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    seed : integer or list of length seg
        Integer that determines which should be used in the randomized version of the dynamic program. This parameter
        can also be an array of list seg: in this situation for each fidelity in the range a different seed will be
        used.
    T : float
        temperature in exp**(delta_F/T). A higher T will allow this function accept more states with low fidelity
    """
    if (n < 2) | (k < (n - 1)):
        sys.exit('In save_data_F_new: should be not((n < 2) | (k < (n - 1)))')

    if da_type == 'random':
        if isinstance(seed, list) is False:
            filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                       + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '_' + str(seed) + '_' + str(T) \
                       + '.txt'
        elif isinstance(seed, list) is True and len(seed) == seg + 1:
            filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                       + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '_' + str(seed[0]) + '-' \
                       + str(seed[seg]) + '_' + str(T) + '.txt'
        else:
            sys.exit("In store_data_inF: Input parameter seed must of length 1 or length seg.")
    else:
        filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                   + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '.txt'

    if os.path.isfile(filename):
        print("The file", filename, "is already available in the directory.")
    else:
        dataF = np.empty((seg + 1), dtype=object)
        for i in range(seg + 1):
            current_time = time.ctime(time.time())
            F = F_min + (F_max - F_min) * i / seg
            print(i, current_time, F)
            if isinstance(seed, list) is False:
                dataF[i] = dynamic_algorithm(n, k, F, da_type, nstate, inc_rot, 0, seed, T)
            else:
                dataF[i] = dynamic_algorithm(n, k, F, da_type, nstate, inc_rot, 0, seed[i], T)
        pickle.dump(dataF, open(filename, "wb"))
    return


def import_data_intF(n, k, da_type='mpc', F_min=0, F_max=1, seg=100, nstate=2, inc_rot=0, seed=10, T=0.0009):
    if (n < 2) | (k < (n - 1)):
        sys.exit('In save_data_F_new: should be not((n < 2) | (k < (n - 1))).')

    if da_type == 'random':
        if isinstance(seed, list) is False:
            filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                       + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '_' + str(seed) + '_' + str(T) \
                       + '.txt'
        elif isinstance(seed, list) is True and len(seed) == seg + 1:
            filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                       + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '_' + str(seed[0]) + '-' \
                       + str(seed[seg]) + '_' + str(T) + '.txt'
        else:
            sys.exit("In import_data_inF: Input parameter seed must of length 1 or length seg.")
    else:
        filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_intF_' + str(F_min) + '_' + str(F_max) \
                   + '_' + str(seg) + '_' + str(nstate) + '_' + str(inc_rot) + '.txt'

    if os.path.isfile(filename):
        dataF = pickle.load(open(filename, 'rb'))
    else:
        print('The data file ' + filename + ' is not created yet. It is being created now.')
        store_data_intF(n, k, da_type, F_min, F_max, seg, nstate, inc_rot, seed, T)
        dataF = pickle.load(open(filename, 'rb'))

    return dataF


def store_data_varT(n, k, da_type='random', F=0.9, nstate=200, inc_rot=0, seed=10, T=0.0009):
    """
    Function that executes the randomized version of the dynamic program over a range of temperatures, and stores the
    data file in a .txt file so that it can be loaded later again.

    Parameters
    ----------
    n : positive integer
        Contains the maximum number of parties for which we want to apply the dynamic program
    k : positive integer
        Contains the maximum number of isotropic Bell diagonal states for which we want to apply the dynamic program
    da_type : string
        String describing what version of the deterministic random the algorithm should carry out: 'sp' is a single
        protocol per value of n and k, 'mpc' is multiple protocols per value of n and k based on different conditions,
        'mpF' is multiple protocols per n and k based on the highest fidelity, and 'random' is the randomized version
        of the dynamic program
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    seed : integer
        Integer that determines which should be used in the randomized version of the dynamic program
    T : float
        temperature in exp**(delta_F/T). A higher T will allow this function accept more states with low fidelity
    """
    if (n < 2) | (k < (n - 1)):
        sys.exit('In save_data_F_new: should be not((n < 2)|(k < (n-1)))')
    if da_type != 'random':
        sys.exit('The function import_data_varT is indended for da_type random.')

    T_list = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003,
              0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    seed_list = [3316, 7876, 570, 1264, 9343, 6959, 1162, 2100, 5177, 8559, 5454, 8917, 6232, 2994, 9603, 9296, 8193,
                 9321, 4319, 4239, 4010, 7355, 9398, 9047, 273, 9697, 6637, 8965, 2599, 5148, 6372, 5911, 3844, 17,
                 5263, 200, 4720, 787, 5339, 7157, 8184, 5289, 9342, 9304, 3409, 4122, 2967, 1789, 3048, 4734, 4831,
                 6272, 6897, 8397, 3360, 1109, 8164, 1361, 9541, 5428, 6766, 1837, 8560, 1043, 6328, 701, 1082, 3725,
                 852, 6029, 7106, 8174, 2556, 7533, 6013, 9076, 7502, 4950, 8562, 4164, 561, 6941, 1464, 4496, 4230,
                 8111, 9981, 5976, 9707, 8695, 2589, 3038, 1126, 7144, 6165, 845, 1555, 8660, 9783, 6466, 9075, 9674,
                 1526, 1222, 4328, 4231, 1820, 6602, 6091, 1714, 2421]
    nT = np.size(T_list)

    dataT = np.empty((nT, n + 1, k + 1, nstate), dtype=object)
    for seed in seed_list:
        filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_varT_' + str(F) + '_' + str(nstate) \
                   + '_' + str(inc_rot) + '_' + str(seed) + '.txt'

        if os.path.isfile(filename):
            print("The file", filename, "is already available in the directory.")

        else:
            print("-----")
            localtime = time.asctime(time.localtime(time.time()))
            print(seed, '\t', localtime)
            for num in range(nT):
                T = T_list[num]
                data_random = dynamic_algorithm(n, k, F, da_type, nstate, inc_rot, 0, seed, T)
                for n in range(2, n + 1):
                    for k in range(n - 1, k + 1):
                        for t in range(nstate):
                            dataT[num][n][k][t] = data_random[n][k][t].state[0]
                localtime = time.asctime(time.localtime(time.time()))
                print(data_random[n][k][0].state[0], '\t', localtime)
            pickle.dump(dataT, open(filename, "wb"))
            print("-----")
    return


def import_data_varT(n, k, da_type='random', F=0.9, nstate=200, inc_rot=0, seed=10, T=0.0009):
    if (n < 2) | (k < (n - 1)):
        sys.exit('In save_data_F_new: should be not((n < 2) | (k < (n - 1))).')
    if da_type != 'random':
        sys.exit('The function import_data_varT is intended for da_type random.')

    filename = 'calc_data/' + da_type + '_' + str(n) + '_' + str(k) + '_varT_' + str(F) + '_' + str(nstate) + '_' \
               + str(inc_rot) + '_' + str(seed) + '.txt'

    if os.path.isfile(filename):
        dataT = pickle.load(open(filename, 'rb'))
    else:
        sys.exit(str('The data file ' + filename + ' is not created yet.'))

    return dataT


def spike(n, k, da_type='mpc', F_min=0.8, F_max=1, seg=100, nstate=2, inc_rot=0, F=0.85, seed=10, T=0.0009):
    """
    Function that allows one to find the best protocol found in a range of protocols are the input fidelity.

    Parameters
    ----------
    filename : string
        Contains the location of the .txt file containing the data structure calculated with the dynamic program
    n : positive integer
        Contains the number of parties for which we find the best protocol
    k : positive integer
        Contains the number of isotropic Bell diagonal states for which we find the best protocol
    da_type : string
        String describing what version of the deterministic random the algorithm should carry out: 'sp' is a single
        protocol per value of n and k, 'mpc' is multiple protocols per value of n and k based on different conditions,
        'mpF' is multiple protocols per n and k based on the highest fidelity, and 'random' is the randomized version
        of the dynamic program
    F_min : float between 0 and 1, smaller than F_max
        Minimum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    F_max : float between 0 and 1
        Maximum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    seg : positive integer
        number of fidelity values used in the data structure stored in the .txt file
    nstate : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states for which we want to find the best protocol
    seed : integer
        Integer that determines which should be used in the randomized version of the dynamic program
    T : float
        temperature in exp**(delta_F/T). A higher T will allow this function accept more states with low fidelity

    Returns
    -------
    i_spike : integer
        Describes at what fidelity the protocol can be found that leads to highest output fidelity using F as input
        fidelity
    """
    dataF = import_data_intF(n, k, da_type, F_min, F_max, seg, nstate, inc_rot, seed, T)
    to_compare = 0  # F when input EPR F is 0.98
    i_spike = 0
    for i in range(seg + 1):
        for j in range(nstate):
            protocol = dap.identify_protocol(dataF[i], n, k, j)
            # print(n, k, i, j)
            tmp = dap.operate_protocol(protocol, nstate, F)[0]
            if tmp > to_compare:
                i_spike = i
                to_compare = tmp
    return i_spike


# Multiprocessing version of the algorithm
def save_dataF_mp(n, k, F_min=0, F_max=1, seg=100, ntype=3, inc_rot=0):
    """
    Function that executes the dynamic program and stores the data file in a .txt file so that it can be loaded later
    again, and uses multiprocessing to do so.

    Parameters
    ----------
    n : positive integer
        Contains the maximum number of parties for which we want to apply the dynamic program
    k : positive integer
        Contains the maximum number of isotropic Bell diagonal states for which we want to apply the dynamic program
    F_min : float between 0 and 1, smaller than F_max
        Minimum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    F_max : float between 0 and 1
        Maximum fidelity of the isotropic Bell diagonal states for which we want to apply the dynamic program
    seg : positive integer
        number of fidelity values used in the data structure stored in the .txt file
    ntype : positive integer
        number of protocols stored per value of n and k in the search process where the protocol was found
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    """
    if (n < 2) | (k < (n - 1)):
        sys.exit('In save_data_F_new: should be !((n<2)|(k<(n-1)))')

    return_dict = multiprocessing.Manager().dict()
    n_cpu = 2
    n_steps = floor(seg / n_cpu)
    for j in range(n_steps + 1):
        processes = []
        for i in range(j * n_cpu, min((j + 1) * n_cpu, seg + 1)):
            current_time = time.ctime(time.time())
            print(i, current_time)
            F = F_min + (F_max - F_min) * i / seg
            p = multiprocessing.Process(target=mp_dynamic_algorithm, args=(return_dict, i, n, k, F, ntype, inc_rot))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()

    dataF_new = np.empty((seg + 1), dtype=object)
    for i in range(seg + 1):
        dataF_new[i] = return_dict[i]

    pickle.dump(dataF_new, open('calc_data/' + 'mpc' + '_' + str(n) + '_' + str(k) + '_' + str(F_min) + '_' + str(F_max)
                                + '_' + str(seg) + '_' + str(ntype) + '_' + str(inc_rot) + '_mp.txt', "wb"))


def mp_dynamic_algorithm(return_dict, i, n, k, F, ntype, inc_rot):
    """
    Function that carries out the dynamic algorithm and updates the object data in this process (using the update
    function), catered for multiprocessing.

    Parameters
    ----------
    return_dict : global object in the multiprocessing process
    i : nonnegative integer
        number indicating the memory spot in return_dict
    n : positive integer
        Maximum number of qubits of the GHZ diagonal state stored in data
    k : positive integer
        Maximum number of isotropic Bell diagonal states stored in data
    F : float between 0 and 1
        Fidelity of the isotropic Bell diagonal states used
    ntype : positive integer
        Number of protocols stored per value of n and k in data
    inc_rot : Boolean
        0 means not ancillary permutations are included, 1 means they are included
    """
    return_dict[i] = dynamic_algorithm(n, k, F, 'mpc', ntype, inc_rot, 0)
