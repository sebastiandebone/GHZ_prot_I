"""
2020 Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
"""
import sys
import operations as op
import numpy as np
from math import ceil, log


def Stringent(F):
    """
    Function that calculates the resulting state for the Stringent protocol, in a situation without noise and
    decoherence, and with the use of 42 isotropic Bell diagonal states with fidelity F.
    """
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)

    AB3 = op.Z_purification(AB3, op.Z_purification(AB2, AB1)[0])[0]
    AB3 = op.XY_purification(AB3, op.Z_purification(AB2, AB1)[0])[0]
    AB2 = op.XY_purification(AB2, AB1)[0]
    AB2 = op.Z_purification(AB2, AB1)[0]
    AB2 = op.Z_purification(AB2, AB1)[0]
    AB3 = op.Z_purification(AB3, AB2)[0]
    AB3 = op.XY_purification(AB3, AB2)[0]

    CD3 = AB3

    AC2 = AB2
    ABCD_0 = op.fuse_GHZ_ancilla(AB3, CD3, AC2)

    BD2 = AC2
    ABCD_1 = op.Z_purification(ABCD_0, BD2, [1, 4])[0]
    ABCD_2 = op.Z_purification(ABCD_1, AC2, [2, 3])[0]
    ABCD_3 = op.Z_purification(ABCD_2, BD2, [1, 4])[0]

    return ABCD_3


def Expedient(F):
    """
    Function that calculates the resulting state for the Expedient protocol, in a situation without noise and
    decoherence, and with the use of 22 isotropic Bell diagonal states with fidelity F.
    """
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB3 = op.set_isotropic_state(F)

    AB3 = op.Z_purification(AB3, op.Z_purification(AB2, AB1)[0])[0]
    AB3 = op.XY_purification(AB3, op.Z_purification(AB2, AB1)[0])[0]

    CD3 = AB3

    AC2 = op.XY_purification(AB2, AB1)[0]
    AC2 = op.Z_purification(AC2, AB1)[0]

    ABCD_0 = op.fuse_GHZ_ancilla(AB3, CD3, AC2)

    BD2 = AC2
    ABCD_1 = op.Z_purification(ABCD_0, BD2, [1, 4])[0]
    ABCD_2 = op.Z_purification(ABCD_1, AC2, [2, 3])[0]
    ABCD_3 = op.Z_purification(ABCD_2, BD2, [1, 4])[0]

    return ABCD_3


def Minimize4X_40(F):  # This is the version that uses 40 EPR pairs
    """
    Function that calculates the resulting state for a protocol that I found for creating a 4-partite GHZ state, in a
    situation without noise and decoherence, and with the use of 40 isotropic Bell diagonal states with fidelity F.
    """
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB2 = op.XY_purification(AB2, AB1)[0]

    AC1 = op.set_isotropic_state(F)
    AC2 = op.set_isotropic_state(F)
    AC2 = op.Z_purification(AC2, AC1)[0]

    CD2 = AB2
    BD2 = AC2
    ABCD_0 = op.fuse_GHZ_ancilla(AB2, CD2, AC2)
    ABCD_1 = op.Z_purification(ABCD_0, BD2, [1, 4])[0]

    AB3 = op.set_isotropic_state(F)
    AB4 = op.set_isotropic_state(F)
    AB4 = op.XY_purification(AB4, AB3)[0]
    AB4 = op.Z_purification(AB4, AB3)[0]
    CD4 = AB4
    ABCD_2 = op.Z_purification(ABCD_1, AB4, [1, 2])[0]
    ABCD_3 = op.Z_purification(ABCD_2, CD4, [3, 4])[0]

    AC3 = op.set_isotropic_state(F)
    AC4 = op.set_isotropic_state(F)
    AC4 = op.XY_purification(AC4, AC3)[0]
    AC4 = op.Z_purification(AC4, AC3)[0]
    BD4 = AC4
    ABCD_4 = op.Z_purification(ABCD_3, AC4, [1, 4])[0]
    ABCD_5 = op.Z_purification(ABCD_4, BD4, [2, 3])[0]

    ABCDv2 = ABCD_5
    ABCD_6 = op.XY_purification(ABCD_5, ABCDv2)[0]

    return ABCD_6


def Minimize4X_22(F):  # This is the version that uses 22 EPR pairs
    """
    Function that calculates the resulting state for a protocol that I found for creating a 4-partite GHZ state, in a
    situation without noise and decoherence, and with the use of 22 isotropic Bell diagonal states with fidelity F.
    """
    AB1 = op.set_isotropic_state(F)
    AB2 = op.set_isotropic_state(F)
    AB2 = op.XY_purification(AB2, AB1)[0]
    AB2 = op.XY_purification(AB2, AB1)[0]

    AC1 = op.set_isotropic_state(F)
    AC2 = op.set_isotropic_state(F)
    AC2 = op.Z_purification(AC2, AC1)[0]
    AC2 = op.Z_purification(AC2, AC1)[0]

    CD2 = AB2

    BD1 = op.set_isotropic_state(F)
    BD2 = op.set_isotropic_state(F)
    BD2 = op.XY_purification(BD2, BD1)[0]
    BD2 = op.Z_purification(BD2, BD1)[0]

    ABCD_0 = op.fuse_GHZ_ancilla(AB2, CD2, AC2)
    ABCD_1 = op.Z_purification(ABCD_0, BD2, [1, 4])[0]

    AD1 = op.set_isotropic_state(F)
    AD2 = op.set_isotropic_state(F)
    AD2 = op.XY_purification(AD2, AD1)[0]
    AD2 = op.Z_purification(AD2, AD1)[0]
    AD2 = op.XY_purification(AD2, AD1)[0]
    AD2 = op.Z_purification(AD2, AD1)[0]

    BC2 = AD2

    ABCD_2 = op.Z_purification(ABCD_1, AD2, [2, 4])[0]
    ABCD_3 = op.Z_purification(ABCD_2, BC2, [1, 3])[0]

    return ABCD_3


def double_selection_X(a, b, c):
    """
    Function that calculates the resulting state for the double selection procedure, where the second step is an X
    purification step, in a situation without noise and decoherence. The three input states a, b and c are all Bell
    diagonal states (vectors with 4 elements); the output state a2 is also a Bell diagonal state.
    """
    if np.size(np.shape(a)) != 1 or np.size(np.shape(b)) != 1 or np.size(np.shape(c)) != 1:
        sys.exit("In double_selection_X: one or more of the input states is not a vector but a matrix.")
    if ceil(log(np.size(a), 2)) != 2 or ceil(log(np.size(b), 2)) != 2 or ceil(log(np.size(c), 2)) != 2:
        sys.exit("In double_selection_X: one or more of the input states is not a Bell diagonal state.")

    b2 = op.Z_purification(b, c)[0]
    a2 = op.XY_purification(a, b2)[0]
    return a2


def double_selection_Z(a, b, c):
    """
    Function that calculates the resulting state for the double selection procedure, where the second step is an Z
    purification step, in a situation without noise and decoherence. The three input states a, b and c are all Bell
    diagonal states (vectors with 4 elements); the output state a2 is also a Bell diagonal state.
    """
    if np.size(np.shape(a)) != 1 or np.size(np.shape(b)) != 1 or np.size(np.shape(c)) != 1:
        sys.exit("In double_selection_Z: one or more of the input states is not a vector but a matrix.")
    if ceil(log(np.size(a), 2)) != 2 or ceil(log(np.size(b), 2)) != 2 or ceil(log(np.size(c), 2)) != 2:
        sys.exit("In double_selection_Z: one or more of the input states is not a Bell diagonal state.")

    b2 = op.Z_purification(b, c)[0]
    a2 = op.Z_purification(a, b2)[0]
    return a2


def Minimize3X(F, v_contr):
    """
    Function that calculates the resulting state for a protocol that I found for creating a 3-partite GHZ state, in a
    situation without noise and decoherence, and with the use of isotropic Bell diagonal states with fidelity F. The
    exact number of Bell states used is determined by the vector v_contr. This vector should be a 1-dimensional vector
    of length 11, and determines how often certain blocks in the protocol are repeated. The output of the function
    is E123, a vector of length 8 describing the 3-qubit GHZ diagonal state that is created, and an integer numb,
    describing the total number of isotropic Bell diagonal states used in the protocol.
    """
    E12 = op.set_isotropic_state(F)
    E12anc = E12

    n0 = v_contr[0]
    n1 = v_contr[1]
    n2 = v_contr[2]
    n3 = v_contr[3]
    n4 = v_contr[4]
    n5 = v_contr[5]
    n6 = v_contr[6]
    n7 = v_contr[7]
    n8 = v_contr[8]
    v0 = v_contr[9]
    v1 = v_contr[10]

    # Phase 1
    for i in range(n0):
        if i % 2 == 0:
            E12 = double_selection_Z(E12, E12anc, E12anc)
        else:
            E12 = double_selection_X(E12, E12anc, E12anc)

    for i in range(n1):
        E12 = op.XY_purification(E12, E12anc)[0]

    # Phase 2
    E23 = E12anc
    E23anc = E23
    for i in range(n2):
        if i % 2 == 0:
            E23 = op.XY_purification(E23, E23anc)[0]
        else:
            E23 = op.Z_purification(E23, E23anc)[0]
    for i in range(n3):
        E23 = op.XY_purification(E23, E23anc)[0]
    E123 = op.fuse_GHZ_local(E12, E23)

    # Phase 3
    E23 = E12anc
    E23anc = E23
    for i in range(n4):
        if i % 2 == 0:
            E23 = op.XY_purification(E23, E23anc)[0]
        else:
            E23 = op.Z_purification(E23, E23anc)[0]
    for i in range(n5):
        E23 = op.Z_purification(E23, E23anc)[0]

    # Phase 4
    E13 = E12anc
    E13anc = E13
    for i in range(n6):
        if i % 2 == 0:
            E13 = op.XY_purification(E13, E13anc)[0]
        else:
            E13 = op.Z_purification(E13, E13anc)[0]
    for i in range(n7):
        E13 = op.Z_purification(E13, E13anc)[0]

    for i in range(n8):
        if v0 == 1:
            E123 = op.Z_purification(E123, E23, [2, 3])[0]
        if v1 == 1:
            E123 = op.Z_purification(E123, E13, [1, 3])[0]

    numb = (1 + 2*n0 + n1) + (1 + n2 + n3) + n8*(v0*(n4 + n5 + 1) + v1*(n6 + n7 + 1))
    return E123, numb


def best_3q_prot(F, nrEPR):
    """
    This is a list of the best protocols for 3-partite GHZ states I found in an earlier stage.

    Parameters
    ----------
    F : bool (between 0 and 1)
        Fidelity of the isotropic Bell diagonal states used
    nrEPR : int (between 2 and 23)
        Number of Bell diagonal states

    Returns
    -------
    state
        A one-dimensional vector containing the coefficients of the three-qubit GHZ diagonal state created by the
        protocol
    """
    if nrEPR < 2 or nrEPR > 23:
        sys.exit("In best_3q_prot: the number of EPR pairs must be between 2 and 23.")

    v_contr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Best protocol found when using 2 EPR pairs
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # Best protocol found when using 3 EPR pairs
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # Best protocol found when using 4 EPR pairs
                        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],  # Best protocol found when using 5 EPR pairs
                        [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],  # Best protocol found when using 6 EPR pairs
                        [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1],  # Best protocol found when using 7 EPR pairs
                        [0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 1],  # Best protocol found when using 8 EPR pairs
                        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],  # Best protocol found when using 9 EPR pairs
                        [2, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],  # Best protocol found when using 10 EPR pairs
                        [2, 0, 2, 1, 0, 0, 0, 1, 1, 0, 1],  # Best protocol found when using 11 EPR pairs
                        [2, 0, 2, 1, 0, 0, 1, 1, 1, 0, 1],  # Best protocol found when using 12 EPR pairs
                        [2, 0, 0, 1, 0, 0, 0, 2, 2, 0, 1],  # Best protocol found when using 13 EPR pairs
                        [1, 1, 2, 1, 0, 0, 0, 2, 2, 0, 1],  # Best protocol found when using 14 EPR pairs
                        [2, 0, 2, 1, 0, 0, 0, 2, 2, 0, 1],  # Best protocol found when using 15 EPR pairs
                        [2, 0, 0, 2, 1, 2, 1, 2, 1, 1, 1],  # Best protocol found when using 16 EPR pairs
                        [2, 0, 0, 2, 0, 0, 0, 2, 3, 0, 1],  # Best protocol found when using 17 EPR pairs
                        [4, 0, 0, 2, 0, 0, 0, 2, 2, 0, 1],  # Best protocol found when using 18 EPR pairs
                        [4, 0, 0, 2, 0, 2, 1, 2, 1, 1, 1],  # Best protocol found when using 19 EPR pairs
                        [4, 0, 0, 2, 1, 2, 1, 2, 1, 1, 1],  # Best protocol found when using 20 EPR pairs
                        [2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 1],  # Best protocol found when using 21 EPR pairs
                        [2, 1, 0, 3, 0, 2, 0, 2, 2, 1, 1],  # Best protocol found when using 22 EPR pairs
                        [2, 1, 0, 2, 0, 2, 1, 2, 2, 1, 1]])  # Best protocol found when using 23 EPR pairs

    state, numb = Minimize3X(F, v_contr[int(nrEPR) - 2])
    if numb != nrEPR:
        sys.exit("In best_3q_prot: the number of EPR pairs is not in agreement with the selected protocol.")
    return state
