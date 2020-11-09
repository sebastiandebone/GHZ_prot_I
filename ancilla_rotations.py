"""
2020 Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
_____________________________________________
"""
import sys
import numpy as np
from math import ceil, log


def ancilla_rotation(state, dec):
    """
    Function that performs all possible rotations that are possible under local operations for a Bell diagonal state
    or a GHZ diagonal state. The Bell diagonal state (the case n=2 below) is special in the sense that all its
    coefficients can be switched with each other. Therefore, the n=2 case must be treated separately below. For the
    Bell diagonal state, the index 'dec' takes on values 0 (no rotations), 1 (+- <-> -+), 2, (+- <-> --) and
    3 (-+ <-> --). For a GHZ diagonal state, only specific rotations can be performed under local operations. Here,
    the parameter 'dec' runs from 0 to 2**(n-1)-1 and indicates which ZZ generators are included in the rotation (by
    interpreting 'dec' as a binary bit string, with the first bit indicating the Z_1 Z_2 generator, etc.).
    """
    if np.size(np.shape(state)) != 1:
        sys.exit("In ancilla_rotation: the input GHZ diagonal state is not a vector but a matrix.")
    
    N = np.size(state)
    n = log(N, 2)

    if ceil(n) != n:
        sys.exit("In ancilla_rotation: the size of the GHZ diagonal state coefficient vector is not a power of 2.")
    
    n = int(n)
    
    if n == 2:
        if dec < 0 or dec > 3:
            sys.exit("In ancilla_rotation: the value of the decimal value 'dec' exceeds what is possible for the size "
                     "of state 'state'")
    else:
        if dec < 0 or dec >= 2**(n - 1):
            sys.exit("In ancilla_rotation: the value of the decimal value 'dec' exceeds what is possible for the size "
                     "of state 'state'")
   
    state_return = np.zeros(2**n)

    if n == 2:  # The Bell diagonal state case has to be treated separately and uses a different format
        state_return[0] = state[0]
        if dec == 0:  # If dec = 0, there are no rotations
            state_return[1] = state[1]
            state_return[2] = state[2]
            state_return[3] = state[3]
        elif dec == 1:  # In this case we have that +- <-> -+
            state_return[1] = state[2]
            state_return[2] = state[1]
            state_return[3] = state[3]
        elif dec == 2:  # In this case we have that +- <-> --
            state_return[1] = state[3]
            state_return[2] = state[2]
            state_return[3] = state[1]
        elif dec == 3:  # In this case we have that -+ <-> --
            state_return[1] = state[1]
            state_return[2] = state[3]
            state_return[3] = state[2]

    else:  # For n > 2
        # Create look-up object to determine Hamming weight per decimal number:
        hw = [0]*(2**(n - 1))  # Allocate space for list of Hamming weights
        for i in range(2**(n - 1)):
            hwi = 0  # The value for the Hamming weight of this value i
            icop = i  # Make a copy of i
            for j in range(n - 1):
                hwi += icop % 2  # Add the value of the last bit of icop to hwi
                icop = icop >> 1  # Remove the last bit of icop
            hw[i] = hwi  # Add the Hamming weight of this i to the list hw
            
        fp = 2**(n - 1)  # Number that should be added to convert (+1),sigma_2,...,sigma_n into (-1),sigma_2,...,sigma_n
        for i in range(2**(n - 1)):  # Check all coefficients with sigma_1 = 0
            sp = i & dec  # Binary AND
            if (hw[sp] % 2) == 1:  # Switch if the Hamming weight of sp is odd
                state_return[i] = state[fp + i]
                state_return[fp + i] = state[i]
            else:  # Keep the coefficients in the same order if the Hamming weight of sp is even
                state_return[i] = state[i]
                state_return[fp + i] = state[fp + i]

    return state_return


def smallest_coefficient_sum(state):
    """
    Function that finds the value of the smallest sum of interchangeable coefficients of the n-qubit GHZ diagonal
    state 'state' by trying all rotations on 'state' reachable within local operations. This is done by summing the
    coefficients +, sigma_1, sigma_2, ..., sigma_n for every rotation on 'state', where the coefficient associated
    with +, +, ..., + is excluded from the sum (this is state[0]). The sum is therefore taken over 2**(n-1)-1 terms.
    """
    if np.size(np.shape(state)) != 1:
        sys.exit("In smallest_coefficient_sum: the input GHZ diagonal state is not a vector but a matrix.")

    N = np.size(state)
    n = log(N, 2)

    if ceil(n) != n:
        sys.exit("In smallest_coefficient_sum: the size of the GHZ diagonal state coefficient vector is not a power "
                 "of 2.")

    n = int(n)

    coeff_sum = 1
    if n == 2:  # For a Bell diagonal state
        for i in range(3):  # Try all coefficient rotations
            new_state = ancilla_rotation(state, i)  # Calculate the new state for the rotation
            if new_state[1] < coeff_sum:  # If +- coefficient is the smallest so far, it is marked as the smallest
                coeff_sum = new_state[1]
    else:  # For a GHZ diagonal state with more than 3 qubits or more
        for i in range(2**(n - 1)):  # Try all coefficient rotations reachable under local operations
            new_state = ancilla_rotation(state, i)  # Calculate the new state for the rotation
            new_sum = 0  # The new value for the sum of coefficients
            for j in range(1, 2**(n - 1)):  # Exclude the +, +, ..., + coefficient by starting at index 1
                new_sum += new_state[j]  # Calculate the sum of the coefficients +, sigma_1, sigma_2, ..., sigma_n
            if new_sum < coeff_sum:  # If the sum is the smallest so far, it is marked as the smallest sum
                coeff_sum = new_sum

    return coeff_sum
