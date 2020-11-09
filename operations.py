"""
2020 Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I/
"""
import sys
import numpy as np
from math import ceil, floor, log


def set_isotropic_state(F=0.6, n_or_A=2):
    """
    Create a state that is diagonal in the Bell basis or the GHZ basis, has fidelity F and is isotropic concerning the
    remaining coefficients. The state is represented as a numpy vector that contains the values for the coefficients.
    It is ordered corresponding to the signs of the basis vectors: the coefficient with all plus goes to the first
    index (0), the ones with one minus at the end to the second index (1), etc.
    """
    if F > 1 or F < 0:
        sys.exit("In set_isotropic_state: the fidelity is bigger than 1 or smaller than 0.")
    if np.size(np.shape(n_or_A)) != 1 and np.size(np.shape(n_or_A)) != 0:
        sys.exit("In set_isotropic_state: the second input is not understood.")

    N = np.size(n_or_A)  # It is determined whether the input specifies a number of qubits (n) or the length of a vector
    if N > 1:  # If this is true, the input n_or_A specifies the length of the state
        if ceil(log(N, 2)) != log(N, 2):
            sys.exit("In set_isotropic_state: the size of the diagonal state coefficient vector is not a power of 2.")
    if N == 1:  # If this is true, the input n_or_A specifies the number of qubits that the final state should have
        N = 2 ** n_or_A

    A = np.ones(N) * (1 - F) / (N - 1)
    A[0] = F
    return A


def dec2signs(i, n):
    """
    The decimal number i (between 0 and 2**n) is converted to a bit string list with n binary values
    """

    if i >= 2**n:
        sys.exit("In dec2signs: the size of the decimal index exceeds the maximum bit string length given by input n.")
    if i < 0:
        sys.exit("In dec2signs: the input number i cannot be negative.")

    # Performing this task (1) - this version has proven to be the fastest:
    icop = i  # Make a copy of the number i
    s = [None]*n  # Allocate space for the list s of length n that is going to contain the binary values
    for j in range(n):  # Go over all binary values that are going to represent i
        s[n - j - 1] = icop % 2  # Determine the binary values from the right to left (from last to first in the list)
        icop = icop >> 1  # The last binary bit of icop is removed and the other bits are shifted one place to the right

    """
    Alternative way of performing this task (2) - this version is slower:
    ibin = bin(i)  # Binary version of the decimal number i
    len_ibin = len(ibin) - 2  # Length of the bit string (-2 is because of "0b" in front of every string)
    s = []
    for j in range(n):
        if j < n - len_ibin:
            s.append(0)  # Add as many zeros to the front of the string as requested with the number n
        else:
            s.append(int(ibin[2 + j - n + len_ibin]))  # Fill the rest of the string with the bits from ibin

    Alternative way of performing this task (3) - this version is the slowest:
    s = np.zeros(n)
    for j in range(n):  # Go over all binary values that are going to represent i
       s[j] = floor(i / (2**(n - j - 1))) % 2  # Calculate the binary value
    """
    return s


def double_Z_purification(A_0, B, i=1, k=1):
    """
    Using the Bell diagonal state B to carry out a Z_i Z_(i+k) parity measurement on the n1-qubit GHZ diagonal state
    A_0. This measurement is carried out by using controlled-Z gates and X measurements on the ancilla qubits.
    """
    if np.size(np.shape(A_0)) != 1:
        sys.exit("In double_Z_purification: the input GHZ diagonal state is "
                 "not a vector but a matrix.")
    if np.size(np.shape(B)) != 1:
        sys.exit("In double_Z_purification: the input Bell diagonal state is not a vector but a matrix.")

    N1 = np.size(A_0)
    N2 = np.size(B)
    n1 = log(N1, 2)
    n2 = log(N2, 2)

    if ceil(n1) != n1:
        sys.exit("In double_Z_purification: the size of the GHZ diagonal state coefficient vector is not a power of 2.")
    if ceil(n2) != 2:
        sys.exit("In double_Z_purification: the size of the ancilla Bell diagonal state is not 4.")
    if i > int(n1) - 1 or i < 1:
        sys.exit("In double_Z_purification: the index i indicating which Z_i Z_(i+k) generator one wants to measure is "
                 "out of bounds.")
    if k > int(n1) - i or k < 1:
        sys.exit("In double_Z_purification: the index k indicating which Z_i Z_(i+k) generator one wants to measure is "
                 "out of bounds.")

    n1 = int(n1)  # n1 in this function corresponds to n in the document

    # In this function, i corresponds to b_1 in the document and k corresponds to c_1
    # j in this function corresponds to s' in the document

    A_1 = np.zeros(2 ** n1)
    for s in range(2 ** n1):  # Here, s is taken in the range from 0 to 2**n - 1
        j = 0
        for t in range(k):  # Here, t is taken from 0 to c_1 - 1
            j = j + floor(s / (2 ** (n1 - i - 1 - t)))  # Here, s' is calculated
        j = j % 2
        A_1[s] = B[j * 2] * A_0[s] + B[j * 2 + 1] * A_0[(s + 2 ** (n1 - 1)) % (2 ** n1)]
    A_1 = A_1 / np.sum(A_1)  # Here the normalization takes place (divide by the success probability)
    return A_1


def Z_purification(A_0, B, list_ind=None):
    """
    Function that carries out the non-local Z_(b_1) Z_(b_1+c_1) Z_(b_2) Z_(b_2+c_2) ... Z_(b_m) Z_(b_m+c_m) parity
    measurement on the n1-qubit GHZ diagonal state A_0 by consuming the (n2 = 2*m)-qubit GHZ diagonal state B in the
    process. The list "list_ind" is supposed to contain the values [b_1,b_1+c_1,b_2,b_2+c_2,...,b_m,b_m+c_m].
    The idea is thus to have this list indicating which qubit numbers from A_0 are included in the joint Z parity
    measurement. The input list_ind should be list having an even number of elements, but it can also be a decimal
    value whose binary representation describes which double-Z stabilizer generators are involved in the measurement
    (e.g., for a 4-qubit state, lint_ind = 4 corresponds to 100 and describes measuring the Z1 Z2 operator, whereas
    list_ind = 3 corresponds to 011 and describes measuring Z2 Z4). In this function below, list_ind is split into
    two lists with b_1,...,b_m values and c_1,...,c_m values that are used in the document.
    """
    if np.size(np.shape(A_0)) != 1:
        sys.exit("In Z_purification: the input main GHZ diagonal state is not a vector but a matrix.")
    if np.size(np.shape(B)) != 1:
        sys.exit("In Z_purification: the input ancilla GHZ diagonal state is not a vector but a matrix.")

    N1 = np.size(A_0)
    N2 = np.size(B)
    n1 = log(N1, 2)
    n2 = log(N2, 2)

    if ceil(n1) != n1:
        sys.exit("In Z_purification: the size of the GHZ diagonal state coefficient vector is not a power of 2.")
    if ceil(n2) != n2:
        sys.exit("In Z_purification: the size of the ancilla GHZ diagonal state vector is not a power of 2.")

    list_ind = transform_list_ind(list_ind, n1, n2, 'Z')  # list_ind is transformed to the right format
    anc_weight = len(list_ind)

    if anc_weight == 0:
        sys.exit("In Z_purification: the list of indices indicating on which qubits the joint Z measurement is carried "
                 "out is empty.")
    if anc_weight % 2 != 0:
        sys.exit("In Z_purification: the list of indices indicating on which qubits the joint Z measurement is carried "
                 "out is not of even length.")
    if anc_weight != n2:
        sys.exit("In Z_purification: the size of the ancilla state B is not in agreement with the length of the list "
                 "indices indicating on which qubits of A_0 the measurement is carried out.")
    if list_ind[anc_weight - 1] > int(n1) or list_ind[0] < 1:
        sys.exit("In Z_purification: the indices indicating on which qubits the parity measurement is carried out are "
                 "out of bounds.")

    b_ind = list_ind[0::2]  # Make the list b_1,...,b_m from the document
    c_ind = list_ind[1::2]  # Make the list with all values b_1+c_1,...,b_m+c_m
    for i1 in range(int(anc_weight/2)):
        c_ind[i1] = c_ind[i1] - b_ind[i1]  # Make the list c_1,...,c_m from the document
        if c_ind[i1] < 1:
            sys.exit("In Z_purification: the indices in list_ind are not unique.")  # Check if none of the values of
            # c_ind is zero or negative

    n1 = int(n1)
    n2 = int(n2)
    m = int(n2/2)

    A_1 = np.zeros(2**n1)
    for s in range(2**n1):  # Go over all indices s (in the range from 0 to 2**n - 1) of A_1
        signs_s = dec2signs(s, n1)  # The decimal s is transformed into a list of signs (with 0 as +1 and 1 as -1)
        j = []  # This is going to be a list j[i] of elements containing the products sigma_(b_i+1)*sigma_(b_i+c_i)
        for i in range(m):  # Go over all pairs of Z's involved in the measurement
            j.append(0)
            for t in range(c_ind[i]):  # Here, t is taken from 0 to c_i - 1
                j[i] += signs_s[b_ind[i] + t]  # Given that signs_s[0] = sigma_1, this adds sigma[b_i + 1 + t]
                # # Alternative way of calculating this:
                # j[i] = j[i] + floor(s / (2**(n1 - b_ind[i] - t - 1)))
        jtot = sum(j) % 2  # The index sigma_(b_1+1)*...*sigma_(b_1+c_1)*...*sigma_(b_m+1)*...*sigma_(b_m+c_m)
        for t in range(2**(n2 - 1)):  # Go over all decimal values of lambda_2, lambda_3, ..., lambda_(n2 = 2*m)
            i1 = jtot * (2**(n2 - 1)) + t  # Index of B for index (jtot, lambda_2, lambda_3, ..., lambda_(n2 = 2*m))
            lamprod = 0  # This is going to be the product of all values lambda_2*lambda_4*lambda_6*...*lambda_(2*m)
            tcop = t
            for _ in range(m):
                lamprod += (tcop % 2)  # lambda_(2 * (m - _)) is added to lambda_2 * lambda_4 * ... * lambda_(2*m)
                tcop = tcop >> 2  # Last two bits of tcop are removed and the other bits are shifted one to the right

            # First way of finishing this function (1):
            lamprod = (lamprod % 2) << (n1 - 1)  # Make lamprod into a bit and add (n1 - 1) zero bits behind it
            i2 = lamprod ^ s  # Do a bitwise XOR between lamprod and the first bit of s (sigma_1)
            A_1[s] += B[int(i1)] * A_0[int(i2)]  # Calculate the new coefficient of A_1

            """
            Alternative way of finishing this function (2):
            i2 = (s + 2**(n1 - 1) * (lamprod % 2)) % (2 ** n1)
            A_1[s] += B[int(i1)] * A_0[int(i2)]  # Calculate the new coefficient of A_1

            Alternate way of finishing this function (3):
            sigma_1 = floor(s / 2**(n1 - 1))
            lamprod = (lamprod + sigma_1) % 2  # Here, lambda_2*lambda_3*...*lambda_(2m)*sigma_1 is calculated
            i2 = s + (lamprod - sigma_1)*2**(n1 - 1)  # sigma_1 is replaced for lamprod in the decimal representation s
            A_1[s] += B[int(i1)] * A_0[int(i2)]  # Calculate the new coefficient of A_1
            """
    succ_prob = np.sum(A_1)
    A_1 = A_1 / np.sum(A_1)  # Here the normalization takes place (divide by the success probability)
    return A_1, succ_prob


def XY_purification(A_0, B, list_ind=None):
    """
    Function that carries out a non-local joint Pauli stabilizer measurement where the operator that is measured is
    a product of the all-X stabilizer generators X_1 X_2 ... X_n and the combination of double-Z generators of the
    form Z_(b_1) Z_(b_1+c_1) Z_(b_2) Z_(b_2+c_2) ... Z_(b_m) Z_(b_m+c_m). The purification measurement is carried
    out on the n-qubit GHZ diagonal state A_0 and in the process another GHZ diagonal state B of the exact same size
    is consumed. The list "list_ind" is supposed to contain the values [b_1,b_1+c_1,b_2,b_2+c_2,...,b_m,b_m+c_m]
    indicating which of the qubits are being measured as Y operators in the joint measurement; all qubits not in the
    list are measured as X operators in the joint measurement. The input list_ind should be list having an even
    number of elements, but it can also be a decimal value whose binary representation describes which double-Z
    stabilizer generators are involved in the measurement (e.g., for a 4-qubit state, lint_ind = 4 corresponds to
    100 and describes measuring the - Y1 Y2 Z3 Z4 operator, whereas list_ind = 3 corresponds to 011 and describes
    measuring - X1 Y2 X3 Y4). In this function below, list_ind is split into two lists with b_1,...,b_m values and
    c_1,...,c_m values that are used in the document.
    """
    if np.size(np.shape(A_0)) != 1:
        sys.exit("In XY_purification: the input main GHZ diagonal state is not a vector but a matrix.")
    if np.size(np.shape(B)) != 1:
        sys.exit("In XY_purification: the input ancilla GHZ diagonal state is not a vector but a matrix.")

    N1 = np.size(A_0)
    N2 = np.size(B)
    n1 = log(N1, 2)
    n2 = log(N2, 2)

    if ceil(n1) != n1:
        sys.exit("In XY_purification: the size of the GHZ diagonal state coefficient vector is not a power of 2.")
    if ceil(n2) != n2:
        sys.exit("In XY_purification: the size of the ancilla GHZ diagonal state vector is not a power of 2.")
    if int(n1) != int(n2):
        sys.exit("In XY_purification: the sizes of the main GHZ diagonal state and the ancilla GHZ diagonal state are "
                 "not the same.")

    list_ind = transform_list_ind(list_ind, n1, n2, 'XY')  # list_ind is transformed to the right format
    anc_weight = len(list_ind)

    if anc_weight % 2 != 0:
        sys.exit("In XY_purification: the list of indices indicating which qubits in the joint measurement are"
                 "measured in Y is not of even length.")
    if anc_weight > 0:
        if list_ind[anc_weight - 1] > int(n1) or list_ind[0] < 1:
            sys.exit("In XY_purification: the indices indicating on which qubits the parity measurement is carried out "
                     "are out of bounds.")

    b_ind = list_ind[0::2]  # Make the list b_1,...,b_m from the document
    c_ind = list_ind[1::2]  # Make the list with all values b_1+c_1,...,b_m+c_m
    for i1 in range(int(anc_weight/2)):
        c_ind[i1] = c_ind[i1] - b_ind[i1]  # Make the list c_1,...,c_m from the document
        if c_ind[i1] < 1:
            sys.exit("In XY_purification: the indices in list_ind are not unique.")  # Check if none of the values of
            # c_ind is zero or negative

    n = int(n1)
    m = int(anc_weight/2)

    A_1 = np.zeros(2**n)
    for s in range(2**n):  # Go over all indices s (in the range from 0 to 2**n - 1) of A_1
        signs_s = dec2signs(s, n)  # The decimal s is transformed into a list of signs (with 0 as +1 and 1 as -1)
        js = []  # This is going to be a list js[i] of elements containing the products sigma_(b_i+1)*sigma_(b_i+c_i)
        for i in range(m):  # Go over all pairs of Y operators involved in the measurement
            js.append(0)
            for t in range(c_ind[i]):  # Here, t is taken from 0 to c_i - 1
                js[i] += signs_s[b_ind[i] + t]  # Given that signs_s[0] = sigma_1, this adds sigma[b_i + 1 + t]
        jstot = sum(js) % 2  # The index sigma_(b_1+1)*...*sigma_(b_1+c_1)*...*sigma_(b_m+1)*...*sigma_(b_m+c_m)
        for u in range(2**(n - 1)):  # Go over all decimal values of lambda_2, lambda_3, ..., lambda_n
            signs_u = dec2signs(u, n - 1)  # The decimal u is transformed into a list of signs lambda_2, ..., lambda_n
            jl = []  # This will be a list jl[i] of elements containing the products lambda_(b_i+1)*lambda_(b_i+c_i)
            for i in range(m):  # Go over all pairs of Z's involved in the measurement
                jl.append(0)
                for t in range(c_ind[i]):  # Here, t is taken from 0 to c_i - 1
                    jl[i] += signs_u[b_ind[i] + t - 1]  # Since signs_u[0] = lambda_2, this adds lambda[b_i + 1 + t]
            jltot = sum(jl) % 2  # Index lambda_(b_1+1)*...*lambda_(b_1+c_1)*...*lambda_(b_m+1)*...*lambda_(b_m+c_m)
            ind1B = ((signs_s[0] + jltot + jstot) % 2) << (n - 1)  # The first sign of the index of B with n - 1 zeros
            i1 = ind1B ^ u  # The index of B
            ind1A = jltot << (n - 1) # Part of A_0's first sign with n - 1 zeros at the end (sigma_1 is included below)
            i2 = ind1A ^ s ^ u  # Index of A_0 (first XOR does jtot*sigma_1, the other does lambda_i*sigma_i for i>1)
            A_1[s] += B[int(i1)] * A_0[int(i2)]  # Calculate the new coefficient of A_1
    succ_prob = np.sum(A_1)
    A_1 = A_1 / np.sum(A_1)  # Here the normalization takes place (divide by the success probability)
    return A_1, succ_prob


def purification(main_state, ancilla_state, dec, get_prob=False):
    """
    More general form of the specific purification functions above.

    Parameters
    ----------
    main_state : one-dimensional numpy-object containing the 2**n1 coefficients of an n1-qubit GHZ diagonal state
        Object describing the main state in the purification process
    ancilla_state : one-dimensional numpy-object containing the 2**n2 coefficients of an n2-qubit GHZ diagonal state
        Object describing the ancillary state in the purification process
    dec : integer in range [0, 2**n1]
        Integer describing what type of non-local stabilizer measurement must be carried out on the main state with
        the aid of the ancillary state. This decimal integer is converted to a binary number that describes which
        stabilizer generators of the n1-qubit GHZ state are involved in the stabilizer measurement
    get_prob : Boolean
        Indicates whether or not the success probability should be returned as well as extra output parameter

    Returns
    -------
    one-dimensional numpy-object containing the 2**n1 coefficients of an n1-qubit GHZ diagonal state
        Object describing the purified state
    (optional) float indicating the success probability of the operation
    """
    if np.size(np.shape(main_state)) != 1:
        sys.exit("In purification: the input main GHZ diagonal state is not a vector but a matrix.")
    if np.size(np.shape(ancilla_state)) != 1:
        sys.exit("In purification: the input ancilla GHZ diagonal state is not a vector but a matrix.")

    N1 = np.size(main_state)
    n1 = log(N1, 2)
    if ceil(n1) != n1:
        sys.exit("In purification: the size of the GHZ diagonal state coefficient vector is not a power of 2.")
    n1 = int(n1)

    if dec >= 2**n1:
        sys.exit("In purification: the decimal number describing the stabilizer that is measured is out of bounds.")
    if dec >= 2**(n1 - 1):  # In this situation the XY_purification is called
        dec2 = dec - 2**(n1 - 1)
        if get_prob:
            return XY_purification(main_state, ancilla_state, dec2)
        else:
            return XY_purification(main_state, ancilla_state, dec2)[0]
    else:
        if get_prob:
            return Z_purification(main_state, ancilla_state, dec)
        else:
            return Z_purification(main_state, ancilla_state, dec)[0]


def transform_list_ind(list_ind, n1, n2, list_type='Z'):
    """
    Function that makes sure the input list_ind used for the functions Z_purification and XY_purification has the
    right format for further processing by these functions. The parameter n1 describes the number of qubits involved
    in the GHZ-diagonal diagonal state: there are always (n1 - 1) double-Z generators of the stabilizer (Z_1 Z_2,
    Z_2 Z_3, Z_3 Z_4, ..., Z_(n1 - 1) Z_(n1). The parameter n2 describes the number of qubits of the ancilla state
    that is used to carry out the non-local measurement; for the full weight stabilizer measurement (i.e., the
    measurements carried out by the function XY_purification) we must have n1 = n2.
    """
    list_new = list_ind  # If the list is in the right format, it only has to be sorted (otherwise it is overwritten)
    if list_ind is None:  # If list_ind is not specified, it is interpreted in a pre-determined way for both Z and XY
        if list_type == 'Z':
            list_new = list(range(1, int(n2)+1))  # Without a lint_ind, the Z-function measures joint Z's on the first
            # qubits, where the size of the ancilla state B determines how many of them are included
        elif list_type == 'XY':
            list_new = []  # Without a lint_ind, the XY-function measures A_0 in the all-X operator X_1 X_2 ... X_n
    elif isinstance(list_ind, (int,)):  # If the input list_ind is given as a single decimal integer, this is
        # interpreted as a bit string indicating which stabilizers are included (i.e., have a 1 in this bit string) and
        # which are excluded (i.e., have a 0 in this bit string) in the full product of the joint Pauli measurement.

        if list_ind >= (2 ** (n1 - 1)):
            sys.exit("In transform_list_ind: the decimal index exceeds the bit string length given by input n1.")
        if list_ind < 0:
            sys.exit("In transform_list_ind: the input number i cannot be negative.")

        list_string = dec2signs(list_ind, int(n1) - 1)  # list_string[i] indicates whether Z_(i+1) Z_(i+2) is involved

        list_new = []
        if list_string[0] == 1:  # If the first bit is 1 (the stabilizer Z1 Z2 is involved), qubit 1 must be included.
            list_new.append(1)
        for i in range(int(n1) - 2):
            if list_string[i] ^ list_string[i + 1] == 1:  # If only one of Z_(i+1) Z_(i+2) and Z_(i+2) Z_(i+3) is
                list_new.append(i + 2)                    # involved, we know that qubit (i+2) must be included.
        if list_string[int(n1) - 2] == 1:  # If the bit for stabilizer Z_(n1-1) Z_(n1)) is 1, qubit n1 must be included.
            list_new.append(int(n1))
    list_new.sort()  # The list indicating on which qubits of A_0 the joint Z measurement is carried out is sorted
    return list_new


def full_weight_X_purification(A_0, A):
    """
    Using the ancilla GHZ diagonal state A of size n2=n1 to carry out a X_1 X_2 ... X_(n1) parity measurement on the
    n1-qubit GHZ diagonal state A_0. This measurement is carried out by using controlled-X gates (with the control on
    the ancilla qubits) and Z measurements on the ancilla qubits.
    """
    if np.size(np.shape(A_0)) != 1:
        sys.exit("In full_weight_X_purification: the input main GHZ diagonal state is not a vector but a matrix.")
    if np.size(np.shape(A)) != 1:
        sys.exit("In full_weight_X_purification: the input ancilla GHZ diagonal state is not a vector but a matrix.")

    N1 = np.size(A_0)
    N2 = np.size(A)
    n1 = log(N1, 2)
    n2 = log(N2, 2)

    if ceil(n1) != n1:
        sys.exit("In full_weight_X_purification: the size of the main GHZ diagonal state coefficient vector is not a "
                 "power of 2.")
    if ceil(n2) != n2:
        sys.exit("In full_weight_X_purification: the size of the ancilla GHZ diagonal state coefficient vector is not "
                 "a power of 2.")
    if int(n1) != int(n2):
        sys.exit("In full_weight_X_purification: the sizes of the main GHZ diagonal state and the ancilla GHZ diagonal "
                 "state are not the same.")

    n = int(n1)

    A_1 = np.zeros(2 ** n)
    j = np.zeros(n)
    for i in range(2 ** n):  # Go over all indices of the new function A_1
        for i3 in range(n):
            j[i3] = floor(i / (2 ** (n - i3 - 1))) % 2  # Make a list of all signs sigma_j for this value of i
        for t in range(2 ** (n - 1)):  # Go over all decimal representations of the signs lambda_2, ..., lambda_n
            i1 = j[0] * (2 ** (n - 1)) + t  # Calculate the index of the ancilla state A (sigma_1 corresponds to j[0]
            # and t to the decimal representation of lambda_2, lambda_3, ..., lambda_n)
            i2 = j[0] * (2 ** (n - 1))  # Make a start for the index of A_0, given that its first sign is sigma_1 = j[0]
            for i3 in range(n - 1):  # Go over all indices of j again; from j[1]=sigma_2 to j[n-1]=sigma_n in this case
                s = floor(t / (2 ** (n - i3 - 2))) % 2  # Extract lambda_(i3+2) from the decimal value t
                i2 += ((j[i3 + 1] + s) % 2) * (2 ** (n - 2 - i3))  # Add lambda_(i3+2)*sigma_(i3+2) = s*j[i3+1] to i2
            A_1[i] += A[int(i1)] * A_0[int(i2)]  # Calculate the corresponding coefficient of A_1
    A_1 = A_1 / np.sum(A_1)  # Normalize A_1 by dividing by the success probability
    return A_1


def fuse_GHZ_ancilla(A, B, C, i=0, j=0):
    """
    Merge two remote GHZ states, the m-qubit GHZ state A and the n-qubit GHZ state B, into an (m+n)-qubit GHZ state D,
    by performing a Z_(m-i)^(A) Z_(1+j)^(B) parity measurement using the Bell pair C. This parity measurement is
    carried out by using controlled-Z gates and X measurements on the ancilla qubits.
    """
    if np.size(np.shape(A)) != 1:
        sys.exit("In fuse_GHZ_ancilla: the input GHZ diagonal state A is not a vector but a matrix.")
    if np.size(np.shape(B)) != 1:
        sys.exit("In fuse_GHZ_ancilla: the input GHZ diagonal state B is not a vector but a matrix.")
    if np.size(np.shape(C)) != 1 or ceil(log(np.size(C), 2)) != 2:
        sys.exit("In fuse_GHZ_ancilla: the input Bell diagonal state C is not of the right structure.")

    M = np.size(A)
    N = np.size(B)
    m = log(M, 2)
    n = log(N, 2)

    if ceil(m) != m:
        sys.exit("In fuse_GHZ_ancilla: the size of the GHZ diagonal state coefficient vector A is not a power of 2.")
    if ceil(n) != n:
        sys.exit("In fuse_GHZ_ancilla: the size of the GHZ diagonal state coefficient vector B is not a power of 2.")
    if i < 0 or i > int(m - 1):
        sys.exit("In fuse_GHZ_ancilla: the index i is out of bounds.")
    if j < 0 or j > int(n - 1):
        sys.exit("In fuse_GHZ_ancilla: the index j is out of bounds.")

    m = int(m)
    n = int(n)

    D = np.zeros(2 ** (m + n))  # This is the new (m+n)-qubit GHZ diagonal state D
    for iD in range(2 ** (m + n)):  # Here we go over all coefficients of the new state D
        iA = 0
        for i2 in range(m - 1):
            iA += (floor(iD / (2 ** (m + n - i2 - 2))) % 2) * (2 ** (m - 2 - i2))  # Here we calculate the contributions
            # of sigma_2 to sigma_m to iA
        iAp = iA + (floor(iD / (2 ** (m + n - 1))) % 2) * (2 ** (m - 1))  # Here we calculate iA in case mu_1*mu_2 = +1
        iAm = iA + ((floor(iD / (2 ** (m + n - 1))) + 1) % 2) * (2 ** (m - 1))  # Here we calculate iA in case
        # mu_1*mu_2 = -1

        iB = 0
        for i3 in range(n - 1):
            iB += (floor(iD / (2 ** (n - i3 - 2))) % 2) * (2 ** (n - 2 - i3))  # Here we calculate the contributions of
            # sigma_(m+2) to sigma_(m+n) to iB
        iBp = iB  # Here we calculate iB for mu_1 = +1
        iBm = iB + 2 ** (n - 1)  # Here we calculate iB for mu_1 = -1

        iC = floor(iD / (2 ** (n - 1)))  # Here we calculate the contribution of sigma_(m+1) to iC
        for i4 in range(i):
            iC += floor(iD / 2 ** (n + i - i4 - 1))  # Here we correct for this term with the sigma_(m-i+xi) signs
        for i5 in range(j):
            iC += floor(iD / 2 ** (n - i5 - 2))  # Here we correct for this term with the sigma_(m+chi+1) signs
        iC = iC % 2
        iCp = iC * 2  # Here we calculate iC for mu_2 = +1
        iCm = iC * 2 + 1  # Here we calculate iC for mu_2 = -1

        D[iD] = A[iAp] * B[iBp] * C[iCp] + A[iAm] * B[iBp] * C[iCm] + A[iAm] * B[iBm] * C[iCp] + A[iAp] * B[iBm] * C[
            iCm]

    return D


def fuse_GHZ_local(A, B, i=0, j=0):
    """
    Fuse an m-qubit diagonal GHZ state A and an n-qubit diagonal GHZ state B that share a node for
    qubit (m-i) of A and qubit (1+j) of B. This operation uses a controlled-X gate (with the control
    on the qubit of A, which is the qubit that remains after the measurement) and a Z measurement on the ancilla
    qubit (the qubit from B).
    """
    if np.size(np.shape(A)) != 1:
        sys.exit("In fuse_GHZ_local: the input GHZ diagonal state A is not a vector but a matrix.")
    if np.size(np.shape(B)) != 1:
        sys.exit("In fuse_GHZ_local: the input GHZ diagonal state B is not a vector but a matrix.")

    M = np.size(A)
    N = np.size(B)
    m = log(M, 2)
    n = log(N, 2)

    if ceil(m) != m:
        sys.exit("In fuse_GHZ_local: the size of the GHZ diagonal state coefficient vector A is not a power of 2.")
    if ceil(n) != n:
        sys.exit("In fuse_GHZ_local: the size of the GHZ diagonal state coefficient vector B is not a power of 2.")
    if i < 0 or i > int(m - 1):
        sys.exit("In fuse_GHZ_local: the index i is out of bounds.")
    if j < 0 or j > int(n - 1):
        sys.exit("In fuse_GHZ_local: the index j is out of bounds.")

    m = int(m)
    n = int(n)

    C = np.zeros(2 ** (m + n - 1))
    for iC in range(2 ** (m + n - 1)):
        iA = 0
        for i2 in range(2, m + 1):
            iA += (floor(iC / (2 ** (m + n - 1 - i2)) % 2)) * (2 ** (m - i2))

        iAp = iA + (floor(iC / (2 ** (m + n - 1 - 1))) % 2) * (2 ** (m - 1))  # The index of A with mu_1 = +1
        iAm = iA + ((floor(iC / (2 ** (m + n - 1 - 1))) + 1) % 2) * (2 ** (m - 1))  # The index of A with mu_1 = -1

        iB = 0
        for i3 in range(2, n + 1):  # These are the indices that describe the contributions from the last n-1 signs
            # of B (from the second sign to the nth sign)
            if i3 > j + 2:  # The last signs in the list (from (m+j+2) to (m+n-1))
                iB += (floor(iC / (2 ** (n - i3))) % 2) * (2 ** (n - i3))
            if i3 == j + 2:  # This is the term with the blue and red products
                iBm2 = 0
                for i5 in range(i):  # The red product contribution
                    iBm2 += floor(iC / (2 ** (n + i5 - 1)))  # Here we have m + n - 1 - (m - i5) = n + i5 -1
                for i6 in range(j):  # The blue product contribution
                    iBm2 += floor(iC / (2 ** (n - i6 - 2)))  # Here we have m + n - 1 - (m + i6 + 1) = n - i6 - 2
                iBm2 += floor(iC / (2 ** (n - j - 2)))  # This is the last term in the blue product
                iB += (iBm2 % 2) * (2 ** (n - i3))
            if i3 == j + 1:  # This is the term with the orange and green products
                iBm1 = 0
                for i5 in range(i):  # The green product contribution
                    iBm1 += floor(iC / (2 ** (n + i5 - 1)))
                for i6 in range(j):  # The orange product contribution
                    iBm1 += floor(iC / (2 ** (n - i6 - 2)))
                iB += (iBm1 % 2) * (2 ** (n - i3))
            if i3 < j + 1:
                iB += (floor(iC / (2 ** (n - 1 - i3))) % 2) * (2 ** (n - i3))  # m + n - 1 - (m + i3) = n - 1 - i3
        iBp = iB  # The index of B with mu_1 = +1
        iBm = iB + 2 ** (n - 1)  # The index of B with mu_1 = -1

        C[iC] = A[int(iAp)] * B[int(iBp)] + A[int(iAm)] * B[int(iBm)]  # Total expression for iC index with both
        # mu_1 = +1 and mu_1 = -1

    return C
