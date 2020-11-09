"""
2020 Runsheng Ouyang, Sébastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I/
"""
import sys
import numpy as np
import operations as op
import ancilla_rotations as ar
import copy


class Node:
    """
    A class used to capture the structure of binary tree protocols found by the dynamic algorithm.

    ...

    Attributes
    ----------
    value : matrix containing objects of class Path
        These are the data[n][k] objects referred to above
    left : matrix containing objects of class Path
        These are the data[n1][k2] objects referred to above
    right : matrix containing objects of class Path
        These are the data[n2][k2] objects referred to above
    """
    def __init__(self, value, left=None, right=None, id=None, root=None, lr=None):
        self.value = value  # data[n][k]
        self.left = left    # left one should be data[n1][k1]
        self.right = right  # right one should be data[n2][k2]
        self.id = id
        self.root = root
        self.lr = lr


def identify_protocol(data, n, k, t, id=0, root=None, lr=None):
    """
    Function that identifies a protocol from the binary tree structure of the data object, and outputs an object that
    only contains the operations used for the protocol found at this particular values n and k.

    Parameters
    ----------
    data : (n_max+1, k_max+1) matrix with objects of class Path
        Each element (n, k) of this matrix is used to store information about (n, k) itself and how it's made
    n : positive integer smaller than or equal to n_max
        Number of parties for which we want to update the element in data
    k : positive integer smaller than or equal to k_max
        Number of Bell diagonal states for which we want to update the element in data
    t : positive integer
        Should specify how many protocols are stored per value of n and k
    id : integer
        Identifier of the concerning branch
    root : object
        binary tree protocol with purification and distillation operations
    lr : Boolean
        integer that describes if a protocol is a left (0) or a right (1) branch in the binary tree

    Returns
    -------
    protocol : binary tree with purification and distillation operations
    """
    if (n < 2) | (k < (n - 1)):
        sys.exit("In find_protocol_new: should satisfy (n>=2)&(k>=n-1)!")
    if (n == 2) & (k == 1):
        protocol = Node(data[2][1][t], id=id, root=root, lr=lr)
        return protocol
    else:
        if data[n][k][t].p_or_f == 0:  # purification n=n1 k=k1+k2 n,k,n2,k2 known
            n1 = n
        else:  # fusion n=n1+n2-1 k=k1+k2 n,k,n2,k2 known
            n1 = n + 1 - data[n][k][t].n2
        k1 = k - data[n][k][t].k2
        n2 = data[n][k][t].n2
        k2 = data[n][k][t].k2
        t1 = data[n][k][t].t1
        t2 = data[n][k][t].t2
        protocol = Node(data[n][k][t], id=id, root=root, lr=lr)
        # if t1 == 1:
        #     print(n, k, t, "->", n1, k1, t1)
        # if t == 1:
        #     print(n, k, t, "->", n1, k1, t1, data[n][k][t].t)
        protocol.left = identify_protocol(data, n1, k1, t1, id, protocol, 0)  # preorder
        protocol.right = identify_protocol(data, n2, k2, t2, id, protocol, 1)
        return protocol


def protocol_add_id_nrs(protocol):
    """
    Function that adds identification numbers to the nodes in a binary tree GHZ creation protocol

    Parameters
    ----------
    protocol : binary tree with purification and distillation operations

    Returns
    -------
    protocol : binary tree with purification and distillation operations
    """
    # non-recursive preorder
    id = 0
    if protocol == None:
        return
    myStack = []
    node = protocol
    while node or myStack:
        while node:  # function here
            node.id = id
            id += 1
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    return protocol


def protocol_debug(protocol, id):
    # non-recursive preorder
    myStack = []
    node = protocol
    while node or myStack:
        while node:
            # function here
            if node.id == id:
                print(node.lr)
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    return


def protocol_swap_branches(protocol, id_1, id_2):
    """
    Function that swaps to branches of a protocol based on identification labels

    Parameters
    ----------
    protocol : binary tree with purification and distillation operations
    id_1 : nonnegative integer
        id label of the first branch that is swapped
    id_2 : nonnegative integer
        id label of the second branch that is swapped

    Returns
    -------
    protocol : binary tree with purification and distillation operations
    """
    # non-recursive preorder
    if protocol == None:
        return
    # id and save
    myStack = []
    node = protocol
    while node or myStack:
        while node:
            # function here
            if node.id == id_1:
                nodesave1 = copy.deepcopy(node)
            elif node.id == id_2:
                nodesave2 = copy.deepcopy(node)
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    # swap id1 id2
    myStack = []
    node = protocol
    while node or myStack:
        while node:
            # function here
            if node.id == id_1:
                node.left = copy.deepcopy(nodesave2.left)
                node.right = copy.deepcopy(nodesave2.right)
                node.value = copy.deepcopy(nodesave2.value)
                node.id = id_1
                node.lr = copy.deepcopy(nodesave2.lr)
                # print("test1")
            elif node.id == id_2:
                node.left = copy.deepcopy(nodesave1.left)
                node.right = copy.deepcopy(nodesave1.right)
                node.value = copy.deepcopy(nodesave1.value)
                node.id = id_2
                node.lr = copy.deepcopy(nodesave2.lr)
                # print("test2")
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    # change root1
    myStack = []
    node = protocol
    flag = 0
    while node or myStack:
        while node:
            # function here
            if node.id == id_1:
                while node.root:
                    if node.lr == 0:  # left
                        node.root.value.k = node.value.k + node.root.right.value.k
                    elif node.lr == 1:  # right
                        node.root.value.k = node.value.k + node.root.left.value.k
                        node.root.value.k2 = node.value.k2
                    node = node.root
                flag = 1
                break
            myStack.append(node)
            node = node.left
        if flag == 1:
            break
        node = myStack.pop()
        node = node.right
    # change root2
    myStack = []
    node = protocol
    flag = 0
    while node or myStack:
        while node:
            # function here
            if node.id == id_2:
                while node.root:
                    if node.lr == 0:  # left
                        node.root.value.k = node.value.k + node.root.right.value.k
                    elif node.lr == 1:  # right
                        node.root.value.k = node.value.k + node.root.left.value.k
                        node.root.value.k2 = node.value.k2
                    node = node.root
                flag = 1
                break
            myStack.append(node)
            node = node.left
        if flag == 1:
            break
        node = myStack.pop()
        node = node.right
    return protocol


def operate_protocol(protocol, ntype, F):
    """
    Function that allows one to execute a binary tree protocol created by `identify_protocol' for the requested
    isotropic Bell diagonal state fidelity F.

    Parameters
    ----------
    protocol : binary tree with purification and distillation operations
    ntype : positive integer
        number of protocols stored per value of n and k in the search process where the protocol was found
    F : float between 0 and 1
        Describes fidelity of isotropic Bell diagonal states used in the protocol

    Returns
    -------
    final_state : one-dimensional numpy object of length 2**n
        Here, n is the size of the n-qubit GHZ diagonal state created by the protocol
    """
    # create storage
    sparse_data = np.empty((protocol.value.n + 1, protocol.value.k + 1, ntype), dtype=object)
    for n in range(protocol.value.n + 1):
        for k in range(protocol.value.k + 1):
            for t in range(ntype):
                sparse_data[n][k][t] = None
    # preorder, tree fill sparse_data
    myStack = []
    node = protocol
    while node or myStack:
        while node:
            # if node.value.t == 1:
            #     print("Yes")
            sparse_data[node.value.n][node.value.k][node.value.t] = node.value
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    # operation
    for n in range(protocol.value.n + 1):
        for k in range(protocol.value.k + 1):
            for t in range(ntype):
                if sparse_data[n][k][t] != None:
                    if (n == 2) & (k == 1):
                        sparse_data[n][k][t].state = op.set_isotropic_state(F, 2)
                    else:
                        if sparse_data[n][k][t].p_or_f == 0:  # purification n=n1 k=k1+k2 n,k,n2,k2 known
                            n1 = n
                            n2 = sparse_data[n][k][t].n2
                            k1 = k - sparse_data[n][k][t].k2
                            k2 = sparse_data[n][k][t].k2
                            t1 = sparse_data[n][k][t].t1
                            t2 = sparse_data[n][k][t].t2
                            r2 = sparse_data[n][k][t].r2
                            dec = sparse_data[n][k][t].dec
                            # if t1 == 1:
                            #     print("Yes")
                            # if sparse_data[n1][k1][t1] is None:
                            #     print(n, k, t)
                            #     print(n1, k1, t1)
                            #     print(F)
                            sparse_data[n][k][t].state = \
                                op.purification(sparse_data[n1][k1][t1].state,
                                                ar.ancilla_rotation(sparse_data[n2][k2][t2].state, r2), dec)
                        else:  # fusion n=n1+n2-1 k=k1+k2 n,k,n2,k2 known
                            n1 = n + 1 - sparse_data[n][k][t].n2
                            n2 = sparse_data[n][k][t].n2
                            k1 = k - sparse_data[n][k][t].k2
                            k2 = sparse_data[n][k][t].k2
                            t1 = sparse_data[n][k][t].t1
                            t2 = sparse_data[n][k][t].t2
                            r1 = sparse_data[n][k][t].r1
                            r2 = sparse_data[n][k][t].r2
                            i = sparse_data[n][k][t].i
                            j = sparse_data[n][k][t].j
                            sparse_data[n][k][t].state = \
                                op.fuse_GHZ_local(ar.ancilla_rotation(sparse_data[n1][k1][t1].state, r1),
                                                  ar.ancilla_rotation(sparse_data[n2][k2][t2].state, r2), i, j)
    final_state = sparse_data[protocol.value.n][protocol.value.k][protocol.value.t].state
    return final_state


def operate_protocol_no_data(protocol, F):
    nmax = protocol.value.n
    kmax = protocol.value.k
    # change state
    myStack = []
    node = protocol
    while node or myStack:
        while node:
            # function here
            if node.value.n == 2 and node.value.k == 1:
                node.value.state = op.set_isotropic_state(F, 2)
            else:
                node.value.state = op.set_isotropic_state(0, node.value.n)
            myStack.append(node)
            node = node.left
        node = myStack.pop()
        node = node.right
    #
    myStack = []
    node = protocol
    flag = 0
    while node or myStack:
        if flag == 1:
            break
        while node:
            # function here
            print(node.value.n, node.value.k, node.id)
            if node.left.value.state[0] != 0:
                if node.right.value.state[0] != 0:
                    if node.value.p_or_f == 0:
                        node.value.state = op.purification(node.left.value.state,
                                                           ar.ancilla_rotation(node.right.value.state, node.value.r2),
                                                           node.value.dec)
                    else:
                        node.value.state = op.fuse_GHZ_local(ar.ancilla_rotation(node.left.value.state, node.value.r1),
                                                             ar.ancilla_rotation(node.right.value.state, node.value.r2),
                                                             node.value.i, node.value.j)
                    if not (node.value.n == nmax and node.value.k == kmax and node.left.value.state[0] != 0 and node.right.value.state[0] != 0):
                        node = myStack.pop()
                    else:
                        flag = 1
                        break
                else:
                    myStack.append(node)
                    node = node.right
            else:
                myStack.append(node)
                node = node.left
    final_state = protocol.value.state
    return final_state
