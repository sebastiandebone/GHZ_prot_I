"""
2020 Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
_____________________________________________
"""
import random


def av_calc(pb, ts, fr, fl, nr_iter, print_results=False):
    """
    Function that allows one to execute a binary tree protocol created by `identify_protocol' for the requested
    isotropic Bell diagonal state fidelity F.

    Parameters
    ----------
    pb : structure, with probabilities per sublevel
    ts : structure with same dimensions as pb, with time steps required to complete each (sub)level
    fr : one-dimensional array of length len(pb), with failure reset level of each of the main levels
    fl : structure with same dimensions as pb, with failure reset levels of each of the individual sublevels

    These objects have, for example, the following structure
    pb = [[[p0, p1], [p2]],  [[p3], [p4], [p5]], [[p6]]]
    ts = [[[t0, t1], [t2]],  [[t3], [t4], [t5]], [[t6]]]
    fr = [l0                 l1,                 l2]
    fl = [[[s0, s1], [s2]],  [[s3], [s4], [s5]], [[s6]]]
    In this case, this means that the sublevel associated with p0+p1 is carried out in parallel with the sublevel
    indicated with p2 (with p0 and p1 carried out in series), and the one that takes the longest is included in the
    total time evaluation. When both the sublevels are completed, the protocol carries on with three sublevels p3, p4
    and p5 all carried out in parallel. Finally, p6 is carried out by itself. When any sublevel fails, and the
    corresponding value l0, l1 or l2 of that level is a different level from where this sublevel is, the protocol is
    reset to the corresponding value l0, l1 or l2. If this failure reset level is the same level as the where the
    protocol is, the protocol is reset to the corresponding failure reset sublevel indicated in the structure fl.

    Returns
    -------
    nts_bins : array of integers
        array of length nr_iter that contains the number of times steps required to complete each of the Monte Carlo
        iterations of the protocol
    """
    nts_bins = [None]*nr_iter               # Storage for the total time for each Monte Carlo iteration
    n_levels = 0
    for i in range(len(pb)):
        for j in range(len(pb[i])):
            for k in range(len(pb[i][j])):
                n_levels = n_levels + 1     # Here we calculate the number of levels and sublevels

    for i_n in range(nr_iter):              # Loop over the requested number of Monte Carlo iterations

        nts_tot = 0
        s = [None] * len(pb)                        # Array indicating whether the levels have succeeded
        nts = [None] * len(pb)                      # Total time spend on each of the levels: this is a temporary bin
        for i in range(len(pb)):
            s[i] = [None] * len(pb[i])              # We place dummies just to capture the structure of s and nts
            nts[i] = [None] * len(pb[i])
            for j in range(len(pb[i])):
                s[i][j] = [False] * len(pb[i][j])   # We reset s such that all levels are false again
                nts[i][j] = [0]*len(pb[i][j])       # And nts such that all levels are zero again
        allS = [False] * n_levels                   # The array version of s is set to false for all levels again

        while all(allS) is False:           # We continue until all elements of s are True
            contfor = True                  # This parameter determines whether or not we should start at i = 0 again.
            for i in range(len(pb)):                                # We loop over all levels and sublevels
                for j in range(len(pb[i])):
                    for k in range(len(pb[i][j])):
                        if s[i][j][k] is False and contfor:         # If s[i][j][k] is False, we throw a dice, otherwise
                                                                    # we continue to the next level
                            nts[i][j][k] = nts[i][j][k] + ts[i][j][k]   # We add the number of time steps for this level
                            if random.random() <= pb[i][j][k]:      # Compare if a random number between 0 and 1 is
                                                                    # equal than or smaller than pb[i][j][k]
                                s[i][j][k] = True                   # If this is the case the level has succeeded
                                if print_results:
                                    print("level [{},{},{}] succeeded.".format(i, j, k))
                                if j == len(pb[i]) - 1 and k == len(pb[i][j]) - 1:  # If we reached the end of a
                                    bins_j = [0]*len(pb[i])                         # sublevel, we empty the nts bin
                                    for j5 in range(len(pb[i])):                    # and add the maximum of all sub-
                                        bins_j[j5] = sum(nts[i][j5])                # levels to the number of time
                                    nts_tot = nts_tot + max(bins_j)                 # steps used
                                    if print_results and max(bins_j) > 0:
                                        print("added {} time steps.".format(max(bins_j)))
                                    for j6 in range(len(pb[i])):
                                        for k6 in range(len(pb[i][j6])):
                                            nts[i][j6][k6] = 0      # And we set the time step bins to zero again
                            else:                                   # If the random number is bigger, the level failed
                                for i2 in range(fr[i], i):          # All levels between this level and the failure
                                    for j2 in range(len(pb[i2])):   # reset level are set to zero again
                                        for k2 in range(len(pb[i2][j2])):
                                            s[i2][j2][k2] = False
                                if i > fr[i]:                       # If the FRL is not the current level, all parallel
                                    for j3 in range(j):             # are reset as well.
                                        for k3 in range(len(pb[i][j3])):
                                            s[i][j3][k3] = False
                                for k4 in range(fl[i][j][k], k):    # All previous events in this branch of the sublevel
                                    s[i][j][k4] = False             # are reset
                                if print_results:
                                    print("level [{},{},{}] failed.".format(i, j, k))
                                contfor = False     # We (effectively) break the for loop and start over at i=0
                                                    # (the first level). In the new loop all levels that still have
                                                    # s[i][j][k] = True are skipped, because they don't have to be
                                                    # done again.
            n_levels = 0
            for i in range(len(pb)):
                for j in range(len(pb[i])):
                    for k in range(len(pb[i][j])):
                        allS[n_levels] = s[i][j][k] # Here w convert the s structure to an array such that the while
                        n_levels = n_levels + 1     # loop understands how to read it.

        nts_bins[i_n] = nts_tot + 2
        if print_results:
            print("reached the end in {} time steps.\n".format(nts_tot + 2))

    return nts_bins     # For each Monte Carlo iteration we return the number of time steps
