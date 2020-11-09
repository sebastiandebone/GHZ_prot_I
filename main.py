"""
2020 Runsheng Ouyang, Sebastian de Bone (QuTech)
https://github.com/sebastiandebone/ghz_prot_I
"""
import sys
import os.path
import numpy as np
import operations as op
import da_search as das
import da_protocols as dap
import pickle
import plots_paper as pp
import ancilla_rotations as ar
import math


if __name__ == "__main__":

    # pp.plot_infidelity_with_80_Bell_pairs_for_diff_number_of_parties()
    pp.plot_Bell_pairs_needed_for_diff_number_of_parties()
