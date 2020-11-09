# GHZ_prot_I

This repository contains the dynamic programs for finding GHZ creation protocols described in the paper *Protocols for creating and distilling multipartite GHZ states with Bell pairs* by S. de Bone, R. Ouyang, K. Goodenough and D. Elkouss, and contains functions that produce the figures in this paper.

## Dynamic programs

* `da_search.py`, dynamic program used to search the protocol space;
* `da_protocols.py`, functions that can be used to extract and operate the protocols found by the dynamic program.

There are four versions of the dynamic program included in `da_search.py`: 

* `'sp'`, the standard dynamic program that stores one protocol per value of `n` (the number of parties) and `k` (the number of isotropic Bell pairs used);
* `'mpc'`, the dynamic program that stores multiple protocols per value of `n` and `k`, based on different criteria;
* `'mpF'`, the dynamic program that stores multiple protocols per value of `n` and `k`, based on the highest fidelity;
* `'random'`, the randomized verions of the dynamic algorithm.

The file `da_search.py` also contains functions that allow one to store the calculated data and the found protocols in .txt files (by using the build-in Python library `pickle`), which is sometimes preferable because running the dynamic programs can sometimes take one or more days. Files with this type of data are saved in the `calc_data` folder of the repository. Pre-calculated data used to generate the figures in the paper can be downloaded at https://figshare.com/s/6cd706c723a6eeb5ae8e. 

## Operations

* `operations.py`, operations on Bell diagonal states and GHZ diagonal states;
* `known_protocols.py`, functions with known protocols, as *Expedient* and *Stringent*;
* `ancilla_rotations.py`, functions that perform bilocal Clifford to Bell diagonal and GHZ diagonal states, and by doing so permute the diagonal coefficients of these states (not discussed in the paper).

The functions in `operations.py` are used by the dynamic programs and by the other two files with operations. The `ancilla_rotations.py` are not used to produce the results of  the dynamic programs in the paper, as it turned out that including them in the search did not lead to better end results.

## Requirements
This repository is tested with Python 3.8. There are some additional libraries that needs to be installed, which are:

* `numpy` is used to describe the Bell and GHZ diagonal states (version 1.18.2);
* `matplotlib.pyplot` is used to make the figures of the paper (version 3.3.1);

## Running the dynamic programs
Running the dynamic programs can be done by importing the file `da_search.py`, and calling the function `dynamic_algorithm` in these files. This function typically takes `n` (the maximal number of network parties), `k` (the maximal number of isotropic Bell pairs), `F` (the fidelity of the isotropic Bell pairs), `da_type` (the type of dynamic program, *i.e.*, `sp`, `mpc`, `mpF` or `random`), and some other parameters as input. For example, for the most standard form the of the dynamic program, we can find protocols up to `n_max=4` network parties and `k_max=42` isotropic Bell of fidelity `F=0.9` by calling the function `dynamic_algorithm` in the following way:

```python
import da_search as das
n_max=4
k_max=42
fid=0.9
da_type='sp'
data = das.dynamic_algorithm(n_max, k_max, fid, da_type)
```

The object `data` now contains objects `data[n][k][0]` for all values of `n` and `k` up to `n_max` and `k_max`, with each of these smaller object containing information as how this object is made from smaller objects and what state is produced at this stage of the protocol with isotropic Bell pairs of fidelity `fid=0.9`. This state is found with as the `numpy` object `data[n][k][0].state`, with `data[n][k][0].state[0]` indicating the fidelity of this state. 

Each `data[n][k][0]` object also contains other information, such as from which smaller `data[n][k][0]` objects the protocol is composed, and in which way; *e.g.*, `data[n][k][0].p_or_f` tells you if purification (*i.e.*, distillation) or fusion is used to create `data[n][k][0].state` and `data[n][k][0].n2` and `data[n][k][0].k2` indicate which `data[n2][k2][0]` is used as ancillary state in this process.

The randomized version of the dynamic program can be run in, *e.g.*, the following way:

```python
import da_search as das
n_max=8
k_max=80
fid=0.9
da_type='random'
n_state=200
inc_rot=0
show_or_not=0
seed=3316
temp=0.00001
data_random = das.dynamic_algorithm(n_max, k_max, fid, da_type, n_state, inc_rot, show_or_not, seed, temp)
```

Here, `show_or_not` is a parameter that prints status updates (this parameter can also be included in the other dynamic program functions), `n_state` describes the number of protocols stored per value of `n` and `k`, `inc_rot` indicates whether or not the permutations from `ancilla_rotations.py` need to be included, `seed` is the random seed used for the randomized dynamic program and `temp` is the *temperature* that characterizes the probability distribution for acceptance in the randomized version of the dynamic program. The `200` states and protocols that are stored per value of `n` and `k` can now be reached via `data[n][k][t]`, where `t` runs between `0` and `199`.

## Protocols
From the `data` objects one can extract protocols (which will appear as *binary tree* protocols) by calling on the function `identify_protocol(data, n, k, t)` in `da_protocols.py` (where `t` refers to the protocol number stored per value of `n` and `k`). One can then operate the binary tree protocol found with `identify_protocol` by using the function `operate_protocol(protocol, n_state, fid)` in `da_protocols.py`, where `protocol` is the protocol found with `identify_protocol`, `n_state` describes the number of states included in the search, and `fid` is the fidelity of the isotropic Bell pairs used to operate the protocol.

## Reproducing the figures in the paper
The code that produces the figures can be found in the file `plots_paper.py`. One can reproduce these figures by importing this file and running functions like `figure4()` within this file:

```python
import plots_paper.py as pp
pp.figure4()
```

Depending on whether there are already .txt files with pre-calculated data in the `calc_data` folder, running the necessary dynamic program for these figures can take extremely long. Files with pre-calculated data can be found on https://figshare.com/s/6cd706c723a6eeb5ae8e, and need to be placed in the `calc_data` folder of the repository. After generating, the figures will be saved as .pdf files in the `figures` folder of the repository.
