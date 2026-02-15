# calibrating-deepire-suppementary-materials
Supplementary materials for reproducing experiments reported in the IJCAR 2026 submission titled "Teaching Vampire New Tricks: An Experimental Study of Neural Clause Selection"

### Vampire

Vampire's modified executable came from the https://github.com/vprover/vampire/tree/mtpa-gnn branch (commit tagged as https://github.com/vprover/vampire/releases/tag/mtpa-gnn-ijcar2026).

It was compiled using the inofficial Makefile path (not via cmake). See https://github.com/vprover/vampire/tree/mtpa-gnn-ijcar2026/Makefile
By mistake, on linux, this Makefile hardwires the path to libtorch. You will need to install libtorch on your own (ideally version 2.5) and update this path there.

### Traning Scritps

Training scripts come from the repo https://github.com/quickbeam123/lawa, branch https://github.com/quickbeam123/lawa/tree/mtpa-gnn, tag https://github.com/quickbeam123/lawa/releases/tag/IJCAR2026

The main entry point is the script `elooper.py` to be run as in `elooper.py <num_loops> <num_cores>`. We used 20-30 loops and ran on servers with 128 available cores using 120 of these for the training.

This scripts looks into `hyperparams.py` for all other configuration.

### Dataset splits


### Hyperparams used by individual experiments


### Continuous Development

Note that since the submission both the vampire branch https://github.com/vprover/vampire/tree/mtpa-gnn and the training scripts branch https://github.com/quickbeam123/lawa/tree/mtpa-gnn may advanced a bit, as new features are added and tried out. It is probably better to try using the most recent version (of both), unless reproducibility is the main concern.

Feel free to let me know (using github issues is an option) should you have any problems running this.
