# Project: Learned proximal networks for solving certain (non-) convex high-dimensional Hamilton--Jacobi Equations and optimal control problems

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the nascent implementation document of our project on LPNs for non-convex optimal control. This work is based on the paper: [What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r) 

--------------------

## Purpose
This project aims to apply LPNs to solve a broad class of non-convex Hamilton--Jacobi PDEs for optimal control. 

- This Github repo was forked from the Sulam-Group on Github.

- This README file is in construction and will be updated as the project moves forward.

- Provided below is a set of instructions to install the LPN package and run the examples contained in the original Repo of the Sulam-Group.


## Installation
The code is implemented with Python 3.9.16 and PyTorch 1.12.0. Install the conda environment by

```
conda env create -f environment.yml
```

Install the `lpn` package

```
pip install -e .
```

## Datasets

There are no datasets placed yet in `data/` folder.


## How to Run the Code

Code of the main functionalities of LPN is placed in the `lpn` folder.

Code for repoducing the experiments in the paper is placed in the `exps` folder. There are five python notebooks 'exp_1_minplus_XD.ipynb', where
X = 2,4,8,16,32. There is also a subdirectory called `exps/old_laplacian_experiment` that contains the old 1D Laplacian experiment of the original
LPN report.

### Min-plus HJ PDEs experiments

Simply run the notebooks. There is documentation in the notebooks as well. It will take a while for the higher dimensional cases.


## Notes and References

- See the overleaf document for more information.

The citation below is for the original LPN paper
```bib
@inproceedings{
    fang2024whats,
    title={What's in a Prior? Learned Proximal Networks for Inverse Problems},
    author={Zhenghan Fang and Sam Buchanan and Jeremias Sulam},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```