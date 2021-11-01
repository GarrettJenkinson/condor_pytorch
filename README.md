
<img src="https://github.com/GarrettJenkinson/condor_pytorch/blob/main/docs/img/condor.png" width=300>

**CONDOR pytorch implementation for ordinal regression with deep neural networks.**

![Continuous Integration](https://github.com/GarrettJenkinson/condor_pytorch/actions/workflows/python-test.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GarrettJenkinson/condor_pytorch/blob/master/LICENSE)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

**Documentation: [https://GarrettJenkinson.github.io/condor_pytorch](https://GarrettJenkinson.github.io/condor_pytorch)**

---

## About  

CONDOR, short for CONDitionals for Ordinal Regression, is a method for ordinal regression with deep neural networks, 
which addresses the rank inconsistency issue of other ordinal regression frameworks.

It is compatible with any state-of-the-art deep neural network architecture, requiring only modification of the output layer, the labels, the loss function.

This repository implements the CONDOR functionality (neural network layer, loss function, and dataset utilities) for convenient use. 
Examples are provided via the "Tutorials" that can be found on the documentation website at [https://GarrettJenkinson.github.io/condor_pytorch](https://GarrettJenkinson.github.io/condor_pytorch).

We also have [CONDOR implemented for Tensorflow](https://github.com/GarrettJenkinson/condor_tensorflow).

<img src="https://github.com/GarrettJenkinson/condor_pytorch/raw/main/docs/img/rankconsistent.png" width=500>

---

## Installation or Docker
---

You can install the latest stable release of `condor_pytorch` directly from Python's package index via `pip` by executing the following code from your command line:  

```bash
pip install condor-pytorch
```
We also provide Dockerfile's to help get up and started quickly with `condor_pytorch`.
The cpu image can be built and ran as follows, with tutorial jupyter notebooks
built in.

```bash
# Create a docker image, only done once
docker build -t cpu_pytorch -f cpu.Dockerfile ./

# run image to serve a jupyter notebook
docker run -it -p 8888:8888 --rm cpu_pytorch

# how to run bash inside container (with python that will have deps)
docker run -u $(id -u):$(id -g) -it -p 8888:8888 --rm cpu_pytorch bash
```

An NVIDIA based gpu optimized container can be built and run 
as follows (without interactive ipynb capabilities).

```bash
# only needs to be built once
docker build -t gpu_pytorch -f gpu.Dockerfile ./

# use the image after building it
docker run -it -p 8888:8888 --rm gpu_pytorch
```

## Cite as

If you use CONDOR as part of your workflow in a scientific publication, please consider citing the CONDOR repository with the following DOI:

- [Jenkinson, Khezeli, Oliver, Kalantari, Klee. Universally rank consistent ordinal regression in neural networks, arXiv:2110.07470, 2021.](https://arxiv.org/abs/2110.07470)

```
@article{condor2021,
title = "Universally rank consistent ordinal regression in neural networks",
journal = "arXiv",
volume = "2110.07470",
year = "2021",
url = "https://arxiv.org/abs/2110.07470",
author = "Garrett Jenkinson and Kia Khezeli and Gavin R. Oliver and John Kalantari and Eric W. Klee",
keywords = "Deep learning, Ordinal regression, neural networks, Machine learning, Biometrics"
}
```

**Acknowledgments**: Many thanks to [the CORAL ordinal authors](https://github.com/ck37/coral-ordinal) and 
[the CORAL pytorch authors](https://github.com/Raschka-research-group/coral_pytorch) whose repos provided a roadmap for this codebase.
