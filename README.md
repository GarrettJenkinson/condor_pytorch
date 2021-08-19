
<img src="docs/img/condor.png" width=300>

**CONDOR pytorch implementation for ordinal regression with deep neural networks.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/GarrettJenkinson/condor_pytorch/blob/master/LICENSE)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

**Documentation: [https://GarrettJenkinson.github.io/condor_pytorch](https://GarrettJenkinson.github.io/condor_pytorch)**

---

## About  

CONDOR, short for CONditionals for ORdinal Regression, is a method for ordinal regression with deep neural networks, 
which addresses the rank inconsistency issue of other ordinal regression frameworks.

It is compatible with any state-of-the-art deep neural network architecture, requiring only modification of the output layer, the labels, the loss function.

This repository implements the CONDOR functionality (neural network layer, loss function, and dataset utilities) for convenient use. 
Examples are provided via the "Tutorials" that can be found on the documentation website at [https://GarrettJenkinson.github.io/condor_pytorch](https://GarrettJenkinson.github.io/condor_pytorch).

We also have condor [implemented for Tensorflow](https://github.com/GarrettJenkinson/condor_ordinal).

---

## Cite as

If you use CONDOR as part of your workflow in a scientific publication, please consider citing the CONDOR repository with the following DOI:

- TBD publication

```
@article{condor2021,
title = "Universally rank consistent ordinal regression in neural networks",
journal = "TBD",
volume = "TBD",
pages = "TBD",
year = "TBD",
issn = "TBD",
doi = "TBD",
url = "TBD",
author = "Garrett Jenkinson and Kia Khezeli and Gavin Oliver and John Kalantari and Eric Klee",
keywords = "Deep learning, Ordinal regression, neural networks, Machine learning, Biometrics"
}
```

