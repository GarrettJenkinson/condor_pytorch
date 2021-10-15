# Installing `condor_pytorch`


## Requirements

Condor requires the following software and packages:

- [Python](https://www.python.org) >= 3.6
- [PyTorch](http://www.pytorch.org) >= 1.5.0


## PyPI

You can install the latest stable release of `condor_pytorch` directly from Python's package index via `pip` by executing the following code from your command line:  

```bash
pip install condor-pytorch
```


## Latest GitHub Source Code

<br>

You want to try out the latest features before they go live on PyPI? Install the `condor_pytorch` dev-version latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/GarrettJenkinson/condor_pytorch.git
```

<br>


Alternatively, you download the package manually from [GitHub](https://github.com/GarrettJenkinson/pytorch_condor) via the [Dowload ZIP](https://github.com/GarrettJenkinson/condor_pytorch/archive/main.zip) button, unzip it, navigate into the package directory, and execute the following command:

```bash
python setup.py install
```

---

## Docker
---

If one does not wish to install things locally, running a docker container
can make it simple to run Condor pytorch.

We provide Dockerfile's to help get up and started quickly with `condor_pytorch`.
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
