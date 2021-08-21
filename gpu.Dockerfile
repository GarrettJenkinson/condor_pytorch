FROM nvcr.io/nvidia/pytorch:21.07-py3

COPY ./ /etc/condor_pytorch/
RUN chmod -R +rwx /etc/condor_pytorch
RUN python3 -m pip install /etc/condor_pytorch

RUN mkdir -p /condor/condor-tutorials && chmod -R a+rwx /condor/
COPY ./docs/tutorials/*.ipynb /condor/condor-tutorials/
