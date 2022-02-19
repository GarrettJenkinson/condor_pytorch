FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image 

RUN python3 -m pip install --no-cache-dir jupyter torch 
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

COPY ./ /etc/condor_pytorch/
RUN chmod -R +rwx /etc/condor_pytorch
RUN python3 -m pip install /etc/condor_pytorch

RUN mkdir -p /condor/condor-tutorials && chmod -R a+rwx /condor/
COPY ./docs/tutorials/*.ipynb /condor/condor-tutorials/

RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get update && apt-get install -y --no-install-recommends wget git
RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /condor
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "jupyter notebook --notebook-dir=/condor --ip 0.0.0.0 --no-browser --allow-root"]
