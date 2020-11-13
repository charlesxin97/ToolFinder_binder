FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y \
  gcc

RUN git clone https://github.com/alvinzhou66/classification-of-scientific-software

RUN conda install -c conda-forge yarn

RUN cd classification-of-scientific-software && yarn install

RUN pip install --upgrade pip

COPY requirements.txt /opt/app/requirements.txt

RUN pip install -r /opt/app/requirements.txt

WORKDIR classification-of-scientific-software
