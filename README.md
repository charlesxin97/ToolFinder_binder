## Classification of Scientific Software
 Software Metadata Classification Project, for study purpose at USC | DSCI560.
 
 Team members: Xihao Zhou, Ruohan Gao, Gan Xin, Hao Yang, Yifan Li, Dongsheng Yang
## Documentation
[Placeholder for documentation link(github.page)]
## Citation and Dataset
All datasets we used for this project are in /dataset folder.
[![DOI](https://zenodo.org/badge/309178983.svg)](https://zenodo.org/badge/latestdoi/309178983)

## Installation (If you want to retrain our model)
To run the scripts in the project, you can either use the requirements.txt to setup your environment locally or you can use the dockerfile to create a container locally.  
1. Virtual environment(Need python 3.6.x).  
Firstly create your virtual env and activate it
```
python3 -m venv your_venv_name
. ./your_venv_name/bin/activate
```
Then use pip to install the packages
```
pip install -r requirements.txt
```

2. Docker.  
Install Docker first.
In the directory which has <i>our Dockerfile</i>, build the docker container:
```
docker build -t pulkit/capturing:1.0 .
```
Run it
```
docker run -ti --name capturing pulkit/capturing:1.0
```
Space holder
## Usage
An interactive Bokeh visualization which can handle URL inputs(any URL with description, don't need to be .md file), return function prediction result. Also, visualize our training result and compaire with SOMEF.

[Placeholder for steps to use the interface]
