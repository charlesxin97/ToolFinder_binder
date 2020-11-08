## Classification of Scientific Software
 Software Metadata Classification Project, for study purpose at USC | DSCI560.
 
 <b>Team members</b>: Xihao Zhou, Ruohan Gao, Gan Xin, Hao Yang, Yifan Li, Dongsheng Yang
## Documentation
[Placeholder for documentation link(github.page)]
## Citation and Dataset
All datasets we used for this project are in /dataset folder.
[![DOI](https://zenodo.org/badge/309178983.svg)](https://zenodo.org/badge/latestdoi/309178983)

## Installation (If you want to retrain our model)
To run the scripts in the project, you can either use the requirements.txt to setup your environment locally or you can manually build the same environment on your local machine, or use <i>our Dockerfile</i> to build a docker container.
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
2. Local (make sure you have python 3.6.x or 3.7.x).  
Download Zip or clone my reporsitory.
```
git@github.com:alvinzhou66/classification-of-scientific-software.git
```
Move into the Repo and install the packages using the requirements.txt file.
```
pip install -r requirements.txt
```
3. Docker.  
Install Docker first.
In the directory which has <i>our Dockerfile</i>, build the docker container:
```
docker build -t coss
```
Run it
```
docker run -ti --name coss coss
```
--------------------------------------

For <b>binary classifiers</b>, just run the 4 ipynb script in "/binary_classifier" folder.

For <b>functional classifier</b>, move to "/functional_classifier" and run des_fuc.ipynb first, then run func_class.ipynb.
## Usage (interactive visualization)
An interactive Bokeh visualization which can handle URL inputs(any URL with description, don't need to be .md file), return function prediction result. Also, visualize our training result and compaire with SOMEF.

[Placeholder for steps to use the interface]
