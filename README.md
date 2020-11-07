# Classification of Scientific Software
 The course project from group 3 at USC DSCI560.
## Usage
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
In the directory, build the docker container:
```
docker build -t pulkit/capturing:1.0 .
```
Run it
```
docker run -ti --name capturing pulkit/capturing:1.0
```
Space holder
