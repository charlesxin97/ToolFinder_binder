# ToolFinder [![DOI](https://zenodo.org/badge/309178983.svg)](https://zenodo.org/badge/latestdoi/309178983)
 Software Metadata Classification Project, for study purpose at USC | DSCI560.
 
 <b>Team members</b>: Xihao Zhou, Ruohan Gao, Gan Xin, Hao Yang, Yifan Li, Dongsheng Yang
## Documentation
https://alvinzhou66.github.io/ToolFinder/
## Citation and Dataset
All datasets we used for this project are in /dataset folder.


## Installation (If you want to retrain our model)
To run the scripts in the project, you can either use the requirements.txt to setup your environment locally or you can manually build the same environment on your local machine, or use <i>our Dockerfile</i> to build a docker container.
1. Virtual environment(Need python 3.7.x).  
Firstly create your virtual env and activate it
```
python3 -m venv your_venv_name
. ./your_venv_name/bin/activate
```
Then use pip to install the packages
```
pip install -r requirements.txt
```
2. Local (make sure you have python 3.7.x).  
Download Zip or clone my reporsitory.
```
git@github.com:alvinzhou66/ToolFinder.git
```
Move into the Repo and install the packages using the requirements.txt file.
```
pip install -r requirements.txt
```
3. Docker.  
Install Docker first.  

In the directory which has <i>our Dockerfile (in folder -> /Docker_workspace)</i>, build the docker container:
```
docker build -t coss .
```
Run it
```
docker run -p 5006:5006 -it coss
```
<b>Please make sure your docker server is up-to-date!!!</b>
We failed to build docker container and receive the following error on some machines, it says we don't have this file in our REPO, but we do! So please clean your docker image history and data or update it to the latest version, it may solve this problem.
![image](/images/1.png)
--------------------------------------

For <b>binary classifiers</b>, just run the 4 ipynb script in "/binary_classifier" folder.

For <b>functional classifier</b>, move to "/functional_classifier" and run des_fuc.ipynb first, then run func_class.ipynb.
## Usage (interactive visualization)
1. Functional classifier. 
An interactive Bokeh visualization which can handle URL inputs(any URL with description, don't need to be .md file), return function prediction result. Also, visualize our training result and compaire with SOMEF.
<b>After finishing the installation of the virtual environment or docker container (as shown in above), you can activate the virtual environment and use that for running the visualization.</b>  
```
. ./your_venv_name/bin/activate
```
You need to go to folder of our repository locally and cd into the directory of visualization, and start the bokeh server application.   
```
cd visualization
bokeh serve --show interactive_ui.py
```
Then go to your <b>localhost:5006</b> port to see the visualization result.

To use the functional classifier, you need to input the url into the box and click the predict button. Then the result will show in the pie chart, which contains the probabilities of your input project being different type of scientific software. The result may show after several seconds due to crawling the website and the inference of the model.  
![image](/images/pie.png) 

2. Binary classifiers.  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alvinzhou66/ToolFinder/main?filepath=%2Fbinary_classifier%2FSOMEF_BIN_classifier.ipynb) 
 
To use the binary classifiers, you can use the <i>binder badge</i> above, or you need to first:  
```
cd binary_classifier
```
Then run "SOMEF_BIN_classifier.ipynb" to use this Jupyter Notebook to see the result.
![image](/images/line.png)