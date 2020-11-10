# ToolFinder

<b>Authors</b>:

Xihao Zhou: xihaozho@usc.edu, [CV](https://drive.google.com/file/d/1yEpapHdKz51QFCS7keB8RNbyjCUqYrxw/view)  
Ruohan Gao: ruohanga@usc.edu, [CV](https://drive.google.com/file/d/1ED3TDFpMZiveP1AULPJFEyZnrXP03x0l/view)  
Gan Xin: gxin@usc.edu, [CV](https://drive.google.com/file/d/18wP5TXjkcd-wG8QZAHv4ryXjECM4f_Fd/view)  
Hao Yang: hyang01@usc.edu, [CV](https://drive.google.com/file/d/1xEe-r8aZ-ZbUzmCEh_GU6n3RTWaZTSlX/view)  
Yifan Li: yli04705@usc.edu, [CV](https://drive.google.com/file/d/1GWmU-6UdR4Eowt9dZ0PV4nmUjKzM4hXh/view)  
Dongsheng Yang: dongshen@usc.edu, [CV](https://drive.google.com/file/d/1GklAEbLkHt-TFZnA_3y7p3dS2MJqTZUg/view)

## Description
The purpose of ToolFinder is to automatically analysis the functions and metadata from online .md files or description websites of scientific software. Thanks to ToolFinder, now scientific researchers and software developers can save a lot of time on reading descriptions and installation steps.

ToolFinder can work with any URL input, this URL doesn't have to be a README.md, any URL with text should be able to use ToolFinder for classification.

For any problems when using ToolFinder, please open an issue on our [GitHub](https://github.com/alvinzhou66/ToolFinder/issues) repository.

## Features
For any given URL link, we can extract the following information:
- <b>Function</b>: Now this project basically faces to data science researchers so it ONLY has "DA-data analysis", "DM-data management" and "DL-deep learning" 3 classes. For any URL input, you will have the predicted probability for these 3 classes.
- <b>Description</b>: All sentences about the description of what this software does.
- <b>Citation</b>: All sentences related to the preferred citation as the authors have stated.
- <b>Installation</b>: All sentences about the installation steps.
- <b>Invocation</b>: All sentences related to the execution commands for this software.

## Installation and usage
See our GitHub [README.md](https://github.com/alvinzhou66/ToolFinder/blob/main/README.md).

## Used Technologies and Standards
### Scikit Learn
Scikit-learn (formerly scikits.learn and also known as sklearn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
### Pytorch
PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license. Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface.
### Beautifulsoup
Beautiful Soup is a Python package for parsing HTML and XML documents (including having malformed markup, i.e. non-closed tags, so named after tag soup). It creates a parse tree for parsed pages that can be used to extract data from HTML, which is useful for web scraping.
