import pandas as pd
from bokeh.models import DatePicker
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import Div, ColumnDataSource, HoverTool, SingleIntervalTicker, LinearAxis, Range1d, CustomJS, TextInput
from bokeh.io import show
from bokeh.palettes import GnBu3, OrRd3
from bokeh.models.annotations import Title
from bokeh.io import show
import random
import pandas as pd
import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import re
import urllib
from bs4 import BeautifulSoup
from functional_classifier_module import *

title = Div(text="""<b>Classification of scientific software project</b><br>There are two parts in this 
visualization page<br> (1) The functional classifier<br>(2) The visualization of the improvement of the other four 
classifiers<br>""", width=1000, height=100)

div_functional_classifier = Div(text="""To use the functional classifier, you need to input a target url which could be 
a repository of a scientific software or the url for its readme file. Our model will crawl the text from the website 
and then use them to predict which type of the software it is as well as the corresponding possibility for 
each type""", width=1000, height=100)

text_input = TextInput(value="e.g. https://github.com/alvinzhou66/classification-of-scientific-software",
                       title="Enter the project url for functional classifier here")
text_input.on_change("value", lambda x, y, z: print(text_input.value))


def func_classifier_callback(attr, old, new):
	print(text_input.value)


div_classifier_results = Div(text="""We also improve the four existing classifiers, we will show the statistics 
accuracy, precision, recall and F_measure with respect to epochs. In addition we will provide the existing results 
for comparison.""", width=1000, height=100)

curdoc().add_root(column(title, div_functional_classifier, text_input, div_classifier_results))
