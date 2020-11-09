import pandas as pd
from bokeh.models import DatePicker
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import Div, ColumnDataSource, HoverTool, SingleIntervalTicker, LinearAxis, Range1d
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

title = Div(text="""<b>Classification of scientific software project</b><br>There are two parts in this 
visualization page<br> (1) The functional classifier<br>(2) The visualization of the improvement of the other four 
classifiers<br>""", width=1000, height=100)

div_functional_classifier = Div(text=""" """, width=1000, height=100)

div_classifier_results = Div(text=""" """, width=1000, height=100)
