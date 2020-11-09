import pandas as pd
from bokeh.models import DatePicker
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import Div, ColumnDataSource, HoverTool, SingleIntervalTicker, LinearAxis, Range1d, CustomJS, \
	TextInput, Button
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
from math import pi
from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum

title = Div(text="""<b>Classification of scientific software project</b><br>There are two parts in this 
visualization page<br> (1) The functional classifier<br>(2) The visualization of the improvement of the other four 
classifiers<br>""", width=1000, height=100)

div_functional_classifier = Div(text="""To use the functional classifier, you need to input a target url which could be 
a repository of a scientific software or the url for its readme file. Our model will crawl the text from the website 
and then use them to predict which type of the software it is as well as the corresponding possibility for 
each type""", width=1000, height=100)

# x = {
# 	'Data Analysis': 0.33,
# 	'Deep Learning': 0.33,
# 	'Data Management': 0.33
# }

source_pie = ColumnDataSource(data=dict(
	start=[0, 0.33 * 2 * pi, 0.66 * 2 * pi], end=[0.33 * 2 * pi, 0.66 * 2 * pi, 2 * pi],
	name=['Probability of Data Analysis', 'Probability of Deep Learning', 'Probability of Data '
	                                                                      'Management'],
	color=Category20c[3], value=[0.33, 0.33, 0.33]
))

p = figure(plot_height=350, title="Probability of being each functional type for the project (changes may take seconds"
                                  ")", toolbar_location=None,
           tools="hover", tooltips="@name: @value", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4,
        start_angle='start', end_angle='end',
        line_color="white", fill_color='color', legend_field='name', source=source_pie)

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

software_type = ''
prob_dict = {}


def func_classifier_callback():
	url = text_input.value
	global software_type, prob_dict
	software_type, prob_dict = functional_predict(url)
	print(software_type, prob_dict)
	new_dict = dict(name=['Probability of Data Analysis', 'Probability of Deep Learning', 'Probability of '
	                                                                                      'Data '
	                                                                                      'Management'],
	                color=Category20c[3])
	start = [0, prob_dict['Data Analysis'] * 2 * pi,
	         prob_dict['Data Analysis'] * 2 * pi + prob_dict['Deep Learning'] * 2 * pi]
	end = [prob_dict['Data Analysis'] * 2 * pi,
	       prob_dict['Data Analysis'] * 2 * pi + prob_dict['Deep Learning'] * 2 * pi, 2 * pi]
	value = [prob_dict['Data Analysis'], prob_dict['Deep Learning'], prob_dict['Data Management']]
	new_dict['start'] = start
	new_dict['end'] = end
	new_dict['value'] = value
	source_pie.data = new_dict


button = Button(label="predict", button_type="success")
text_input = TextInput(value="https://github.com/alvinzhou66/classification-of-scientific-software",
                       title="Enter the project url for functional classifier here, for example:")

button.on_click(func_classifier_callback)
div_classifier_results = Div(text="""We also improve the four existing classifiers, we will show the statistics 
accuracy, precision, recall and F_measure with respect to epochs. In addition we will provide the existing results 
for comparison.""", width=1000, height=100)

curdoc().add_root(column(title, div_functional_classifier, text_input, button, p, div_classifier_results))
