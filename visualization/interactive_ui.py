import pandas as pd
from bokeh.models import DatePicker
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import Div, ColumnDataSource, HoverTool, SingleIntervalTicker, LinearAxis, Range1d, CustomJS, \
	TextInput, Button, RadioButtonGroup
from bokeh.io import show
from bokeh.palettes import GnBu3, OrRd3
from bokeh.models.annotations import Title
import random
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

title = Div(text="""<b>Classification of scientific software project</b><br>There are two parts in this 
visualization page<br> (1) The functional classifier<br>(2) The visualization of the improvement of the other four 
classifiers<br>""", width=1000, height=100)

div_functional_classifier = Div(text="""To use the functional classifier, you need to input a target url which could be 
a repository of a scientific software or the url for its readme file. Our model will crawl the text from the website 
and then use them to predict which type of the software it is as well as the corresponding possibility for 
each type""", width=1000, height=100)

# pie chart
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

# classifier results
best_citation = {'accuracy': 0.92, 'precision': 0.89, 'recall': 0.98, 'f': 0.93}
best_description = {'accuracy': 0.83, 'precision': 0.85, 'recall': 0.79, 'f': 0.82}
best_installation = {'accuracy': 0.9, 'precision': 0.92, 'recall': 0.9, 'f': 0.91}
best_invocation = {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.94, 'f': 0.9}
citation_accu = np.load('./data/citation_accu.npy')
citation_prec = np.load('./data/citation_prec.npy')
citation_recall = np.load('./data/citation_recall.npy')
citation_f = np.load('./data/citation_f.npy')
citation_accu = np.nan_to_num(citation_accu, nan=0.0)
citation_prec = np.nan_to_num(citation_prec, nan=0.0)
citation_recall = np.nan_to_num(citation_recall, nan=0.0)
citation_f = np.nan_to_num(citation_f, nan=0.0)

description_accu = np.load('./data/description_accu.npy')
description_prec = np.load('./data/description_prec.npy')
description_recall = np.load('./data/description_recall.npy')
description_f = np.load('./data/description_f.npy')
description_accu = np.nan_to_num(description_accu, nan=0.0)
description_prec = np.nan_to_num(description_prec, nan=0.0)
description_recall = np.nan_to_num(description_recall, nan=0.0)
description_f = np.nan_to_num(description_f, nan=0.0)

installation_accu = np.load('./data/installation_accu.npy')
installation_prec = np.load('./data/installation_prec.npy')
installation_recall = np.load('./data/installation_recall.npy')
installation_f = np.load('./data/installation_f.npy')
installation_accu = np.nan_to_num(installation_accu, nan=0.0)
installation_prec = np.nan_to_num(installation_prec, nan=0.0)
installation_recall = np.nan_to_num(installation_recall, nan=0.0)
installation_f = np.nan_to_num(installation_f, nan=0.0)

invocation_accu = np.load('./data/invocation_accu.npy')
invocation_prec = np.load('./data/invocation_prec.npy')
invocation_recall = np.load('./data/invocation_recall.npy')
invocation_f = np.load('./data/invocation_f.npy')
invocation_accu = np.nan_to_num(invocation_accu, nan=0.0)
invocation_prec = np.nan_to_num(invocation_prec, nan=0.0)
invocation_recall = np.nan_to_num(invocation_recall, nan=0.0)
invocation_f = np.nan_to_num(invocation_f, nan=0.0)

radio_button_group = RadioButtonGroup(labels=['Citation', 'Description', 'Installation', 'Invocation'], active=0)


source_line_accu1 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[2,2,2],
))
source_line_accu2 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[1,1,1],
))
p_accu = figure(title='training(testing) accuracy for the selected binary classifier', x_axis_label='epochs',
                y_axis_label='accuracy', tools="hover", tooltips="After @x epochs, accuracy: @y", height=300)
p_accu.line(source=source_line_accu1, legend_label='train', color='green')
p_accu.line(source=source_line_accu2, legend_label='best', color='red')

source_line_prec1 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[2,2,2],
))
source_line_prec2 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[1,1,1],
))
p_prec = figure(title='training(testing) precision for the selected binary classifier', x_axis_label='epochs',
                y_axis_label='precision', tools="hover", tooltips="After @x epochs, accuracy: @y", height=300)
p_prec.line(source=source_line_prec1, legend_label='train', color='green')
p_prec.line(source=source_line_prec2, legend_label='best', color='red')

source_line_recall1 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[2,2,2],
))
source_line_recall2 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[1,1,1],
))
p_recall = figure(title='training(testing) recall for the selected binary classifier', x_axis_label='epochs',
                  y_axis_label='recall', tools="hover", tooltips="After @x epochs, accuracy: @y", height=300)
p_recall.line(source=source_line_recall1, legend_label='train', color='green')
p_recall.line(source=source_line_recall2, legend_label='best', color='red')

source_line_f1 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[2,2,2],
))
source_line_f2 = ColumnDataSource(data=dict(
	x=[1,2,3],
	y=[1,1,1],
))
p_f = figure(title='training(testing) F_measure for the selected binary classifier', x_axis_label='epochs',
             y_axis_label='F_measure', tools="hover", tooltips="After @x epochs, accuracy: @y", height=300)
p_f.line(source=source_line_f1, legend_label='train', color='green')
p_f.line(source=source_line_f2, legend_label='best', color='red')

new_accu1 = dict(x=list(range(len(list(citation_accu)))), y=list(citation_accu/100))
new_accu2 = dict(x=list(range(len(list(citation_accu)))), y=[best_citation['accuracy'] for i in range(len(list(citation_accu)))])
source_line_accu1.data = new_accu1
source_line_accu2.data = new_accu2

new_prec1 = dict(x=list(range(len(list(citation_prec)))), y=list(citation_prec/100))
new_prec2 = dict(x=list(range(len(list(citation_prec)))),
                 y=[best_citation['precision'] for i in range(len(list(citation_prec)))])
source_line_prec1.data = new_prec1
source_line_prec2.data = new_prec2

new_recall1 = dict(x=list(range(len(list(citation_recall)))), y=list(citation_recall/100))
new_recall2 = dict(x=list(range(len(list(citation_recall)))),
                 y=[best_citation['recall'] for i in range(len(list(citation_recall)))])
source_line_recall1.data = new_recall1
source_line_recall2.data = new_recall2

new_f1 = dict(x=list(range(len(list(citation_f)))), y=list(citation_f/100))
new_f2 = dict(x=list(range(len(list(citation_f)))),
                 y=[best_citation['f'] for i in range(len(list(citation_f)))])
source_line_f1.data = new_f1
source_line_f2.data = new_f2


def radio_button_callback(new):
	which_classifier = radio_button_group.active
	print(radio_button_group.active)
	if which_classifier == 0:
		new_accu1 = dict(x=list(range(len(list(citation_accu)))), y=list(citation_accu/100))
		new_accu2 = dict(x=list(range(len(list(citation_accu)))), y=[best_citation['accuracy'] for i in range(len(list(citation_accu)))])
		source_line_accu1.data = new_accu1
		source_line_accu2.data = new_accu2

		new_prec1 = dict(x=list(range(len(list(citation_prec)))), y=list(citation_prec/100))
		new_prec2 = dict(x=list(range(len(list(citation_prec)))),
		                 y=[best_citation['precision'] for i in range(len(list(citation_prec)))])
		source_line_prec1.data = new_prec1
		source_line_prec2.data = new_prec2

		new_recall1 = dict(x=list(range(len(list(citation_recall)))), y=list(citation_recall/100))
		new_recall2 = dict(x=list(range(len(list(citation_recall)))),
		                 y=[best_citation['recall'] for i in range(len(list(citation_recall)))])
		source_line_recall1.data = new_recall1
		source_line_recall2.data = new_recall2

		new_f1 = dict(x=list(range(len(list(citation_f)))), y=list(citation_f/100))
		new_f2 = dict(x=list(range(len(list(citation_f)))),
		                 y=[best_citation['f'] for i in range(len(list(citation_f)))])
		source_line_f1.data = new_f1
		source_line_f2.data = new_f2
		pass
	elif which_classifier == 1:
		new_accu1 = dict(x=list(range(len(list(description_accu)))), y=list(description_accu / 100))
		new_accu2 = dict(x=list(range(len(list(description_accu)))),
		                 y=[best_description['accuracy'] for i in range(len(list(description_accu)))])
		source_line_accu1.data = new_accu1
		source_line_accu2.data = new_accu2

		new_prec1 = dict(x=list(range(len(list(description_prec)))), y=list(description_prec / 100))
		new_prec2 = dict(x=list(range(len(list(description_prec)))),
		                 y=[best_description['precision'] for i in range(len(list(description_prec)))])
		source_line_prec1.data = new_prec1
		source_line_prec2.data = new_prec2

		new_recall1 = dict(x=list(range(len(list(description_recall)))), y=list(description_recall / 100))
		new_recall2 = dict(x=list(range(len(list(description_recall)))),
		                   y=[best_description['recall'] for i in range(len(list(description_recall)))])
		source_line_recall1.data = new_recall1
		source_line_recall2.data = new_recall2

		new_f1 = dict(x=list(range(len(list(description_f)))), y=list(description_f / 100))
		new_f2 = dict(x=list(range(len(list(description_f)))),
		              y=[best_description['f'] for i in range(len(list(description_f)))])
		source_line_f1.data = new_f1
		source_line_f2.data = new_f2
	elif which_classifier == 2:
		new_accu1 = dict(x=list(range(len(list(installation_accu)))), y=list(installation_accu / 100))
		new_accu2 = dict(x=list(range(len(list(installation_accu)))),
		                 y=[best_installation['accuracy'] for i in range(len(list(installation_accu)))])
		source_line_accu1.data = new_accu1
		source_line_accu2.data = new_accu2

		new_prec1 = dict(x=list(range(len(list(installation_prec)))), y=list(installation_prec / 100))
		new_prec2 = dict(x=list(range(len(list(installation_prec)))),
		                 y=[best_installation['precision'] for i in range(len(list(installation_prec)))])
		source_line_prec1.data = new_prec1
		source_line_prec2.data = new_prec2

		new_recall1 = dict(x=list(range(len(list(installation_recall)))), y=list(installation_recall / 100))
		new_recall2 = dict(x=list(range(len(list(installation_recall)))),
		                   y=[best_installation['recall'] for i in range(len(list(installation_recall)))])
		source_line_recall1.data = new_recall1
		source_line_recall2.data = new_recall2

		new_f1 = dict(x=list(range(len(list(installation_f)))), y=list(installation_f / 100))
		new_f2 = dict(x=list(range(len(list(installation_f)))),
		              y=[best_installation['f'] for i in range(len(list(installation_f)))])
		source_line_f1.data = new_f1
		source_line_f2.data = new_f2
	elif which_classifier == 3:
		new_accu1 = dict(x=list(range(len(list(invocation_accu)))), y=list(invocation_accu / 100))
		new_accu2 = dict(x=list(range(len(list(invocation_accu)))),
		                 y=[best_invocation['accuracy'] for i in range(len(list(invocation_accu)))])
		source_line_accu1.data = new_accu1
		source_line_accu2.data = new_accu2

		new_prec1 = dict(x=list(range(len(list(invocation_prec)))), y=list(invocation_prec / 100))
		new_prec2 = dict(x=list(range(len(list(invocation_prec)))),
		                 y=[best_invocation['precision'] for i in range(len(list(invocation_prec)))])
		source_line_prec1.data = new_prec1
		source_line_prec2.data = new_prec2

		new_recall1 = dict(x=list(range(len(list(invocation_recall)))), y=list(invocation_recall / 100))
		new_recall2 = dict(x=list(range(len(list(invocation_recall)))),
		                   y=[best_invocation['recall'] for i in range(len(list(invocation_recall)))])
		source_line_recall1.data = new_recall1
		source_line_recall2.data = new_recall2

		new_f1 = dict(x=list(range(len(list(invocation_f)))), y=list(invocation_f / 100))
		new_f2 = dict(x=list(range(len(list(invocation_f)))),
		              y=[best_invocation['f'] for i in range(len(list(invocation_f)))])
		source_line_f1.data = new_f1
		source_line_f2.data = new_f2


radio_button_group.on_click(radio_button_callback)
# layout
curdoc().add_root(column(title, div_functional_classifier, text_input, button, p, div_classifier_results,
                         radio_button_group, row(p_accu, p_prec), row(p_recall, p_f)))
