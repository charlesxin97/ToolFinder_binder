import random
import pandas as pd
import nltk
from nltk.corpus import treebank
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('treebank')
nltk.download('stopwords')
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


class FFN(nn.Module):

	def __init__(self, layer_arch, input_size, output_size, bias=True):
		super(FFN, self).__init__()
		self.layer_arch = layer_arch
		self.input_size = input_size
		self.output_size = output_size
		self.bias = bias
		self.build_model()

	def build_model(self):
		model_arch = []
		unit = self.input_size
		for i, num in enumerate(self.layer_arch):
			model_arch.append(("dense_" + str(i), nn.Linear(unit, num, bias=self.bias)))
			model_arch.append(("nonlinear_" + str(i), nn.ReLU()))
			if (i == 1):
				model_arch.append(("dropout_" + str(i), nn.Dropout()))
			unit = num
		model_arch.append(("dense_final", nn.Linear(unit, self.output_size, bias=self.bias)))
		model_arch.append(("act_final", nn.Sigmoid()))
		self.model = nn.Sequential(OrderedDict(model_arch))

	def forward(self, inputs):
		return self.model(inputs)


def functional_predict(input_url='https://github.com/alvinzhou66/classification-of-scientific-software'):
	url = input_url
	response = urllib.request.urlopen(url)
	page = response.read()
	soup = BeautifulSoup(page, "lxml")
	# kill all script and style elements
	for script in soup(["script", "style"]):
		script.extract()  # rip it out
	# get text
	text = soup.get_text()
	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drp blank lines
	text = 'n'.join(chunk for chunk in chunks if chunk)
	input_text = text.split('.')

	vectorizer = joblib.load("../model/vectorizer.m")
	input_sentences = ''
	for line in input_text:
		x_new_input = vectorizer.transform([line]).toarray()
		model = FFN([1024, 2048, 1024, 512, 256], x_new_input.shape[1], 2)
		model.load_state_dict(torch.load('../model/description.pt'))
		model.eval()
		test_pred = model(torch.tensor(x_new_input).float())
		test_pred = torch.argmax(test_pred, dim=1, keepdim=False)
		if test_pred.item() == 1:
			input_sentences += str(line)

	clf = joblib.load("../model/func_classifier.m")
	x_new_input = vectorizer.transform([input_sentences]).toarray()
	result = clf.predict(x_new_input)
	rs1 = ''
	if result == [0]:
		print('This should be a DATA ANALYSIS project.')
		rs1 = 'This should be a DATA ANALYSIS project.'
	elif result == [1]:
		print('This should be a DATA MANAGEMENT project.')
		rs1 = 'This should be a DATA MANAGEMENT project.'
	elif result == [2]:
		print('This should be a DEEP LEARNING project.')
		rs1 = 'This should be a DEEP LEARNING project.'

	prob = clf.decision_function(x_new_input)[0]
	new_prob = []
	for x in prob:
		x = float(x - np.min(prob)) / (np.max(prob) - np.min(prob))
		new_prob.append(x)
	summury = sum(new_prob)
	new_prob = [x / summury for x in new_prob]
	new_prob_dict = {'Data Analysis': new_prob[0], 'Data Management': new_prob[1], 'Deep Learning': new_prob[2]}
	print('Chance to be DA:', new_prob[0])
	print('Chance to be DM:', new_prob[1])
	print('Chance to be DL:', new_prob[2])
	return rs1, new_prob_dict
