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


url = input()
response = urllib.request.urlopen(url)
page = response.read()
soup = BeautifulSoup(page, "lxml")
# kill all script and style elements
for script in soup(["script", "style"]):
   script.extract()    # rip it out
# get text
text = soup.get_text()
# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drp blank lines
text = 'n'.join(chunk for chunk in chunks if chunk)
input_text = text.split('.')


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