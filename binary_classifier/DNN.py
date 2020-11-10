import torch.nn as nn
import torch
from collections import OrderedDict
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
            if (i == 1 or i == 3 or i == 5):
                model_arch.append(("dropout_" + str(i), nn.Dropout()))
            unit = num
        model_arch.append(("dense_final", nn.Linear(unit, self.output_size, bias=self.bias)))
        model_arch.append(("act_final", nn.Sigmoid()))
        self.model = nn.Sequential(OrderedDict(model_arch))

    def forward(self, inputs):
        return self.model(inputs)