import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, out_features, leakyRelu=0.2, hidden_layers=[256, 512, 1024], in_features=100, escalonate=False):
        super(GeneratorNet, self).__init__()
        
        self.layers = hidden_layers.copy()
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu)
                )
            )

        if escalonate:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features),
                    nn.Tanh()
                )
            )
        else:
            self.add_module("out", 
                nn.Sequential(
                    nn.Linear(self.layers[-1], out_features)
                )
            )
    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x
