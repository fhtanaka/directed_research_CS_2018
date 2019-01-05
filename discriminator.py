import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, in_features, leakyRelu=0.2, dropout=0.3, hidden_layers=[1024, 512, 256]):
        super(DiscriminatorNet, self).__init__()
        
        out_features = 1
        self.layers = hidden_layers.copy()
        self.layers.insert(0, in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu),
                    nn.Dropout(dropout)
                )
            )
        
        self.add_module("out", 
            nn.Sequential(
                nn.Linear(self.layers[-1], out_features),
                torch.nn.Sigmoid()
            )
        )

    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x
