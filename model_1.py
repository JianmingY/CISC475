import torch
from torch import nn
import torch.nn.functional as F


class MPL_3_layer_Classifier(nn.Module):

    def __init__(self, N_input = 784, N_tanh_activate = None, N_class = 10,p = 0.3):
        super(MPL_3_layer_Classifier,self).__init__()
        self.f1 = nn.Linear(N_input,N_tanh_activate)
        self.f2 = nn.Linear(N_tanh_activate,N_class)
        self.linear_tanh_classify = nn.Sequential(
            nn.Flatten(),
            self.f1,
            nn.Tanh(),
            self.f2,
            nn.Dropout(p)
            # nn.Softmax()
        )

    def forward(self, X):
        X = self.linear_tanh_classify(X)
        return X

