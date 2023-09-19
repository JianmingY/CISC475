import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class Model4layerMLP(nn.Module):

    def __init__(self,N_input = 784, N_Bottleneck = None, N_output=784):
        super(Model4layerMLP,self).__init__()
        N2 = 392
        self.fc1 = nn.Linear(N_input,N2)
        self.fc2 = nn.Linear(N2,N_Bottleneck)
        self.fc3 = nn.Linear(N_Bottleneck,N2)
        self.fc4 = nn.Linear(N2,N_output)
        self.type = 'MLP4'
        self.input_shape = (1,28*28)

    def forward(self, X):
        # encoder
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)

        # decoder
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)

        return X



# if __name__ == "__train__":
#     ####################################### random tensor test #########################################
#     print("Hello, World!")
    # # instantiate and try function
    #
    # model = Model4layerMLP()
    # X = torch.arange(0, 784)
    # X = X.type(torch.float32)
    # with torch.inference_mode():  # <- context manager, turns off gradient tracking and so to save time, memory
    #     y_preds = model(X)
    #
    # print(y_preds)
    # X = X.reshape(28, 28)
    # y_preds = y_preds.reshape(28, 28)
    # plt.imshow(X)
    # plt.imshow(y_preds)
    # plt.show()