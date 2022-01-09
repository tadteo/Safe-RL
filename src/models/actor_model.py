#!/usr/bin/env python3

import torch
from torch import nn 

import numpy as np
 
def plotDistribution(distribution,device):
    """
    Plot the distribution of the model.
    """
    import matplotlib.pyplot as plt
    #Create grid and multivariate normal
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    
    # Pack X and Y into a single 3-dimensional array
    pos = torch.empty(X.shape + (2,)).to(device)
    pos[:, :, 0] = torch.from_numpy(X)
    pos[:, :, 1] = torch.from_numpy(Y)

    # The distribution on the variables X, Y packed into pos.
    
    Z = distribution.log_prob(pos).exp().cpu().detach().numpy()
    
    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap="viridis")

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap="viridis")

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
    plt.show()

class ActorModel(nn.Module):

    def __init__(self, input_size, output_size, layers_sizes=[32,128,512,128,32], activation=nn.ReLU()):
        super(ActorModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        # self.layers.append(nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=2048, dropout=0.1))
        self.layers.append(nn.Linear(input_size, layers_sizes[0]))
        self.layers.append(activation)
        for i in range(1,len(layers_sizes)):
            self.layers.append(nn.Linear(layers_sizes[i-1],layers_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(layers_sizes[-1],2*output_size))

        self.layers.append(nn.Tanh())
        self.net = nn.Sequential(*self.layers)
        
        self.outputs = dict()
        
        # print("Actor Model structure: ", self.net, "\n\n")
        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    def forward(self, input):
        
        # print("Actor Model forward: ", input.shape)
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.float()
        # X = torch.stack([X,X,X],dim=1)
        # print("Actor Model forward: ", type(input), X.shape)
        mu, sigma = self.net(X).chunk(2,dim=-1)
        # print("Mu: ", mu, "\nSigma: ", sigma)
        sigma = torch.diag_embed(sigma**2)
        
        dist = torch.distributions.MultivariateNormal(mu, sigma)
        
        # plotDistribution(dist,self.device)
        
        
        return dist
