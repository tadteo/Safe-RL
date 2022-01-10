#!/usr/bin/env python3

import torch
from torch import nn 

import numpy as np
import math
import torch.distributions as pyd
import torch.nn.functional as F

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

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class ActorModel(nn.Module):

    def __init__(self, input_size, output_size, layers_sizes=[32,128,512,128,32], activation=nn.ReLU()):
        super(ActorModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = []
        # self.layers.append(nn.TransformerEncoderLayer(d_model=input_size, nhead=8, dim_feedforward=2048, dropout=0.1))
        first_layer = nn.Linear(in_features=input_size, out_features=layers_sizes[0], bias=False)
        
        self.layers.append(first_layer)
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
        # print(type(input))
        if(type(input) == np.ndarray):
            X = torch.from_numpy(input).float().to(self.device)
        else:
            X = input.clone().detach().float()
        print("Actor Model forward: ", type(input), X.shape)
        out = self.net(X)
        mu, log_std = out.chunk(2,dim=-1)
        log_std = torch.tanh(log_std)
        std_dev = log_std.exp()
        print("Mu: ", mu, "\nSigma: ", log_std)
        # log_std = torch.diag_embed(log_std**2)
        print(log_std)
        dist =SquashedNormal(mu, std_dev)
        
        # plotDistribution(dist,self.device)
        #print(mu.shape, log_std.shape)
        #log_prob = torch.tanh(mu + log_std*torch.randn(mu.shape,device=self.device))
        # 
        
        return dist #, log_prob
