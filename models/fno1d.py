import torch.nn as nn
import torch.nn.functional as F

from .basic import SpectralConv1d


class FNO1d(nn.Module):
    def __init__(self, modes1,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=2, out_dim=1,
                 activation='tanh'):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, c)
        output: the solution 
        output shape: (batchsize, x=s, 1)
        """

        self.modes1 = modes1
        self.width = width

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, mode1_num)
            for in_size, out_size, mode1_num
            in zip(self.layers, self.layers[1:], self.modes1)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu_
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(
                batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.activation(x)
