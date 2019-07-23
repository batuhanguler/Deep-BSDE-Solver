import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Resnet(nn.Module):

    def __init__(self, layers, stable, activation):
        super(Resnet, self).__init__()

        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2])
        self.layer2_input = nn.Linear(in_features=layers[0], out_features=layers[2])
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3])
        self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
        self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
        self.layer4_input = nn.Linear(in_features=layers[0], out_features=layers[4])
        self.layer5 = nn.Linear(in_features=layers[4], out_features=layers[5])

        self.activation = activation

        self.epsilon = 0.01
        self.stable = stable

    def stable_forward(self, layer, out):  # Building block for the NAIS-Net
        weights = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        A = RtR + torch.eye(RtR.shape[0]).cuda() * self.epsilon

        return F.linear(out, -A, layer.bias)

    def forward(self, x):
        u = x

        out = self.layer1(x)
        out = self.activation(out)

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer2, out)
            out += self.layer2_input(u)
        else:
            out = self.layer2(out)
        out = self.activation(out)
        out += shortcut

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer3, out)
            out += self.layer3_input(u)
        else:
            out = self.layer3(out)
        out = self.activation(out)
        out += shortcut

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer4, out)
            out += self.layer4_input(u)
        else:
            out = self.layer4(out)

        out = self.activation(out)
        out += shortcut

        out = self.layer5(out)

        return out


class SDEnet(nn.Module):

    def __init__(self, layers, activation):
        super(SDEnet, self).__init__()

        self.layers = nn.ModuleList()
        self.brownian = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            if i > 0 and i < len(layers) - 2:
                self.brownian.append(nn.Linear(in_features=layers[i], out_features=1, bias=False))

        self.activation = activation
        self.epsilon = 1e-4
        self.h = 0.1

    def product(self, layer, out):
        weights = layer.weight
        RtR = torch.matmul(weights.t(), weights)
        A = RtR + torch.eye(RtR.shape[0]).cuda() * self.epsilon

        return F.linear(out, A, layer.bias)

    def forward(self, x):
        out = self.layers[0](x)
        out = self.activation(out)

        for i, layer in enumerate(self.layers[1:-1]):
            shortcut = out
            out = layer(out)
            out = shortcut + self.h * self.activation(out) + self.h ** (1 / 2) * self.product(self.brownian[i],
                                                                                              torch.rand_like(out))
            # out = shortcut + self.activation(out) + 0.4*torch.ones_like(out)*torch.rand_like(out)

        out = self.layers[-1](out)

        return out


class VerletNet(nn.Module):

    def __init__(self, layers, activation):
        super(VerletNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))

        self.h = 0.5
        self.activation = activation

    def transpose(self, layer, out):

        return F.linear(out, layer.weight.t(), layer.bias)

    def forward(self, x):

        out = self.layers[0](x)
        out = self.activation(out)

        z = torch.zeros_like(out)

        for layer in self.layers[1:-1]:
            shortcut = out
            out = self.transpose(layer, out)
            z = z - self.activation(out)
            out = layer(z)
            out = shortcut + self.activation(out)

        out = self.layers[-1](out)

        return out


class VerletNet(nn.Module):

    def __init__(self, layers, activation):
        super(VerletNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))

        self.h = 0.5
        self.activation = activation

    def transpose(self, layer, out):
        return F.linear(out, layer.weight.t(), layer.bias)

    def forward(self, x):

        out = self.layers[0](x)
        out = self.activation(out)

        z = torch.zeros_like(out)

        for layer in self.layers[1:-1]:
            shortcut = out
            out = self.transpose(layer, out)
            z = z - self.activation(out)
            out = layer(z)
            out = shortcut + self.activation(out)

        out = self.layers[-1](out)

        return out