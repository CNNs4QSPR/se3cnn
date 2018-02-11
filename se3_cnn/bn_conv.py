# pylint: disable=C,R,E1101
import torch
from se3_cnn import SE3Convolution
from se3_cnn import SE3BatchNorm
from se3_cnn import SO3


class SE3BNConvolution(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3BatchNorm followed by SE3Convolution
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, eps=1e-5, momentum=0.1, mode='normal', **kwargs):
        super().__init__()
        self.bn = SE3BatchNorm([(m, SO3.dim(R)) for m, R in Rs_in], eps=eps, momentum=momentum, mode=mode)
        self.conv = SE3Convolution(Rs_in=Rs_in, Rs_out=Rs_out, size=size, radial_window_dict=radial_window_dict, **kwargs)

    def forward(self, input):  # pylint: disable=W
        self.bn.update_statistics(input.data)

        # Instead of rescale the input with the running_var
        # We rescale the weights that are much smaller (memory)

        ws = []
        weight_index = 0
        for i, mi in enumerate(self.conv.combination.multiplicities_out):
            var_index = 0
            for j, mj in enumerate(self.conv.combination.multiplicities_in):
                if self.conv.combination.kernels[i][j] is not None:
                    b_el = self.conv.combination.kernels[i][j].size(0)

                    factor = 1 / (self.bn.running_var[var_index: var_index + mj] + self.bn.eps) ** 0.5
                    var_index += mj

                    w = self.conv.weight[weight_index: weight_index + mi * mj * b_el]
                    weight_index += mi * mj * b_el

                    w = w.view(mi, mj, b_el) * torch.autograd.Variable(factor).view(1, -1, 1)
                    ws.append(w.view(-1))

        kernel = self.conv.combination(torch.cat(ws))
        return torch.nn.functional.conv3d(input, kernel, **self.conv.kwargs)


class SE3ConvolutionBN(torch.nn.Module):
    '''
    This class exists to optimize memory consumption.
    It is simply the concatenation of two operations:
    SE3Convolution followed by SE3BatchNorm
    '''

    def __init__(self, Rs_in, Rs_out, size, radial_window_dict, eps=1e-5, momentum=0.1, mode='normal', **kwargs):
        super().__init__()
        self.conv = SE3Convolution(Rs_in=Rs_in, Rs_out=Rs_out, size=size, radial_window_dict=radial_window_dict, **kwargs)
        self.bn = SE3BatchNorm([(m, SO3.dim(R)) for m, R in Rs_out], eps=eps, momentum=momentum, mode=mode)

    def forward(self, input):  # pylint: disable=W
        # Instead of rescale the output with the running_var
        # We rescale the weights that are much smaller (memory)

        ws = []
        weight_index = 0
        var_index = 0
        for i, mi in enumerate(self.conv.combination.multiplicities_out):
            factor = 1 / (self.bn.running_var[var_index: var_index + mi] + self.bn.eps) ** 0.5
            var_index += mi

            for j, mj in enumerate(self.conv.combination.multiplicities_in):
                if self.conv.combination.kernels[i][j] is not None:
                    b_el = self.conv.combination.kernels[i][j].size(0)

                    w = self.conv.weight[weight_index: weight_index + mi * mj * b_el]
                    weight_index += mi * mj * b_el

                    w = w.view(mi, mj, b_el) * torch.autograd.Variable(factor).view(-1, 1, 1)
                    ws.append(w.view(-1))

        kernel = self.conv.combination(torch.cat(ws))
        output = torch.nn.functional.conv3d(input, kernel, **self.conv.kwargs)

        self.bn.update_statistics(output.data)

        return output
