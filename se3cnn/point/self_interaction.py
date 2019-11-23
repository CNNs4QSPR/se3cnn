# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
from functools import reduce
from collections import defaultdict

import torch

from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel


def simplify_Rs(Rs):
    # return simplifed Rs and transformation matrix
    # currently ignores parity
    try:
        mults, Ls, ps = zip(*Rs)
    except:
        mults, Ls = zip(*Rs)
    totals = [mult * (2 * L + 1) for mult, L, p in Rs]
    shuffle = torch.zeros(sum(totals), sum(totals))
    
    # Get total number of multiplicities by L
    d = defaultdict(int)
    for mult, L, p in Rs:
        d[L] += mult
        
    # Rs_new grouped and sorted by L
    Rs_new = sorted([x[::-1] for x in d.items()], key=lambda x: x[1])
    new_mults, new_Ls = zip(*Rs_new)
    new_totals = [mult * (2 * L + 1) for mult, L in Rs_new]
    
    # indices for different mults
    tot_indices = [[sum(totals[0:i]), sum(totals[0:i + 1])] for i in range(len(totals))]
    new_tot_indices = [[sum(new_totals[0:i]), sum(new_totals[0:i + 1])] for i in range(len(new_totals))]

    # group indices by L
    d_t_i = defaultdict(list)
    for L, index in zip(Ls, tot_indices):
        d_t_i[L].append(index)
    
    # 
    total_bounds = sorted(list(d_t_i.items()), key=lambda x: x[0])
    new_total_bounds = list(zip(new_Ls, new_tot_indices))

    for old_indices, (L, new_index) in zip(total_bounds, new_total_bounds):
        old_indices_list = [torch.arange(i[0], i[1]) for i in old_indices[1]]
        new_index_list = torch.arange(new_index[0], new_index[1])
        shuffle[new_index_list, torch.cat(old_indices_list)] = 1
    
    return Rs_new, shuffle

class SortSphericalSignals(torch.nn.Module):
    def __init__(self, Rs):
        super().__init__()
        ljds = []

        j = 0
        for mul, l in Rs:
            d = mul * (2 * l + 1)
            ljds.append((l, j, d))
            j += d

        mixing_matrix = torch.zeros(j, j)

        i = 0
        for _l, j, d in sorted(ljds):
            mixing_matrix[i:i+d, j:j+d] = torch.eye(d)
            i += d

        self.register_buffer('mixing_matrix', mixing_matrix)

    def forward(self, x):
        """
        :param x: tensor [batch, feature, ...]
        """
        output = torch.einsum('ij,zja->zia', (self.mixing_matrix, x.flatten(2))).contiguous()
        return output.view(*x.size())


class ConcatenateSphericalSignals(torch.nn.Module):
    def __init__(self, *Rs):
        super().__init__()
        Rs = reduce(list.__add__, Rs, [])
        self.sort = SortSphericalSignals(Rs)

    def forward(self, *signals):
        combined = torch.cat(signals, dim=1)
        return self.sort(combined)


class SelfInteraction(torch.nn.Module):
    def __init__(self, Rs_in, Rs_out):
        super().__init__()
        self.kernel = Kernel(Rs_in, Rs_out, ConstantRadialModel,
                             get_l_filters=lambda l_in, l_out: [0] if l_in ==
                             l_out else [])

    def forward(self, features):
        """
        :param features: tensor [..., channel]
        :return:         tensro [..., channel]
        """
        *size, n = features.size()
        features = features.view(-1, n)

        k = self.kernel(features.new_zeros(features.size(0), 3))
        features = torch.einsum("zij,zj->zi", (k, features))
        features = features.view(*size, -1)
        return features


class ConvolutionPlusSelfInteraction(torch.nn.Module):
    def __init__(self, Convolution, SelfInteraction, Rs_in, Rs_out):
        super().__init__()
        self.convolution = Convolution(Rs_in, Rs_out)
        self.selfinteraction = SelfInteraction(Rs_in, Rs_out)

    def forward(self, input, geometry, n_norm=1):
        output_conv = self.convolution(input, geometry, n_norm)
        output_si = self.selfinteraction(input)
        return output_conv + output_si


class ApplyKernelPlusSelfInteraction(torch.nn.Module):
    def __init__(self, ApplyKernel, SelfInteraction, Rs_in, Rs_out):
        super().__init__()
        self.applykernel = ApplyKernel(Rs_in, Rs_out)
        self.selfinteraction = SelfInteraction(Rs_in, Rs_out)

    def forward(self, input, geometry, _n_norm=1):
        output_applykernel = self.applykernel(input, geometry, _n_norm)  # zabi
        output_si = self.selfinteraction(input)  # zai
        batch, N, _ = output_si.shape
        I = torch.eye(N, device=input.device).unsqueeze(0).unsqueeze(-1)
        return output_applykernel + I * output_si.unsqueeze(-2)  # zabi
