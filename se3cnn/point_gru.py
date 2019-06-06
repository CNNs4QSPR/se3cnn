import torch
import se3cnn.SO3 as SO3
import se3cnn.point_kernel_vec as pv
import se3cnn.self_interaction as si
import se3cnn.non_linearities as nl


class TFGRU(torch.nn.Module):
    def __init__(self, Rs_hidden):
        super().__init__()

        self.Rs_hidden = Rs_hidden
        self.Rs_hidden_dims = [m * (2 * l + 1) for m, l in Rs_hidden]
        self.Rs_hidden_double = [(2*m, l) for m, l in Rs_hidden]
        dims = [2 * l + 1 for m, l in Rs_hidden for i in range(m)]

        # Nonlinearities
        self.nl_sig_z = nl.NormActivation(
            dims, torch.nn.Sigmoid(), torch.nn.Sigmoid())
        self.nl_sig_r = nl.NormActivation(
            dims, torch.nn.Sigmoid(), torch.nn.Sigmoid())
        self.nl_tanh = nl.NormActivation(
            dims, torch.nn.Tanh(), torch.nn.Tanh())

        # Self-interactions
        self.self_inter_z = si.SelfInteraction(
            self.Rs_hidden_double, self.Rs_hidden)
        self.self_inter_r1 = si.SelfInteraction(self.Rs_hidden, self.Rs_hidden)
        self.self_inter_r2 = si.SelfInteraction(self.Rs_hidden, self.Rs_hidden)
        self.self_inter_pre_h_conv = si.SelfInteraction(
            self.Rs_hidden_double, self.Rs_hidden)

        # Convolutional kernel
        rbf = pv.gaussian_rbf
        radii = torch.linspace(0, 1, 3)
        self.h_conv = pv.SE3PointKernel(
            self.Rs_hidden, self.Rs_hidden, radii, rbf, J_filter_max=2)  # dcnba

    def forward(self, x, h_nei, h_nei_diff, mask):
        """
        x: embedded input node signal. Shape ndb.
        h_nei: h signal for neighbors. Shape nca.
        h_nei_diff: difference vector between input node and neighbors. Shape nba3.
        mask: mask for real neighbors for each example in batch. Shape nba.
        """
        # Convolution of h_nei using relative difference vectors
        batch = x.shape[0]  # [batch, channel, point == 1]
        h_conv = torch.einsum('nca,dcnba,nba->ndba',
                              (h_nei, self.h_conv(h_nei_diff), mask))
        sum_h_conv = h_conv.sum(dim=-1)  # ndb

        # Update gate
        # Concatenate x and sum_h_conv by rep -- this is easier if they are the same size
        x_split = torch.split(x, self.Rs_hidden_dims, -2)
        sum_h_conv_split = torch.split(sum_h_conv, self.Rs_hidden_dims, -2)
        z_input = [None] * len(self.Rs_hidden_dims) * 2
        z_input[::2] = x_split
        z_input[1::2] = sum_h_conv_split
        z_input = torch.cat(z_input, -2)
        z = self.nl_sig_z(self.self_inter_z(z_input))

        # Reset gate
        r_1 = self.self_inter_r1(x)  # ndb
        h_conv_shape = h_conv.shape
        h_conv_reshape = h_conv.view(*list(h_conv.shape[:-2]) + [-1])
        r_2 = self.self_inter_r2(h_conv_reshape).view(h_conv_shape)  # ndba
        r = self.nl_sig_r(r_1.unsqueeze(-2) + r_2)  # Rs_hidden

        # New h
        gated_h_conv = gate_normwise_multiply(
            r, h_conv, self.Rs_hidden, rep_dim=-3)
        sum_gated_h_conv = gated_h_conv.sum(dim=-1)
        sum_gated_h_conv_split = torch.split(
            sum_gated_h_conv, self.Rs_hidden_dims, -2)
        h_conv_input = [None] * len(self.Rs_hidden_dims) * 2
        h_conv_input[::2] = x_split
        h_conv_input[1::2] = sum_gated_h_conv_split

        h_conv_input = torch.cat(h_conv_input, -2)
        pre_h_conv = self.nl_tanh(self.self_inter_pre_h_conv(
            h_conv_input))  # [batch, Rs_hidden, 1]
        new_h = sum_h_conv + gate_normwise_multiply(z, pre_h_conv + sum_h_conv,
                                                    self.Rs_hidden, rep_dim=-2)  # [batch, Rs_hidden, 1]
        return new_h


def gate_normwise_multiply(gate_tensor, target_tensor, Rs, rep_dim=-2):
    Rs_dims = [2 * l + 1 for m, l in Rs for i in range(m)]
    gate_split = torch.split(gate_tensor, Rs_dims, dim=rep_dim)
    gate_norm = [x.norm(2, dim=rep_dim, keepdim=True) for x in gate_split]
    gate_norm = torch.cat(gate_norm, dim=rep_dim)
    scaffold = target_tensor.new_zeros(len(Rs_dims), sum(Rs_dims))
    start = 0
    for i, dim in enumerate(Rs_dims):
        scaffold[i, start:start + dim] = 1.0
        start += dim
    if rep_dim == -2:
        return torch.einsum('nda,dc,nca->nca', (gate_norm, scaffold, target_tensor))
    elif rep_dim == -3:
        return torch.einsum('ndba,dc,ncba->ncba', (gate_norm, scaffold, target_tensor))
