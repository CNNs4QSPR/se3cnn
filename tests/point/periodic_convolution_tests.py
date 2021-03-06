# pylint: disable=C,E1101,E1102
import unittest

import torch
from functools import partial
from se3cnn.point.operations import PeriodicConvolution
from se3cnn.point.kernel import Kernel
from se3cnn.point.radial import ConstantRadialModel

class Tests(unittest.TestCase):
    def test1(self):
        Rs_in = [(2, 0), (0, 1), (2, 2)]
        Rs_out = [(2, 0), (2, 1), (2, 2)]
        K = partial(Kernel, RadialModel=ConstantRadialModel)
        m = PeriodicConvolution(K, Rs_in, Rs_out)

        n = sum(mul * (2 * l + 1) for mul, l in Rs_in)
        x = torch.randn(2, 3, n)
        g = torch.randn(2, 3, 3)

        import pymatgen
        lattice = pymatgen.Lattice.cubic(1.0)

        m(x, g, lattice, 2.0)


unittest.main()
