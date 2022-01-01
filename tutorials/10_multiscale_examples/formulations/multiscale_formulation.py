# Copyright (C) 2016-2022 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

import ufl
import dolfin as df
import multiphenics as mp
from utils import symgrad


class MultiscaleFormulation(object):

    def __init__(self, mesh, sigma, Eps, others):
        self.mesh = mesh
        self.others = others
        self.sigma = sigma
        self.Eps = Eps

        V = self.fluctuation_space()
        R = self.zero_average_space()
        restrictions = [None, None] + self.other_restrictions()
        W = [V, R] + self.other_spaces()
        self.W = mp.BlockFunctionSpace(W, restrict=restrictions)

        self.uu = mp.BlockTrialFunction(self.W)
        self.vv = mp.BlockTestFunction(self.W)
        self.uu_ = mp.block_split(self.uu)
        self.vv_ = mp.block_split(self.vv)

    def __call__(self):
        return self.blocks() + self.bcs() + [self.W]

    def blocks(self):
        dx = ufl.Measure("dx", self.mesh)

        u, p = self.uu_[0:2]
        v, q = self.vv_[0:2]

        aa = [[ufl.inner(self.sigma(u), symgrad(v)) * dx, ufl.inner(p, v) * dx],
              [ufl.inner(q, u) * dx, 0]]

        # Notice that dot(sigma(Eps), symgrad(v)) = dot(Eps, sigma(symgrad(v)))
        ff = [-ufl.inner(self.Eps, self.sigma(v)) * dx, 0]

        return [aa, ff]

    def bcs(self):
        return [[]]

    def fluctuation_space(self):
        return df.VectorFunctionSpace(self.mesh, "CG", self.others["polyorder"])

    def zero_average_space(self):
        return df.VectorFunctionSpace(self.mesh, "Real", 0)

    def other_spaces(self):
        return []

    def other_restrictions(self):
        return []
