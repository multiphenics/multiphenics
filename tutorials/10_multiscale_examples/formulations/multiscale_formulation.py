# Copyright (C) 2016-2021 by the multiphenics authors
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

import sys
import multiphenics as mp
import dolfin as df
sys.path.insert(0, '../core/')
from fenicsUtils import symgrad


class MultiscaleFormulation:

    def __init__(self, mesh, sigma, Eps, others):

        self.mesh = mesh
        self.others = others
        self.sigma = sigma
        self.Eps = Eps

        V = self.flutuationSpace()
        R = self.zeroAverageSpace()
        restrictions = [None, None] + self.otherRestrictions()
        W = [V, R] + self.otherSpaces()
        self.W = mp.BlockFunctionSpace(W, restrict=restrictions)

        self.uu = mp.BlockTrialFunction(self.W)
        self.vv = mp.BlockTestFunction(self.W)
        self.uu_ = mp.block_split(self.uu)
        self.vv_ = mp.block_split(self.vv)

    def __call__(self):
        return self.blocks() + self.bcs() + [self.W]

    def blocks(self):
        dx = df.Measure('dx', self.mesh)

        u, p = self.uu_[0:2]
        v, q = self.vv_[0:2]

        aa = [[df.inner(self.sigma(u), symgrad(v))*dx, df.inner(p, v)*dx],
              [df.inner(q, u)*dx(), 0]]

        # dot(sigma(Eps) , symgrad(v)) = dot(Eps , sigma(symgrad(v))
        ff = [-df.inner(self.Eps, self.sigma(v))*dx, 0]

        return [aa, ff]

    def bcs(self):
        return [[]]

    def enrichBlocks(self):
        pass

    def createMixedSpace(self):
        pass

    def flutuationSpace(self):
        return df.VectorFunctionSpace(self.mesh, "CG",
                                      self.others['polyorder'])

    def zeroAverageSpace(self):
        return df.VectorFunctionSpace(self.mesh, "Real", 0)

    def otherSpaces(self):
        return []

    def otherRestrictions(self):
        return []
