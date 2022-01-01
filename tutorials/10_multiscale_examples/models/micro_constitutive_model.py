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

import numpy as np
import ufl
import dolfin as df
import multiphenics as mp
from timeit import default_timer as timer

from formulations import (
    FormulationDirichletLagrange, FormulationLinear, FormulationMinimallyConstrained, FormulationPeriodic)
from utils import symgrad

list_multiscale_models = {
    "MR": FormulationMinimallyConstrained,
    "per": FormulationPeriodic,
    "lin": FormulationLinear,
    "lag": FormulationDirichletLagrange
}


class MicroConstitutiveModel(object):

    def __init__(self, mesh, lame, model):
        def sigma_law(u):
            return lame[0] * ufl.nabla_div(u) * ufl.Identity(2) + 2 * lame[1] * symgrad(u)

        self.sigma_law = sigma_law

        self.mesh = mesh
        self.model = model
        self.coord_min = np.min(self.mesh.coordinates(), axis=0)
        self.coord_max = np.max(self.mesh.coordinates(), axis=0)

        # it should be modified before computing tangent (if needed)
        self.others = {
            "polyorder": 1,
            "x0": self.coord_min[0], "x1": self.coord_max[0],
            "y0": self.coord_min[1], "y1": self.coord_max[1]
        }

        self.multiscale_model = list_multiscale_models[model]
        self.x = ufl.SpatialCoordinate(self.mesh)
        self.ndim = 2
        self.nvoigt = int(self.ndim*(self.ndim + 1)/2)
        self.Chom_ = None  # will be computed by get_tangent

    def get_tangent(self):
        if self.Chom_ is None:
            self.Chom_ = np.zeros((self.nvoigt, self.nvoigt))

            dy = ufl.Measure("dx", self.mesh)
            vol = df.assemble(df.Constant(1.0) * dy)
            y = ufl.SpatialCoordinate(self.mesh)
            Eps = df.Constant(((0., 0.), (0., 0.)))  # just placeholder

            form = self.multiscale_model(self.mesh, self.sigma_law, Eps, self.others)
            a, f, bcs, W = form()

            start = timer()
            A = mp.block_assemble(a)
            if len(bcs) > 0:
                bcs.apply(A)

            # decompose just once, since the matrix A is the same in every solve
            solver = df.PETScLUSolver("superlu")
            sol = mp.BlockFunction(W)

            end = timer()
            print("time assembling system", end - start)

            for i in range(self.nvoigt):
                start = timer()
                Eps.assign(df.Constant(self.macro_strain(i)))
                F = mp.block_assemble(f)
                if len(bcs) > 0:
                    bcs.apply(F)

                solver.solve(A, sol.block_vector(), F)
                sol.block_vector().block_function().apply("to subfunctions")

                sig_mu = self.sigma_law(df.dot(Eps, y) + sol[0])
                sigma_hom = self.integrate(sig_mu, dy, (2, 2))/vol

                self.Chom_[:, i] = sigma_hom.flatten()[[0, 3, 1]]

                end = timer()
                print("time in solving system", end - start)

        return self.Chom_

    @staticmethod
    def macro_strain(i):
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[i] = 1
        return np.array([[Eps_Voigt[0], Eps_Voigt[2] / 2.],
                        [Eps_Voigt[2] / 2., Eps_Voigt[1]]])

    @staticmethod
    def integrate(u, dx, shape):
        n = len(shape)
        valueIntegral = np.zeros(shape)

        assert n in (1, 2)
        if n == 1:
            for i in range(shape[0]):
                valueIntegral[i] = df.assemble(u[i]*dx)
        elif n == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    valueIntegral[i, j] = df.assemble(u[i, j]*dx)

        return valueIntegral
