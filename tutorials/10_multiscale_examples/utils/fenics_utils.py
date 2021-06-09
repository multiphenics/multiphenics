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

import dolfin as df
import numpy as np


def symgrad(v): return df.sym(df.nabla_grad(v))


def symgrad_voigt(v):
    return df.as_vector([v[0].dx(0), v[1].dx(1), v[0].dx(1) + v[1].dx(0)])


def macro_strain(i):
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.],
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])


def stress2Voigt(s):
    return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])


def strain2Voigt(e):
    return df.as_vector([e[0, 0], e[1, 1], 2*e[0, 1]])


def Integral(u, dx, shape):
    n = len(shape)
    valueIntegral = np.zeros(shape)

    if(n == 1):
        for i in range(shape[0]):
            valueIntegral[i] = df.assemble(u[i]*dx)

    elif(n == 2):
        for i in range(shape[0]):
            for j in range(shape[1]):
                valueIntegral[i, j] = df.assemble(u[i, j]*dx)

    return valueIntegral
