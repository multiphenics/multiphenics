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

"""
Bar problem given a Multiscale constitutive law (random microstructures):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and
traction on the right. The constitutive law is given implicitly by solving a
micro problem in each gauss point of micro-scale (one per element).
We can choose the kinematically constrained model to the
micro problem: Linear, Periodic or Minimally Restricted. Entry to constitutive
law: Mesh (micro), Lamé parameters (variable in micro), Kinematical Model
PS: This is a single core implementation.

Contributed by: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch
"""

import sys
import dolfin as df
import numpy as np
sys.path.insert(0, '../core/')
import micro_constitutive_model as mscm
from fenicsUtils import symgrad_voigt


class myChom(df.UserExpression):
    def __init__(self, microModels,  **kwargs):
        self.microModels = microModels
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[:] = self.microModels[cell.index].getTangent().flatten()

    def value_shape(self):
        return (3, 3,)


def getFactorBalls(seed=1):
    Nballs = 4
    ellipseData = np.zeros((Nballs, 3))
    xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs), 1.0 - 0.5/np.sqrt(Nballs),
                       int(np.sqrt(Nballs)))

    grid = np.meshgrid(xlin, xlin)
    ellipseData[:, 0] = grid[0].flatten()
    ellipseData[:, 1] = grid[1].flatten()
    np.random.seed(seed)
    ellipseData[:, 2] = r0 + np.random.rand(Nballs)*(r1 - r0)

    fac = df.Expression('1.0', degree=2)  # ground substance
    radiusThreshold = 0.01

    str_fac = 'A*exp(-a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ))'

    for xi, yi, ri in ellipseData[:, 0:3]:
        fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                                  a=- np.log(radiusThreshold)/ri**2,
                                  x0=xi, y0=yi, degree=2)

    return fac


r0 = 0.3
r1 = 0.5
Lx = 2.0
Ly = 0.5
Nx = int(sys.argv[1])
Ny = int(sys.argv[2])

lamb_matrix = 1.0
mu_matrix = 0.5
NxMicro = NyMicro = int(sys.argv[3])
LxMicro = LyMicro = 1.0
contrast = 10.0

ty = -0.01

# Create mesh and define function space
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")
Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

leftBnd = df.CompiledSubDomain('near(x[0], 0.0) && on_boundary')
rightBnd = df.CompiledSubDomain('near(x[0], Lx) && on_boundary', Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
leftBnd.mark(boundary_markers, 1)
rightBnd.mark(boundary_markers, 2)

# defining the micro model
nCells = mesh.num_cells()
meshMicro = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(LxMicro, LyMicro),
                             NxMicro, NyMicro, "right/left")

facs = [getFactorBalls(i) for i in range(nCells)]
params = [[fac_i*lamb_matrix, fac_i*mu_matrix] for fac_i in facs]
microModels = [mscm.MicroConstitutiveModel(meshMicro, pi, 'per')
               for pi in params]

Chom = myChom(microModels, degree=0)

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))

# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(df.dot(Chom, symgrad_voigt(uh)), symgrad_voigt(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)
df.solve(a == b, uh, bcL)

# Save solution in VTK format
fileResults = df.XDMFFile("bar_multiscale_standalone.xdmf")
fileResults.write(uh)

# # plot microstructures
# fac_h = df.project(facs[0], df.FunctionSpace(meshMicro, 'CG', 1))
# fac_h.rename("constrast", "label")
# df.plot(df.project(facs[0], df.FunctionSpace(meshMicro, 'CG', 1)))
# fileResults_2 = df.XDMFFile("microstructure_2.xdmf")
# fileResults_2.write(fac_h)
