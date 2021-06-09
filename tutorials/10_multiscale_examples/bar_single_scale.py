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
Bar problem given a constitutive law (single-scale):
Problem in [0,Lx]x[0,Ly], homogeneous dirichlet on left and traction on the
right. We use an isotropic linear material, given two lamé parameters.

Contributed by: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch
"""
import sys
import dolfin as df
import numpy as np
from ufl import nabla_div
sys.path.insert(0, '../core/')
from fenicsUtils import symgrad

Lx = 2.0
Ly = 0.5
Nx = int(sys.argv[1])
Ny = int(sys.argv[2])

facAvg = 6.0  # roughly to approximate single scale to mulsticale results
lamb = facAvg*1.0
mu = facAvg*0.5
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

# Define boundary condition
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)

ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
traction = df.Constant((0.0, ty))


def sigma(u):
    return lamb*nabla_div(u)*df.Identity(2) + 2*mu*symgrad(u)


# Define variational problem
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)
a = df.inner(sigma(uh), df.grad(vh))*df.dx
b = df.inner(traction, vh)*ds(2)

# Compute solution
uh = df.Function(Uh)

# linear_solver ops: "superlu" or "mumps"
df.solve(a == b, uh, bcs=bcL, solver_parameters={"linear_solver": "superlu"})


def test_norm():
    # using Nx = 100 , Ny = 50
    assert np.abs(np.linalg.norm(uh.vector().get_local()[:]) -
                  5.285161122867508) < 1e-14


# Save solution in VTK format
fileResults = df.XDMFFile("bar_single_scale.xdmf")
fileResults.write(uh)
