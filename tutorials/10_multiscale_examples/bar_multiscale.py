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
Bar problem in [0, Lx] x [0, Ly], with homogeneous dirichlet on left and
traction on the right.
Comparison between single scale and multiscale constitutive laws:
* for the single scale constitutive law we use an isotropic linear material,
  given two Lamé parameters.
* the multiscale constitutive law (with random microstructures) is given implicitly
  by solving a micro problem in each gauss point of micro-scale (one per element).
  We can choose the kinematically constrained model to the micro problem:
  Linear, Periodic or Minimally Restricted.
  Entries to constitutive law: Mesh (micro), Lamé parameters (variable in micro),
  Kinematic Model.

When run on more then one core, this is a parallel implementation. Note that:
* random seed in the multiscale constitutive law is chosen to match the global cell index.
* each local problem is solved on one process, only the macro-scale mesh is partitioned.

Contributed by: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com,
felipe.figueredorocha@epfl.ch
"""

import numpy as np
import ufl
import dolfin as df
from mpi4py import MPI
from models import MicroConstitutiveModel
from utils import symgrad, symgrad_voigt

# Geometry definitions
Lx = 2.0
Ly = 0.5
Nx = 12
Ny = 4

# Create mesh (to be partitioned in parallel) and define function space
mesh = df.RectangleMesh(MPI.COMM_WORLD, df.Point(0.0, 0.0), df.Point(Lx, Ly),
                        Nx, Ny, "right/left")
Uh = df.VectorFunctionSpace(mesh, "Lagrange", 1)

# Define subdomains and markers for boundary conditions
left_bnd = df.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
right_bnd = df.CompiledSubDomain("near(x[0], Lx) && on_boundary", Lx=Lx)

boundary_markers = df.MeshFunction("size_t", mesh, dim=1, value=0)
left_bnd.mark(boundary_markers, 1)
right_bnd.mark(boundary_markers, 2)

# Define boundary conditions
bcL = df.DirichletBC(Uh, df.Constant((0.0, 0.0)), boundary_markers, 1)
ty = -0.01
traction = df.Constant((0.0, ty))

# Trial and test functions
uh = df.TrialFunction(Uh)
vh = df.TestFunction(Uh)

# Define measures
dx = df.dx
ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# ~~~ PART I: single scale constitutive law ~~~ #

# Define single scale constitutive parameters
fac_avg = 4.0  # roughly to approximate single scale to mulsticale results
lamb = fac_avg * 1.0
mu = fac_avg * 0.5

# Define single scale constitutive law
def sigma(u):
    return lamb * ufl.nabla_div(u) * df.Identity(2) + 2 * mu * symgrad(u)

# Define single scale variational problem
a_single_scale = df.inner(sigma(uh), df.grad(vh)) * dx
f_single_scale = df.inner(traction, vh) * ds(2)

# Compute single scale solution
uh_single_scale = df.Function(Uh)
df.solve(a_single_scale == f_single_scale, uh_single_scale, bcs=bcL,
         solver_parameters={"linear_solver": "mumps"})

# Save single scale solution in XDMF format
file_results = df.XDMFFile("bar_single_scale.xdmf")
file_results.write(uh_single_scale)

# ~~~ PART II: multiscale constitutive law ~~~ #

# Define the mesh of the micro model. Note that such mesh is associated to the current processor only
# and not partitioned across multiple processors.
Nx_micro = Ny_micro = 50
Lx_micro = Ly_micro = 1.0
mesh_micro = df.RectangleMesh(MPI.COMM_SELF, df.Point(0.0, 0.0), df.Point(Lx_micro, Ly_micro),
                              Nx_micro, Ny_micro, "right/left")

# Auxiliary function to generate a random structure
def getFactorBalls(seed=1):
    Nballs = 4
    ellipse_data = np.zeros((Nballs, 3))
    xlin = np.linspace(0.0 + 0.5/np.sqrt(Nballs), 1.0 - 0.5/np.sqrt(Nballs),
                       int(np.sqrt(Nballs)))

    grid = np.meshgrid(xlin, xlin)
    ellipse_data[:, 0] = grid[0].flatten()
    ellipse_data[:, 1] = grid[1].flatten()
    np.random.seed(seed)
    r0 = 0.3
    r1 = 0.5
    ellipse_data[:, 2] = r0 + np.random.rand(Nballs)*(r1 - r0)

    contrast = 10.0
    radius_threshold = 0.01
    str_fac = "A*exp(-a*((x[0] - x0)*(x[0] - x0) + (x[1] - y0)*(x[1] - y0) ))"
    fac = df.Expression("1.0", degree=2)  # ground substance
    for xi, yi, ri in ellipse_data[:, 0:3]:
        fac = fac + df.Expression(str_fac, A=contrast - 1.0,
                                  a=- np.log(radius_threshold)/ri**2,
                                  x0=xi, y0=yi, degree=2)

    return fac

# Generate a random (but reproducible) structure ruled by the global cell index
facs = [getFactorBalls(c.global_index()) for c in df.cells(mesh)]

# Define multiscale constitutive parameters
lamb_matrix = 1.0
mu_matrix = 0.5
params = [[fac_i * lamb_matrix, fac_i * mu_matrix] for fac_i in facs]

# Auxiliary class to evaluate the expression of the C^{hom} tensor
class ChomExpression(df.UserExpression):
    def __init__(self, micro_models,  **kwargs):
        self.micro_models = micro_models
        super().__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        values[:] = self.micro_models[cell.index].get_tangent().flatten()

    def value_shape(self):
        return (3, 3)

# Loop over possible boundary models implemented in our formulations:
all_uh_multiscale = dict()
for boundary_model in ("per", "lin", "MR", "lag"):
    print("Boundary model:", boundary_model)

    # Define multiscale constitutive law
    micro_models = [MicroConstitutiveModel(mesh_micro, pi, boundary_model) for pi in params]
    Chom = ChomExpression(micro_models, degree=0)

    # Define multiscale variational problem
    a_multiscale = df.inner(df.dot(Chom, symgrad_voigt(uh)), symgrad_voigt(vh)) * dx
    f_multiscale = df.inner(traction, vh) * ds(2)

    # Compute multiscale solution
    uh_multiscale = df.Function(Uh)
    df.solve(a_multiscale == f_multiscale, uh_multiscale, bcL,
             solver_parameters={"linear_solver": "mumps"})
    all_uh_multiscale[boundary_model] = uh_multiscale

    # Save single scale solution in XDMF format
    file_results = df.XDMFFile(MPI.COMM_WORLD, "bar_multiscale_" + boundary_model +  ".xdmf")
    file_results.write(uh_multiscale)


# ~~~ PART III: compare single scale and multiscale results ~~~ #

# Compute the error between single scale and multiscale solutions
error = dict()
for (boundary_model, uh_multiscale) in all_uh_multiscale.items():
    eh = df.Function(Uh)
    eh.vector().set_local(uh_multiscale.vector().get_local()[:]
                          - uh_single_scale.vector().get_local()[:])
    eh.vector().apply("")
    error[boundary_model] = df.norm(eh)

# The multiscale solutions associated to "lin" and "lag" models should match
assert np.isclose(error["lin"], error["lag"])

# Compare the errors between single scale and multiscale solutions to reference values
assert np.isclose(error["per"], 0.006319071443377064)
assert np.isclose(error["lin"], 0.007901978018038507)
assert np.isclose(error["MR"], 0.0011744201374588351)
